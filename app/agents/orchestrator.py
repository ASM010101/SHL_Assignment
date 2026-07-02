"""
Main Agent Orchestrator for SHL Assessment Recommender.

Coordinates the intent detection, profile analysis, planning, retrieval,
routing, and response validation stages for each request.

Design Decision: Modular pipeline with robust error boundaries and structured logging.
Improves: Schema compliance (OutputValidator), Recall@10 (HybridRetriever + LLM Rerank),
          Coherence (Planner + Analyzer), Observability (structured log).
"""

from typing import Any, Optional

from app.catalog.loader import CatalogStore
from app.agents.intent_detector import IntentDetector, Intent
from app.agents.conversation_analyzer import ConversationAnalyzer, format_conversation
from app.agents.conversation_planner import ConversationPlanner, Action
from app.agents.clarifier import Clarifier
from app.agents.recommender import Recommender
from app.agents.comparator import Comparator
from app.agents.guardrails import Guardrails
from app.retrieval.hybrid_retriever import HiringProfile
from app.utils.llm_client import GeminiClient
from app.utils.validators import OutputValidator
from app.utils.logger import setup_logger, RequestLogger
from app.utils.helpers import extract_assessment_names

logger = setup_logger(__name__)


class Orchestrator:
    """Coordinates all agents, retrieval, and guardrail components.

    Attributes:
        catalog: Shared CatalogStore instance.
        llm_client: Client for Google Gemini API.
        intent_detector: Categorizes user intents.
        analyzer: Extracts hiring profile states from history.
        planner: Chooses the next agent action deterministically.
        clarifier: Generates targeted questions.
        recommender: Performs retrieval and LLM reranking.
        comparator: Handles side-by-side test differences.
        guardrails: Layered protection checks.
        validator: Validates response formats and links.
    """

    def __init__(
        self,
        catalog: CatalogStore,
        retriever: Any,  # HybridRetriever
        ranker: Any,      # Ranker
    ) -> None:
        self.catalog = catalog
        self.llm_client = GeminiClient()
        self.intent_detector = IntentDetector()
        self.analyzer = ConversationAnalyzer()
        self.planner = ConversationPlanner()
        self.clarifier = Clarifier()
        self.recommender = Recommender(catalog, retriever, ranker)
        self.comparator = Comparator(catalog)
        self.guardrails = Guardrails()
        self.validator = OutputValidator(
            catalog_urls=catalog.all_urls,
            catalog_names=catalog.all_names,
            catalog_name_to_url=catalog.name_to_url,
            catalog_url_to_type=catalog.url_to_type,
            catalog_name_to_type=catalog.name_to_type,
        )

    def process_message(
        self,
        messages: list[dict[str, str]],
        request_id: str = "default",
    ) -> dict[str, Any]:
        """Process incoming chat history and formulate next response.

        Args:
            messages: List of message dicts with 'role' and 'content'.
            request_id: Identifier for logging context.

        Returns:
            Dict matching required response schema.
        """
        with RequestLogger(logger, request_id=request_id) as rl:
            turn_count = len(messages)
            rl.log("turn_count", turn_count)

            if not messages:
                logger.warning("Empty messages list received.")
                return {
                    "reply": "How can I help you find the right SHL assessments today?",
                    "recommendations": [],
                    "end_of_conversation": False,
                }

            latest_msg = messages[-1].get("content", "").strip()
            rl.log("latest_message_len", len(latest_msg))

            # ─── Layer 1: Input Guardrails ────────────────────────────────────
            input_check = self.guardrails.check_input(latest_msg)
            if not input_check.is_safe:
                rl.log("refusal_reason", input_check.violation_type)
                return {
                    "reply": input_check.refusal_message,
                    "recommendations": [],
                    "end_of_conversation": False,
                }

            # ─── Layer 2: Analysis & Intent Detection ─────────────────────────
            # Format history for LLM
            formatted_history = format_conversation(messages)

            # Analyze intent (Deterministic + LLM fallback)
            intent = self._detect_intent(latest_msg, messages)
            rl.log("detected_intent", intent.value)

            # Extract accumulated hiring profile from history
            profile = self._analyze_conversation(messages)
            rl.log("profile_confidence", f"{profile.confidence:.2f}")

            # ─── Layer 3: Action Planning ─────────────────────────────────────
            # Determine if we previously recommended items to maintain continuity
            previous_recs = self._extract_previous_recommendations(messages)
            has_prev_recs = len(previous_recs) > 0
            rl.log("has_previous_recommendations", has_prev_recs)

            # Count how many times we have asked clarification questions
            clarification_count = self._count_clarification_turns(messages)
            rl.log("clarification_count", clarification_count)

            action = self.planner.plan(
                intent=intent,
                profile=profile,
                turn_count=turn_count,
                has_active_recommendations=has_prev_recs,
                clarification_count=clarification_count,
            )
            rl.log("planner_action", action.value)

            # ─── Layer 4: Routing & Execution ─────────────────────────────────
            reply, recommendations, end_of_conversation = self._execute_action(
                action=action,
                intent=intent,
                profile=profile,
                latest_msg=latest_msg,
                messages=messages,
                formatted_history=formatted_history,
                previous_recs=previous_recs,
                turn_count=turn_count,
            )

            # ─── Layer 5: Output Validation ───────────────────────────────────
            reply, recommendations, end_of_conversation = self.validator.validate_and_fix(
                reply=reply,
                recommendations=recommendations,
                end_of_conversation=end_of_conversation,
            )

            # Ensure output safety
            output_check = self.guardrails.check_output(reply, recommendations, self.catalog.all_urls)
            if not output_check.is_safe:
                logger.error("Guardrail blocked hallucinated output URL.")
                # Strip recommendations but keep the reply
                recommendations = []

            rl.log("recommendation_count", len(recommendations))
            rl.log("end_of_conversation", end_of_conversation)

            return {
                "reply": reply,
                "recommendations": recommendations,
                "end_of_conversation": end_of_conversation,
            }

    # ─── Pipeline Helpers ────────────────────────────────────────────────────

    def _detect_intent(self, message: str, messages: list[dict[str, str]]) -> Intent:
        """Classify user intent using deterministic rules first, then LLM."""
        # Check rule patterns first
        det_intent = self.intent_detector.detect(message, messages, llm_intent=None)
        # If we got greeting, off_topic, or injection, trust it
        if det_intent in (Intent.INJECTION, Intent.OFF_TOPIC, Intent.GREETING, Intent.GOODBYE):
            return det_intent

        # Use LLM call to classify
        from app.prompts.templates import INTENT_DETECTION_PROMPT
        history_str = format_conversation(messages[:-1]) if len(messages) > 1 else "None"
        prompt = INTENT_DETECTION_PROMPT.format(
            conversation=history_str,
            latest_message=message,
        )
        llm_resp = self.llm_client.generate(
            system_instruction="You are an intent classifier. Choose exactly ONE intent from the list.",
            prompt=prompt,
        )
        return self.intent_detector.detect(message, messages, llm_intent=llm_resp)

    def _analyze_conversation(self, messages: list[dict[str, str]]) -> HiringProfile:
        """Reconstruct the HiringProfile using LLM content analysis."""
        prompt = self.analyzer.get_analysis_prompt(messages)
        llm_resp = self.llm_client.generate(
            system_instruction="You extract structured Hiring Profiles in JSON. Return only the JSON object.",
            prompt=prompt,
        )
        return self.analyzer.analyze(messages, llm_response=llm_resp)

    def _extract_previous_recommendations(self, messages: list[dict[str, str]]) -> list[dict[str, str]]:
        """Scan history backwards to find the last assistant reply with recommendations.

        Recreates the latest list of recommended tests if the user changes constraints.
        """
        # Search backwards for a recommendation block in assistant responses
        # The recommendation details might be formatted as tables or lists, but we can look
        # for name matches against catalog
        for msg in reversed(messages):
            if msg.get("role") == "assistant":
                content = msg.get("content", "")
                names = extract_assessment_names(content, self.catalog.all_names_list)
                if names:
                    recs = []
                    for name in names:
                        item = self.catalog.get_by_name(name)
                        if item:
                            recs.append({
                                "name": item.name,
                                "url": item.link,
                                "test_type": item.test_type_code,
                            })
                    return recs
        return []

    def _count_clarification_turns(self, messages: list[dict[str, str]]) -> int:
        """Count how many times the assistant has asked clarification questions."""
        count = 0
        for msg in messages:
            if msg.get("role") == "assistant":
                content = msg.get("content", "").lower()
                # Simple heuristics for clarifying content
                if "?" in content and not any(kw in content for kw in ["url", "here are", "recommend"]):
                    count += 1
        return count

    def _execute_action(
        self,
        action: Action,
        intent: Intent,
        profile: HiringProfile,
        latest_msg: str,
        messages: list[dict[str, str]],
        formatted_history: str,
        previous_recs: list[dict[str, str]],
        turn_count: int,
    ) -> tuple[str, list[dict[str, str]], bool]:
        """Routes execution to the relevant sub-agent based on planned action."""
        if action == Action.GREET_AND_CLARIFY:
            return self.clarifier.format_greeting_response(), [], False

        elif action == Action.CLARIFY:
            missing = self.analyzer.get_missing_fields(profile)
            # Use clarifier question mapping first
            reply = self.clarifier.get_clarification_question(profile, missing, turn_count)
            return reply, [], False

        elif action == Action.RECOMMEND:
            # Retrieve candidates
            candidates = self.recommender.retrieve_candidates(profile)
            # Invoke LLM to select/describe from candidates
            prompt = self.recommender.get_recommendation_prompt(formatted_history, profile, candidates)
            llm_resp = self.llm_client.generate(
                system_instruction="You select the most relevant assessments from the candidates list. Return JSON.",
                prompt=prompt,
            )
            reply, recs = self.recommender.parse_llm_recommendations(llm_resp, candidates)
            # Format reply text with a Markdown table of recommendations for rich readability
            formatted_reply = self._append_recommendations_table(reply, recs)
            return formatted_reply, recs, False

        elif action == Action.REFINE:
            # Re-retrieve with new profile constraints
            candidates = self.recommender.retrieve_candidates(profile)
            # Call refinement prompt
            prompt = self.recommender.get_refinement_prompt(
                formatted_history, previous_recs, latest_msg, candidates
            )
            llm_resp = self.llm_client.generate(
                system_instruction="You modify the list of recommendations based on user requests. Return JSON.",
                prompt=prompt,
            )
            reply, recs = self.recommender.parse_refinement_response(llm_resp, candidates, previous_recs)
            formatted_reply = self._append_recommendations_table(reply, recs)
            return formatted_reply, recs, False

        elif action == Action.COMPARE:
            # Extract names to compare
            names = extract_assessment_names(latest_msg, self.catalog.all_names_list)
            prompt = self.comparator.get_comparison_prompt(latest_msg, names)

            if prompt:
                reply = self.llm_client.generate(
                    system_instruction="You compare SHL assessments using catalog data only.",
                    prompt=prompt,
                )
            else:
                # Fallback to static matching
                if len(names) >= 2:
                    item1 = self.catalog.get_by_name(names[0])
                    item2 = self.catalog.get_by_name(names[1])
                    if item1 and item2:
                        reply = self.comparator.generate_static_comparison(item1, item2)
                    else:
                        reply = "Which assessments would you like to compare? Please mention names like OPQ32r or Verify G+."
                else:
                    reply = "Which assessments would you like to compare? Please mention names like OPQ32r or Verify G+."

            # Comparison turns do NOT include recommendations in the response schema itself
            # but we preserve the active recommendations if they exist.
            return reply, previous_recs, False

        elif action == Action.REFUSE:
            # Polite refusal
            from app.prompts.templates import REFUSAL_PROMPT
            refusal_type = "prompt injection" if intent == Intent.INJECTION else "off-topic request"
            prompt = REFUSAL_PROMPT.format(message=latest_msg, refusal_type=refusal_type)
            reply = self.llm_client.generate(
                system_instruction="You politely refuse off-topic requests. Keep it brief.",
                prompt=prompt,
            )
            return reply, [], False

        elif action == Action.END_CONVERSATION:
            from app.prompts.templates import COMPLETION_PROMPT
            recs_str = ", ".join(r["name"] for r in previous_recs)
            prompt = COMPLETION_PROMPT.format(
                conversation=formatted_history,
                recommendations=recs_str,
            )
            reply = self.llm_client.generate(
                system_instruction="Provide a brief closing message. Keep it to 1-2 sentences.",
                prompt=prompt,
            )
            return reply, previous_recs, True

        # Fallback
        return "How can I help you find the right SHL assessments today?", [], False

    def _append_recommendations_table(self, reply: str, recs: list[dict[str, str]]) -> str:
        """Construct a formatted Markdown table of recommendations to append to the reply text.

        This matches the style shown in the sample conversations (C1, C3, C5, C9).
        """
        if not recs:
            return reply

        # Start Markdown table
        table_lines = [
            "",
            "| # | Name | Test Type | Keys | Duration | Languages | URL |",
            "|---|------|-----------|------|----------|-----------|-----|"
        ]

        for idx, rec in enumerate(recs, 1):
            name = rec["name"]
            test_type = rec["test_type"]

            # Lookup catalog details to enrich the table
            catalog_item = self.catalog.get_by_name(name)
            if catalog_item:
                keys = ", ".join(catalog_item.keys)
                duration = catalog_item.duration or "—"
                url = catalog_item.link

                # Format languages (showing top 4 + abbreviation if many)
                languages_list = catalog_item.languages
                if len(languages_list) > 4:
                    langs_str = ", ".join(languages_list[:4]) + f" _(+{len(languages_list) - 4} more)_"
                elif languages_list:
                    langs_str = ", ".join(languages_list)
                else:
                    langs_str = "—"
            else:
                keys = "—"
                duration = "—"
                langs_str = "—"
                url = rec["url"]

            table_lines.append(
                f"| {idx} | {name} | {test_type} | {keys} | {duration} | {langs_str} | <{url}> |"
            )

        # Join the reply and table with newlines
        return reply.strip() + "\n" + "\n".join(table_lines)
