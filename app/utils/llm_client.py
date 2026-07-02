"""
Gemini LLM Client for SHL Assessment Recommender.

Invokes the Google Gemini API to generate structured content and text.
Includes a direct HTTP fallback if the google-genai package encounters import or API issues.

Design Decision: Dual-mode invocation (SDK + direct REST) for extreme reliability.
Improves: Robustness, Performance, Timeout resilience.
"""

import json
import httpx
from typing import Any, Optional

from app.config import (
    GEMINI_API_KEY, LLM_MODEL, LLM_TEMPERATURE, LLM_MAX_TOKENS,
    LLM_PROVIDER, OLLAMA_BASE_URL, OLLAMA_MODEL, OLLAMA_API_KEY
)
from app.utils.logger import setup_logger

logger = setup_logger(__name__)

# Try importing the official SDK
try:
    from google import genai
    from google.genai import types
    sdk_available = True
except ImportError:
    sdk_available = False
    logger.warning("google-genai SDK not available. Using direct HTTP fallback.")


class GeminiClient:
    """Invokes the Google Gemini API.

    Supports both SDK and REST fallback modes.
    Includes quota-aware cooldown to avoid repeated 429 timeouts.
    """

    def __init__(self) -> None:
        self.api_key = GEMINI_API_KEY
        self.model = LLM_MODEL
        self.temperature = LLM_TEMPERATURE
        self.max_tokens = LLM_MAX_TOKENS
        self.client = None
        self._quota_exhausted_at: float = 0.0  # timestamp of last 429 error
        self._quota_cooldown: float = 60.0  # seconds to wait before retrying

        if sdk_available and self.api_key:
            try:
                self.client = genai.Client(api_key=self.api_key)
                logger.info("GenAI SDK client initialized with model: %s", self.model)
            except Exception as e:
                logger.warning("GenAI SDK initialization failed: %s. Using REST mode.", e)

    def _is_quota_exhausted(self) -> bool:
        """Check if we're in a quota cooldown period."""
        import time
        if self._quota_exhausted_at > 0:
            elapsed = time.time() - self._quota_exhausted_at
            if elapsed < self._quota_cooldown:
                return True
            # Cooldown expired, try again
            self._quota_exhausted_at = 0.0
        return False

    def _mark_quota_exhausted(self) -> None:
        """Mark that we hit a quota limit."""
        import time
        self._quota_exhausted_at = time.time()
        logger.warning("Quota exhausted. Entering cooldown for %.0fs.", self._quota_cooldown)

    def generate(self, system_instruction: str, prompt: str) -> str:
        """Generate content from LLM given a system instruction and prompt.

        Args:
            system_instruction: System prompt framing rules and scope.
            prompt: Task-specific prompt text.

        Returns:
            Text content response from model.
        """
        if LLM_PROVIDER == "ollama":
            return self._generate_ollama(system_instruction, prompt)

        if not self.api_key:
            logger.warning("No GEMINI_API_KEY configured. Returning fallback response.")
            return self._get_fallback_response(prompt)

        # Skip API calls if in quota cooldown
        if self._is_quota_exhausted():
            logger.debug("Skipping API call (quota cooldown active). Using fallback.")
            return self._get_fallback_response(prompt)

        # Method 1: GenAI SDK
        if self.client is not None:
            try:
                config = types.GenerateContentConfig(
                    system_instruction=system_instruction,
                    temperature=self.temperature,
                    max_output_tokens=self.max_tokens,
                )
                response = self.client.models.generate_content(
                    model=self.model,
                    contents=prompt,
                    config=config,
                )
                if response.text:
                    return response.text
            except Exception as e:
                error_str = str(e)
                if "429" in error_str or "RESOURCE_EXHAUSTED" in error_str:
                    self._mark_quota_exhausted()
                    return self._get_fallback_response(prompt)
                logger.error("SDK generate_content failed: %s. Falling back to HTTP REST API.", e)

        # Method 2: Direct HTTP REST API
        return self._generate_http(system_instruction, prompt)

    def _generate_http(self, system_instruction: str, prompt: str) -> str:
        """Call the Gemini REST API directly using httpx.

        Provides a fail-safe fallback when SDK fails or is missing.
        """
        # Mapping model to correct API string format
        # If user specifies 'gemini-2.0-flash', end point is:
        # https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:generateContent?key={self.api_key}"

        payload = {
            "contents": [
                {
                    "parts": [
                        {"text": prompt}
                    ]
                }
            ],
            "systemInstruction": {
                "parts": [
                    {"text": system_instruction}
                ]
            },
            "generationConfig": {
                "temperature": self.temperature,
                "maxOutputTokens": self.max_tokens,
            }
        }

        headers = {
            "Content-Type": "application/json"
        }

        try:
            with httpx.Client(timeout=15.0) as client:
                response = client.post(url, json=payload, headers=headers)
                if response.status_code == 200:
                    resp_json = response.json()
                    candidates = resp_json.get("candidates", [])
                    if candidates:
                        parts = candidates[0].get("content", {}).get("parts", [])
                        if parts:
                            return parts[0].get("text", "")
                if response.status_code == 429:
                    self._mark_quota_exhausted()
                else:
                    logger.error(
                        "Gemini HTTP API error: status=%d body=%s",
                        response.status_code,
                        response.text[:200]
                    )
        except Exception as e:
            logger.error("HTTP request to Gemini API failed: %s", e)

        return self._get_fallback_response(prompt)

    def _generate_ollama(self, system_instruction: str, prompt: str) -> str:
        """Call the local Ollama API directly using httpx."""
        base = OLLAMA_BASE_URL.rstrip('/')
        if base.endswith("/api"):
            url = f"{base}/generate"
        else:
            url = f"{base}/api/generate"
        payload = {
            "model": OLLAMA_MODEL,
            "system": system_instruction,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": self.temperature,
            }
        }
        headers = {
            "Content-Type": "application/json"
        }
        if OLLAMA_API_KEY:
            headers["Authorization"] = f"Bearer {OLLAMA_API_KEY}"

        try:
            with httpx.Client(timeout=30.0) as client:
                response = client.post(url, json=payload, headers=headers)
                if response.status_code == 200:
                    resp_json = response.json()
                    return resp_json.get("response", "")
                logger.error(
                    "Ollama API error: status=%d body=%s",
                    response.status_code,
                    response.text[:200]
                )
        except Exception as e:
            logger.error("HTTP request to Ollama API failed: %s", e)

        # Automatic Local Fallback if cloud fails
        if base != "http://localhost:11434":
            logger.warning("Primary Ollama service failed. Falling back to local Ollama (http://localhost:11434/api/generate)...")
            local_url = "http://localhost:11434/api/generate"
            local_payload = {
                "model": "gemma4:31b",
                "system": system_instruction,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": self.temperature,
                }
            }
            try:
                with httpx.Client(timeout=45.0) as client:
                    local_resp = client.post(local_url, json=local_payload, timeout=45.0)
                    if local_resp.status_code == 200:
                        logger.info("Local Ollama fallback succeeded!")
                        return local_resp.json().get("response", "")
            except Exception as ex:
                logger.error("Local Ollama fallback also failed: %s", ex)

        return self._get_fallback_response(prompt)

    def _get_fallback_response(self, prompt: str) -> str:
        """Generates deterministic local responses if Gemini call fails."""
        # Simple JSON structures based on prompt text
        prompt_lower = prompt.lower()
        if "conversationanalysis" in prompt_lower or "extract the following fields" in prompt_lower:
            return json.dumps({
                "role": "unknown",
                "seniority": None,
                "skills": [],
                "soft_skills": [],
                "domain": None,
                "industry": None,
                "languages": [],
                "purpose": None,
                "assessment_types": [],
                "job_level": None,
                "constraints": [],
                "raw_jd": None,
                "add_requests": [],
                "remove_requests": []
            })
        elif "intent" in prompt_lower or "classify the intent" in prompt_lower:
            return "SEARCH"
        elif "recommendation" in prompt_lower or "select the most relevant" in prompt_lower:
            return json.dumps({
                "reply": "Here is a candidate selection from the catalog.",
                "selected_indices": [1]
            })
        elif "refinement" in prompt_lower:
            return json.dumps({
                "reply": "I have updated the list of recommendations.",
                "selected_indices": [],
                "kept_from_previous": []
            })
        return "I apologize, but I am currently unable to process this request. Let me assist you with the general catalog."
