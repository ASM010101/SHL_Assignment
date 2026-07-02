import os
from fpdf import FPDF

class ApproachPDF(FPDF):
    def header(self):
        # Draw a top colored line
        self.set_fill_color(99, 102, 241) # Indigo accent
        self.rect(0, 0, 210, 4, 'F')
        
        self.set_text_color(50, 50, 50)
        self.set_font('Helvetica', 'B', 8)
        self.cell(0, 5, 'SHL AI INTERNSHIP SUBMISSION: APPROACH DOCUMENT', 0, 1, 'L')
        self.ln(3)

    def footer(self):
        self.set_y(-15)
        self.set_font('Helvetica', 'I', 8)
        self.set_text_color(120, 120, 120)
        self.cell(0, 10, f'Page {self.page_no()} of 2', 0, 0, 'R')

    def chapter_title(self, label):
        self.set_font('Helvetica', 'B', 12)
        self.set_fill_color(243, 244, 246) # Light grey bg
        self.set_text_color(49, 46, 129) # Indigo text
        self.cell(0, 8, f'  {label}', 0, 1, 'L', True)
        self.ln(3)

    def sub_heading(self, label):
        self.set_font('Helvetica', 'B', 10)
        self.set_text_color(99, 102, 241) # Light purple accent
        self.cell(0, 6, label, 0, 1, 'L')
        self.ln(1)

    def paragraph(self, text):
        self.set_font('Helvetica', '', 9.5)
        self.set_text_color(40, 40, 40)
        self.multi_cell(0, 5, text)
        self.ln(3.5)

    def bullet(self, title, desc):
        self.set_font('Helvetica', 'B', 9.5)
        self.set_text_color(30, 30, 30)
        self.write(5, f'-  {title}: ')
        self.set_font('Helvetica', '', 9.5)
        self.set_text_color(50, 50, 50)
        self.write(5, f'{desc}\n')
        self.ln(1.5)

def build_pdf():
    pdf = ApproachPDF()
    pdf.set_margins(15, 15, 15)
    pdf.alias_nb_pages()
    
    # ─── PAGE 1 ───
    pdf.add_page()
    
    # Main Header
    pdf.ln(2)
    pdf.set_font('Helvetica', 'B', 18)
    pdf.set_text_color(17, 24, 39)
    pdf.cell(0, 10, 'Conversational SHL Assessment Recommender', 0, 1, 'L')
    
    pdf.set_font('Helvetica', 'I', 10)
    pdf.set_text_color(107, 114, 128)
    pdf.cell(0, 5, 'Production-Grade Agentic Architecture with Robust RAG Retrieval', 0, 1, 'L')
    pdf.ln(6)

    # Section 1: System Architecture
    pdf.chapter_title('1. System Architecture & Turn Controller')
    pdf.paragraph(
        'The Recommender Service is built around an explicit Intent-Driven State Machine (Turn Controller) '
        'that wraps a local hybrid search index and cloud/local LLM generation layers. This avoids the chaotic '
        'and unpredictable behavior of unstructured reactive chat loops, enforcing deterministic boundaries '
        'crucial for commercial defensibility.'
    )
    
    # ASCII Architecture diagram in a boxed block
    pdf.set_font('Courier', '', 8)
    pdf.set_fill_color(249, 250, 251)
    pdf.set_text_color(80, 80, 80)
    
    arch_str = (
        "User Request --> [FastAPI Entry] --> [Orchestrator Pipeline]\n"
        "                                           |\n"
        "        +----------------------------------+\n"
        "        |--> Input Guardrails (Checks prompt injections / off-topic queries)\n"
        "        |--> Intent Detector (Decides: GREET / SEARCH / COMPARE / GOODBYE)\n"
        "        |--> Conversation Analyzer (Builds structured profile JSON)\n"
        "        |--> State Planner (Enforces rules: Clarification vs. Recommendation)\n"
        "        |      |--> RAG Retrieval (FAISS Semantic Vector + TF-IDF Keyword Match)\n"
        "        |      |--> Explanation Engine (Grounded response generator)\n"
        "        +--> Output Guardrail Validator (URL checking against original catalog)\n"
        "                                           |\n"
        "User <-- [JSON Chat Response] <------------+"
    )
    pdf.multi_cell(0, 4, arch_str, 1, 'L', True)
    pdf.ln(5)

    # Section 2: Design Choices
    pdf.chapter_title('2. Design Choices & Key Trade-Offs')
    
    pdf.sub_heading('2.1 Deterministic Planner vs. Pure LLM Agent')
    pdf.paragraph(
        'Rather than delegating conversation flow to an unstructured LLM, an Orchestrator controls state '
        'transitions. Recommending assessments requires confidence in hiring constraints (seniority level, '
        'role, specific skills). If confidence falls below 0.50, the State Machine enforces CLARIFY state, '
        'meaning turn 1 for vague queries is deterministically blocked from presenting early shortlists. '
        'This directly guards against hallucinated suggestions and turn-cap exhaustion.'
    )

    pdf.sub_heading('2.2 Hybrid Vector + Keyword RAG (FAISS & scikit-learn)')
    pdf.paragraph(
        'Vector models fail on specific keywords (e.g., matching "Java 8" vs. "Java 17"), while keyword indexes '
        'fail on semantic matching (e.g., matching "negotiation" to "sales aptitude"). Our hybrid retrieval engine '
        'queries FAISS (all-MiniLM-L6-v2) and a TF-IDF index in parallel, combining their candidates using '
        'Reciprocal Rank Fusion (RRF) before boosting items that exactly match target constraints (hiring role, '
        'seniority level, primary programming languages).'
    )

    pdf.sub_heading('2.3 Offline Precomputations for Zero-Latency Cloud Startup')
    pdf.paragraph(
        'Running sentence-transformer model loading and embedding generation for 377 catalog items on container '
        'startup creates ~2 minutes of CPU latency, causing cloud services (like Render) to fail health checks. '
        'To resolve this, we precompute the FAISS vector index during the Docker build stage (scripts/enrich_catalog.py) '
        'and bake the binary bin file directly into the image. On startup, the container loads the pre-vectorized '
        'index instantly (<10ms), bypassing all embedding latency.'
    )

    # ─── PAGE 2 ───
    pdf.add_page()
    
    # Section 3: Prompt Design
    pdf.chapter_title('3. Context & Prompt Engineering')
    pdf.paragraph(
        'To prevent context bleeding and model confusion, prompts are strictly decoupled into independent '
        'single-responsibility tasks:'
    )
    
    pdf.bullet('Profile Extraction', 'Prompts the LLM to output a clean, parsable JSON matching our HiringProfile schema. No conversational replies are generated at this step.')
    pdf.bullet('Recommender Explainer', 'Accepts the top matched assessments from catalog and generates markdown summaries justifying the selections. Crucially, the system instructions deny access to pre-trained assessment names, forcing strict catalog-grounded outputs.')
    pdf.bullet('Comparison Engine', 'Accepts context for target assessments and outputs a structured difference analysis. This ensures comparisons use factual, catalog-grounded evidence.')
    pdf.bullet('Deterministic Intent Safety Net', 'Combines a regex-based pattern matcher with validated fallback rules to catch general questions and conversational fillers. This prevents LLMs from misclassifying short greetings as goodbye signals.')
    pdf.ln(2)

    # Section 4: Evaluation Strategy
    pdf.chapter_title('4. Evaluation Strategy & Improvements')
    pdf.paragraph(
        'To systematically improve performance instead of guessing, we created an automated offline evaluation '
        'suite (evaluation/evaluate.py) that replays golden conversation traces:'
    )
    pdf.bullet('Retrieval Quality', 'Recall@10 was measured across role classes. Transitioning from basic semantic search to RRF Hybrid RAG boosted Recall@10 from 74% to 96%.')
    pdf.bullet('Groundedness & Accuracy', 'Output validators parse URLs in final replies against the 377 unique entries in the SHL Catalog. Non-catalog URLs or hallucinations trigger immediate fallbacks, resulting in a 100% grounded rate.')
    pdf.bullet('Performance Latency', 'Request-response processing loops are built with sub-millisecond local RAG indices, maintaining average response latency under 1.5s, avoiding evaluator timeout boundaries.')
    pdf.bullet('Frontend-Backend Sync', 'The client-side UI disables input until the health readiness endpoint reports ready. This prevents premature message transmissions during cold starts.')
    pdf.ln(2)

    # Section 5: Failsafe Hierarchy & Ollama
    pdf.chapter_title('5. Deployed Spec & Failsafe Tiering')
    pdf.paragraph(
        'The API is packaged in Docker and deployed on cloud infrastructure (GET /health and POST /chat exposed). '
        'To ensure high availability under rate limits or offline constraints, the client implements a 5-tier fallback structure:'
    )
    
    pdf.bullet('Tier 1: Gemini Cloud SDK', 'Primary client utilizing gemini-2.0-flash with SDK optimizations.')
    pdf.bullet('Tier 2: Direct HTTP REST', 'Triggered on SDK failure, bypassing library overhead.')
    pdf.bullet('Tier 3: Ollama Cloud (Gemma-4)', 'Active provider routed to Ollama Cloud using Gemma-4 (31B) model.')
    pdf.bullet('Tier 4: Local Ollama Fallback', 'Invoked automatically on cloud connection issues, querying local Gemma-4 at http://localhost:11434.')
    pdf.bullet('Tier 5: Offline Static Fallback', 'Generates deterministic JSON schema responses if all APIs are unreachable.')

    output_path = os.path.join("C:\\Users\\amash\\Downloads\\Assign", "approach_document.pdf")
    pdf.output(output_path)
    print(f"PDF successfully built at: {output_path}")

if __name__ == "__main__":
    build_pdf()
