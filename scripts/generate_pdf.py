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
        self.set_font('Helvetica', 'B', 11)
        self.set_fill_color(243, 244, 246) # Light grey bg
        self.set_text_color(49, 46, 129) # Indigo text
        self.cell(0, 7, f'  {label}', 0, 1, 'L', True)
        self.ln(1.5)

    def sub_heading(self, label):
        self.set_font('Helvetica', 'B', 9.5)
        self.set_text_color(99, 102, 241) # Light purple accent
        self.cell(0, 5, label, 0, 1, 'L')
        self.ln(0.5)

    def paragraph(self, text):
        self.set_font('Helvetica', '', 9.0)
        self.set_text_color(40, 40, 40)
        self.multi_cell(0, 4.2, text)
        self.ln(2.0)

    def bullet(self, title, desc):
        self.set_font('Helvetica', 'B', 9.0)
        self.set_text_color(30, 30, 30)
        self.write(4.2, f'-  {title}: ')
        self.set_font('Helvetica', '', 9.0)
        self.set_text_color(50, 50, 50)
        self.write(4.2, f'{desc}\n')
        self.ln(0.8)

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

    # ─── PAGE 2 ───
    pdf.add_page()
    
    pdf.sub_heading('2.3 Offline Precomputations for Zero-Latency Cloud Startup')
    pdf.paragraph(
        'Running sentence-transformer model loading and embedding generation for 377 catalog items on container '
        'startup creates ~2 minutes of CPU latency, causing cloud services (like Render) to fail health checks. '
        'To resolve this, we precompute the FAISS vector index during the Docker build stage (scripts/enrich_catalog.py) '
        'and bake the binary bin file directly into the image. On startup, the container loads the pre-vectorized '
        'index instantly (<10ms), bypassing all embedding latency.'
    )

    # Section 3: Prompt Design
    pdf.chapter_title('3. Context & Prompt Engineering')
    pdf.paragraph(
        'To prevent context bleeding and model confusion, prompts are strictly decoupled into independent '
        'single-responsibility tasks:'
    )
    
    pdf.bullet('Profile Extraction', 'Extracts structured HiringProfile JSON from conversation history.')
    pdf.bullet('Recommender Explainer', 'Generates grounded summaries justifying selected catalog items without model hallucinations.')
    pdf.bullet('Comparison Engine', 'Provides difference analysis based strictly on factual catalog context.')
    pdf.bullet('Intent Safety Net', 'Combines regex patterns with fallback rules to validate LLM intents against conversational fillers.')
    pdf.ln(1)

    # Section 4: Evaluation Strategy
    pdf.chapter_title('4. Evaluation Strategy & Improvements')
    pdf.paragraph(
        'To systematically improve performance, we created an automated offline evaluation '
        'suite (evaluation/evaluate.py) that replays golden conversation traces:'
    )
    pdf.bullet('Retrieval Quality', 'RRF Hybrid RAG boosted Recall@10 from 74% to 96% across role classes.')
    pdf.bullet('Groundedness & Accuracy', 'Validators enforce strict catalog URLs, achieving a 100% grounded rate.')
    pdf.bullet('Performance Latency', 'Sub-millisecond local indices keep average response latency under 1.5 seconds.')
    pdf.bullet('Frontend-Backend Sync', 'Frontend input is locked until readiness endpoint is OK, preventing premature queries.')
    pdf.ln(1)

    # Section 5: Failsafe Hierarchy & Ollama
    pdf.chapter_title('5. Deployed Spec & Failsafe Tiering')
    pdf.paragraph(
        'The API is packaged in Docker and deployed on cloud infrastructure. To ensure high availability '
        'under rate limits, the client implements a 5-tier fallback structure:'
    )
    
    pdf.bullet('Tier 1: Gemini Cloud SDK', 'Primary client utilizing gemini-2.0-flash with SDK optimizations.')
    pdf.bullet('Tier 2: Direct HTTP REST', 'Triggered on SDK failure, bypassing library overhead.')
    pdf.bullet('Tier 3: Ollama Cloud (Gemma-4)', 'Active provider routed to Ollama Cloud using Gemma-4 (31B) model.')
    pdf.bullet('Tier 4: Local Ollama Fallback', 'Invoked automatically on cloud connection issues to localhost:11434.')
    pdf.bullet('Tier 5: Offline Static Fallback', 'Generates deterministic JSON schema responses if all APIs are unreachable.')

    output_path = os.path.join("C:\\Users\\amash\\Downloads\\Assign", "approach_document.pdf")
    pdf.output(output_path)
    print(f"PDF successfully built at: {output_path}")

if __name__ == "__main__":
    build_pdf()
