"""
FastAPI Application Entry Point for SHL Assessment Recommender.

Defines application lifespan logic to preload the catalog, build
search indices (FAISS vector and TF-IDF keyword), and initialize the orchestrator.

Design Decision: Pre-build indices during startup so requests are processed instantly.
Improves: Performance (low request latency, no runtime indexing), Timeout resilience.
"""

from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import router
from app.catalog.loader import CatalogStore
from app.catalog.enrichment import enrich_catalog, save_enriched_catalog
from app.retrieval.embeddings import generate_embeddings
from app.retrieval.vector_store import VectorStore
from app.retrieval.keyword_search import KeywordSearchEngine
from app.retrieval.hybrid_retriever import HybridRetriever
from app.retrieval.ranker import Ranker
from app.agents.orchestrator import Orchestrator
from app.config import HOST, PORT, CATALOG_PATH, ENRICHED_CATALOG_PATH
from app.utils.logger import setup_logger

logger = setup_logger(__name__)


import asyncio

async def initialize_system_async(app: FastAPI):
    """Performs catalog loading, embedding generation, index building, and orchestrator setup asynchronously."""
    app.state.initialization_status = {"status": "loading", "step": "Starting up system...", "progress": 5}
    
    try:
        # Step 1: Loading Catalog
        app.state.initialization_status = {"status": "loading", "step": "Loading SHL Assessment Catalog from disk...", "progress": 15}
        await asyncio.sleep(0.1)
        
        catalog = CatalogStore()
        import os
        catalog_to_load = CATALOG_PATH
        if os.path.exists(ENRICHED_CATALOG_PATH):
            catalog_to_load = ENRICHED_CATALOG_PATH
            logger.info("Enriched catalog found. Loading: %s", ENRICHED_CATALOG_PATH)
        else:
            logger.info("Enriched catalog not found. Loading raw: %s", CATALOG_PATH)
            
        catalog.load(catalog_to_load)
        
        # Step 2: Enrich catalog if raw
        if catalog_to_load == CATALOG_PATH:
            app.state.initialization_status = {"status": "loading", "step": "Performing in-memory catalog enrichment...", "progress": 25}
            await asyncio.sleep(0.1)
            enrich_catalog(catalog)
            try:
                save_enriched_catalog(catalog, ENRICHED_CATALOG_PATH)
            except Exception as e:
                logger.warning("Could not save enriched catalog to disk: %s", e)
                
        # Get retrieval texts for index building
        retrieval_texts = catalog.get_retrieval_texts()
        
        # Step 3: Load or Generate FAISS index
        vector_store = VectorStore()
        from app.config import FAISS_INDEX_PATH
        import os
        
        # Try loading index from disk first to bypass CPU-heavy generation on startup
        loaded_successfully = False
        if os.path.exists(FAISS_INDEX_PATH) or os.path.exists(FAISS_INDEX_PATH + ".npy"):
            app.state.initialization_status = {"status": "loading", "step": "Loading precomputed FAISS vector search index...", "progress": 50}
            await asyncio.sleep(0.01)
            loaded_successfully = vector_store.load(FAISS_INDEX_PATH)
            
        if not loaded_successfully:
            # Fallback: load model and generate embeddings on the fly
            app.state.initialization_status = {"status": "loading", "step": "Loading embedding model (all-MiniLM-L6-v2)...", "progress": 40}
            await asyncio.sleep(0.01)
            
            app.state.initialization_status = {"status": "loading", "step": "Generating search vectors (RAG embeddings)...", "progress": 65}
            await asyncio.sleep(0.01)
            
            loop = asyncio.get_running_loop()
            embeddings = await loop.run_in_executor(None, generate_embeddings, retrieval_texts)
            
            app.state.initialization_status = {"status": "loading", "step": "Building local FAISS vector search index...", "progress": 80}
            await asyncio.sleep(0.01)
            vector_store.build(embeddings)
            try:
                vector_store.save(FAISS_INDEX_PATH)
            except Exception as e:
                logger.warning("Could not save vector index to disk: %s", e)
        
        # Step 5: Build TF-IDF
        app.state.initialization_status = {"status": "loading", "step": "Building local keyword index (TF-IDF)...", "progress": 90}
        await asyncio.sleep(0.1)
        keyword_engine = KeywordSearchEngine()
        keyword_engine.build(retrieval_texts)
        
        # Step 6: Build retriever, ranker, and orchestrator
        app.state.initialization_status = {"status": "loading", "step": "Configuring orchestration pipeline...", "progress": 95}
        await asyncio.sleep(0.1)
        retriever = HybridRetriever(catalog, vector_store, keyword_engine)
        ranker = Ranker()
        orchestrator = Orchestrator(catalog, retriever, ranker)
        
        # Step 7: Finalize App State
        app.state.catalog = catalog
        app.state.orchestrator = orchestrator
        app.state.initialization_status = {"status": "ok", "step": "Ready", "progress": 100}
        logger.info("Asynchronous startup complete. Service ready for traffic.")
        
    except Exception as e:
        logger.critical("Asynchronous startup failed with exception: %s", e, exc_info=True)
        app.state.initialization_status = {"status": "error", "step": f"Initialization failed: {str(e)}", "progress": 0}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifecycle context manager to perform startup initialization and shutdown cleanup."""
    logger.info("Initializing SHL Assessment Recommender Service...")
    
    # Initialize status variable
    app.state.initialization_status = {"status": "loading", "step": "Starting up...", "progress": 5}
    
    # Launch startup sequence in the background, freeing uvicorn to start serving requests immediately
    asyncio.create_task(initialize_system_async(app))
    
    yield
    
    logger.info("Shutting down SHL Assessment Recommender Service...")


# ─── FastAPI Initialization ──────────────────────────────────────────────────

app = FastAPI(
    title="SHL Conversational Assessment Recommender",
    description="Conversational agent to recommend SHL assessments based on roles and requirements.",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS configurations
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include Router
app.include_router(router)


# Self-start script for convenience
if __name__ == "__main__":
    import uvicorn
    logger.info("Starting local server on %s:%d", HOST, PORT)
    uvicorn.run("app.main:app", host=HOST, port=PORT, reload=False)
