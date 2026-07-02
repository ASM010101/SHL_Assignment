"""
Catalog enrichment script.

Pre-enriches the catalog data and saves it to data/enriched_catalog.json.
This can be run during build/deployment steps to avoid startup enrichment cost.
"""

from pathlib import Path
import sys

# Add project root to python path
sys.path.append(str(Path(__file__).parent.parent.resolve()))

from app.catalog.loader import CatalogStore
from app.catalog.enrichment import enrich_catalog, save_enriched_catalog
from app.utils.logger import setup_logger

logger = setup_logger(__name__)


def main():
    logger.info("Starting offline catalog enrichment...")
    catalog = CatalogStore()
    try:
        catalog.load("data/shl_product_catalog.json")
        enrich_catalog(catalog)
        save_enriched_catalog(catalog, "data/enriched_catalog.json")
        logger.info("Offline catalog enrichment finished successfully.")
        
        # Generate and save FAISS index during build
        logger.info("Starting offline FAISS index generation...")
        retrieval_texts = catalog.get_retrieval_texts()
        
        from app.retrieval.embeddings import generate_embeddings
        from app.retrieval.vector_store import VectorStore
        from app.config import FAISS_INDEX_PATH
        
        embeddings = generate_embeddings(retrieval_texts)
        
        vector_store = VectorStore()
        vector_store.build(embeddings)
        vector_store.save(FAISS_INDEX_PATH)
        logger.info("Offline FAISS index generation finished successfully.")
    except Exception as e:
        logger.error("Build-time precompute failed: %s", e)
        sys.exit(1)


if __name__ == "__main__":
    main()
