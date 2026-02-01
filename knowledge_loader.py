"""
Script to build vector database from golf instruction documents.

Usage:
    python knowledge_loader.py --knowledge_dir knowledge/golf_instruction/
"""

import argparse
from pathlib import Path
from rag_feedback_generator import RAGFeedbackGenerator


def main():
    parser = argparse.ArgumentParser(description='Build golf instruction knowledge base')
    parser.add_argument(
        '--knowledge_dir',
        type=str,
        default='knowledge/golf_instruction/',
        help='Path to directory containing golf instruction documents (.txt files)'
    )
    parser.add_argument(
        '--vector_db_dir',
        type=str,
        default='knowledge/vector_db',
        help='Path to store vector database'
    )
    parser.add_argument(
        '--embedding_model',
        type=str,
        default='all-MiniLM-L6-v2',
        help='Sentence transformer model for embeddings'
    )
    parser.add_argument(
        '--use_ollama',
        action='store_true',
        help='Use Ollama for embeddings instead of sentence-transformers'
    )
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("Golf Instruction Knowledge Base Builder")
    print("=" * 60)
    
    # Create RAG generator (will build vector DB)
    generator = RAGFeedbackGenerator(
        knowledge_base_path=args.knowledge_dir,
        vector_db_path=args.vector_db_dir,
        use_rag=True,
        embedding_model=args.embedding_model,
        use_ollama_embeddings=args.use_ollama,
        use_llm=False  # Don't need LLM for building DB
    )
    
    if generator.use_rag:
        print("\n✅ Knowledge base built successfully!")
        if hasattr(generator, 'collection') and isinstance(generator.collection, dict):
            pickle_path = Path(args.vector_db_dir) / 'vector_db.pkl'
            if pickle_path.exists():
                print(f"   Storage: Pickle file (persisted)")
                print(f"   Location: {pickle_path}")
            else:
                print(f"   Storage: In-memory (will save to pickle)")
                print(f"   Location: {args.vector_db_dir}/vector_db.pkl")
            print("   Note: Install ChromaDB for advanced persistent storage: pip install chromadb")
        else:
            print(f"   Vector DB location: {args.vector_db_dir}")
    else:
        print("\n❌ Failed to build knowledge base")
        print("   Check that:")
        print("   1. Knowledge directory exists and contains .txt files")
        print("   2. Required packages are installed (sentence-transformers, chromadb)")


if __name__ == '__main__':
    main()

