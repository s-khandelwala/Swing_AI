"""
RAG-Enhanced Golf Swing Feedback Generator

Extends LLMFeedbackGenerator with Retrieval-Augmented Generation
for factually grounded, knowledge-based feedback.
"""

import numpy as np
from typing import Dict, List, Optional
import json
import os
import pickle
from pathlib import Path
from llm_feedback_generator import LLMFeedbackGenerator

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    print("⚠️  sentence-transformers not installed. Install with: pip install sentence-transformers")

try:
    import chromadb
    CHROMADB_AVAILABLE = True
except ImportError:
    CHROMADB_AVAILABLE = False
    print("⚠️  chromadb not installed. Install with: pip install chromadb")

try:
    import ollama
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False


class RAGFeedbackGenerator(LLMFeedbackGenerator):
    """
    RAG-Enhanced feedback generator that retrieves relevant golf instruction
    knowledge before generating feedback.
    """
    
    def __init__(
        self,
        knowledge_base_path: str = 'knowledge/golf_instruction/',
        vector_db_path: str = 'knowledge/vector_db',
        use_rag: bool = True,
        retrieval_k: int = 5,  # Number of chunks to retrieve
        embedding_model: str = 'all-MiniLM-L6-v2',
        use_ollama_embeddings: bool = False,  # Use Ollama for embeddings if available
        **kwargs
    ):
        """
        Args:
            knowledge_base_path: Path to golf instruction documents
            vector_db_path: Path to store vector database
            use_rag: If True, use RAG; if False, fall back to base LLM
            retrieval_k: Number of knowledge chunks to retrieve
            embedding_model: Sentence transformer model for embeddings
            use_ollama_embeddings: Use Ollama embeddings API if available
            **kwargs: Passed to parent LLMFeedbackGenerator
        """
        super().__init__(**kwargs)
        
        self.knowledge_base_path = Path(knowledge_base_path)
        self.vector_db_path = Path(vector_db_path)
        self.use_rag = use_rag and (SENTENCE_TRANSFORMERS_AVAILABLE or OLLAMA_AVAILABLE)
        self.retrieval_k = retrieval_k
        self.embedding_model_name = embedding_model
        self.use_ollama_embeddings = use_ollama_embeddings
        
        # Initialize RAG components
        self.vector_db = None
        self.embedder = None
        self.collection = None
        
        if self.use_rag:
            self._initialize_rag()
    
    def _initialize_rag(self):
        """Initialize RAG components (vector DB and embedder)"""
        try:
            # Initialize embedder
            if self.use_ollama_embeddings and OLLAMA_AVAILABLE:
                print("✅ Using Ollama for embeddings")
                self.embedder = 'ollama'  # Will use Ollama API
            elif SENTENCE_TRANSFORMERS_AVAILABLE:
                print(f"✅ Loading embedding model: {self.embedding_model_name}")
                self.embedder = SentenceTransformer(self.embedding_model_name)
            else:
                print("⚠️  No embedding model available. RAG disabled.")
                self.use_rag = False
                return
            
            # Initialize vector database
            if CHROMADB_AVAILABLE:
                self._initialize_chromadb()
            else:
                print("⚠️  ChromaDB not available. Using simple in-memory storage (not persisted).")
                print("   Install ChromaDB for persistent storage: pip install chromadb")
                self._initialize_simple_db()
            
            # Check if knowledge base exists (use parent directory to include drills)
            knowledge_dir = self.knowledge_base_path
            if knowledge_dir.name == 'golf_instruction':
                knowledge_dir = knowledge_dir.parent  # Use parent to include drills folder
            
            if not knowledge_dir.exists():
                print(f"⚠️  Knowledge base not found at {knowledge_dir}")
                print("   Run knowledge_loader.py to build knowledge base first.")
                self.use_rag = False
            else:
                # Load or build vector database
                if CHROMADB_AVAILABLE and self.collection:
                    count = self.collection.count()
                    if count == 0:
                        print("📚 Building vector database from knowledge base...")
                        self._build_vector_db()
                    else:
                        print(f"✅ Vector database loaded ({count} chunks)")
                elif not CHROMADB_AVAILABLE:
                    # Simple DB - check if pickle exists and has data
                    pickle_path = self.vector_db_path / 'vector_db.pkl'
                    if pickle_path.exists() and len(self.collection.get('chunks', [])) > 0:
                        print(f"✅ Vector database loaded from pickle ({len(self.collection['chunks'])} chunks)")
                    else:
                        print("📚 Building vector database from knowledge base...")
                        print("   Note: Will save to pickle file for persistence.")
                        self._build_vector_db()
        
        except Exception as e:
            print(f"⚠️  RAG initialization failed: {e}")
            print("   Falling back to base LLM without RAG")
            self.use_rag = False
    
    def _initialize_chromadb(self):
        """Initialize ChromaDB vector database"""
        client = chromadb.PersistentClient(path=str(self.vector_db_path))
        
        # Get or create collection
        try:
            self.collection = client.get_collection("golf_instruction")
            print("✅ Loaded existing vector database")
        except:
            self.collection = client.create_collection(
                name="golf_instruction",
                metadata={"description": "Golf instruction knowledge base"}
            )
            print("✅ Created new vector database")
    
    def _initialize_simple_db(self):
        """Simple in-memory fallback if ChromaDB not available"""
        # Ensure directory exists
        self.vector_db_path.mkdir(parents=True, exist_ok=True)
        
        pickle_path = self.vector_db_path / 'vector_db.pkl'
        
        # Try to load existing data
        if pickle_path.exists():
            try:
                with open(pickle_path, 'rb') as f:
                    data = pickle.load(f)
                    self.collection = {
                        'chunks': data.get('chunks', []),
                        'embeddings': data.get('embeddings', []),
                        'metadata': data.get('metadata', [])
                    }
                    print(f"✅ Loaded {len(self.collection['chunks'])} chunks from pickle file")
                    return
            except Exception as e:
                print(f"⚠️  Error loading pickle file: {e}. Rebuilding...")
        
        # Initialize empty collection
        self.collection = {
            'chunks': [],
            'embeddings': [],
            'metadata': []
        }
        print("⚠️  Using simple in-memory storage (will save to pickle)")
    
    def _build_vector_db(self):
        """Build vector database from knowledge base documents"""
        # Get parent knowledge directory (to include both golf_instruction and drills)
        knowledge_dir = self.knowledge_base_path
        if knowledge_dir.name == 'golf_instruction':
            knowledge_dir = knowledge_dir.parent  # Use parent to include drills folder
        
        if not knowledge_dir.exists():
            print(f"❌ Knowledge base not found: {knowledge_dir}")
            return
        
        # Find all text files
        text_files = list(knowledge_dir.rglob("*.txt"))
        
        # Find all JSON files (for drills)
        json_files = list(knowledge_dir.rglob("*.json"))
        
        # Check for drills.json file (new format)
        drills_json_file = knowledge_dir / 'drills.json'
        if drills_json_file.exists():
            # Count drills in the file
            try:
                with open(drills_json_file, 'r', encoding='utf-8') as f:
                    drills_data = json.load(f)
                    num_drills = len(drills_data.get('drills', []))
                    print(f"📖 Found drills.json with {num_drills} drills")
            except Exception as e:
                print(f"⚠️  Error reading drills.json: {e}")
        
        all_files = text_files + json_files
        
        if not all_files:
            print(f"⚠️  No .txt or .json files found in {knowledge_dir}")
            return
        
        # Count files (excluding drills.json if it exists, as it's handled separately)
        other_json_files = [f for f in json_files if f.name != 'drills.json']
        print(f"📖 Processing {len(text_files)} text files and {len(other_json_files)} other JSON files...")
        
        all_chunks = []
        all_embeddings = []
        all_metadata = []
        
        for file_path in all_files:
            try:
                if file_path.suffix == '.json':
                    # Check if this is the drills.json file (single file with array)
                    if file_path.name == 'drills.json':
                        # Parse drills.json file (contains array of drills)
                        with open(file_path, 'r', encoding='utf-8') as f:
                            drills_data = json.load(f)
                        
                        # Extract drills array
                        drills = drills_data.get('drills', [])
                        chunks = []
                        
                        # Process each drill in the array
                        for drill in drills:
                            drill_chunks = self._parse_drill_json(drill, file_path)
                            chunks.extend(drill_chunks)
                    else:
                        # Parse individual JSON drill file (old format)
                        with open(file_path, 'r', encoding='utf-8') as f:
                            drill_data = json.load(f)
                        
                        # Convert JSON to text chunks
                        chunks = self._parse_drill_json(drill_data, file_path)
                else:
                    # Process text file
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    # Chunk document (simple: by paragraphs, max 500 chars)
                    chunks = self._chunk_document(content, max_chunk_size=500)
                
                # Create embeddings for chunks
                for i, chunk in enumerate(chunks):
                    if self.use_ollama_embeddings and OLLAMA_AVAILABLE:
                        embedding = self._get_ollama_embedding(chunk)
                    else:
                        embedding = self.embedder.encode(chunk).tolist()
                    
                    all_chunks.append(chunk)
                    all_embeddings.append(embedding)
                    all_metadata.append({
                        'source': str(file_path.relative_to(knowledge_dir)),
                        'chunk_id': i,
                        'type': 'drill' if file_path.suffix == '.json' else 'instruction'
                    })
                
                print(f"   OK {file_path.name}: {len(chunks)} chunks")
            
            except Exception as e:
                print(f"   FAILED Error processing {file_path}: {e}")
        
        # Add to vector database
        if CHROMADB_AVAILABLE and self.collection:
            if all_chunks:
                self.collection.add(
                    embeddings=all_embeddings,
                    documents=all_chunks,
                    metadatas=all_metadata,
                    ids=[f"chunk_{i}" for i in range(len(all_chunks))]
                )
                print(f"✅ Added {len(all_chunks)} chunks to vector database")
        else:
            # Simple storage
            self.collection['chunks'] = all_chunks
            self.collection['embeddings'] = all_embeddings
            self.collection['metadata'] = all_metadata
            print(f"✅ Stored {len(all_chunks)} chunks in memory")
            
            # Save to pickle file for persistence
            pickle_path = self.vector_db_path / 'vector_db.pkl'
            try:
                with open(pickle_path, 'wb') as f:
                    pickle.dump({
                        'chunks': all_chunks,
                        'embeddings': all_embeddings,
                        'metadata': all_metadata
                    }, f)
                print(f"✅ Saved {len(all_chunks)} chunks to {pickle_path}")
            except Exception as e:
                print(f"⚠️  Failed to save pickle file: {e}")
    
    def _parse_drill_json(self, drill_data: dict, file_path: Path) -> List[str]:
        """Parse JSON drill file and convert to text chunks"""
        chunks = []
        
        # Extract drill information - handle both old and new formats
        # New format: name, description, categories, errors_fixed, videoUrl
        # Old format: categories, description, videoUrl (name from filename)
        
        drill_name = drill_data.get('name', '')
        if not drill_name:
            # Old format: get name from filename
            drill_name = file_path.stem.replace('-', ' ').replace('_', ' ').title()
        
        categories = drill_data.get('categories', [])
        description = drill_data.get('description', '')
        video_url = drill_data.get('videoUrl', '')
        errors_fixed = drill_data.get('errors_fixed', [])  # New field
        
        # Build text representation
        text_parts = []
        
        # Add drill name
        text_parts.append(f"Drill: {drill_name}")
        
        # Add categories
        if categories:
            if isinstance(categories, list):
                cats_str = ', '.join(categories)
            else:
                cats_str = str(categories)
            text_parts.append(f"Categories: {cats_str}")
        
        # Add errors fixed (new field - very useful for matching)
        if errors_fixed:
            if isinstance(errors_fixed, list):
                errors_str = ', '.join(errors_fixed)
            else:
                errors_str = str(errors_fixed)
            text_parts.append(f"Errors Fixed: {errors_str}")
        
        # Add description
        if description:
            text_parts.append(f"\nDescription: {description}")
        
        # Add video URL if available
        if video_url:
            text_parts.append(f"\nVideo: {video_url}")
        
        # Combine into text
        full_text = "\n".join(text_parts)
        
        # Chunk the text (reuse existing chunking method)
        chunks = self._chunk_document(full_text, max_chunk_size=500)
        
        return chunks
    
    def _chunk_document(self, text: str, max_chunk_size: int = 500) -> List[str]:
        """Split document into chunks"""
        # Simple chunking: by paragraphs, then by sentences if needed
        paragraphs = text.split('\n\n')
        chunks = []
        
        for para in paragraphs:
            para = para.strip()
            if not para:
                continue
            
            if len(para) <= max_chunk_size:
                chunks.append(para)
            else:
                # Split long paragraphs by sentences
                sentences = para.split('. ')
                current_chunk = ""
                
                for sentence in sentences:
                    if len(current_chunk) + len(sentence) <= max_chunk_size:
                        current_chunk += sentence + ". "
                    else:
                        if current_chunk:
                            chunks.append(current_chunk.strip())
                        current_chunk = sentence + ". "
                
                if current_chunk:
                    chunks.append(current_chunk.strip())
        
        return chunks
    
    def _get_ollama_embedding(self, text: str) -> List[float]:
        """Get embedding using Ollama"""
        try:
            response = ollama.embeddings(
                model=self.model_name,  # Use same model as generation
                prompt=text
            )
            return response['embedding']
        except Exception as e:
            print(f"⚠️  Ollama embedding failed: {e}")
            # Fallback to sentence transformer if available
            if SENTENCE_TRANSFORMERS_AVAILABLE:
                embedder = SentenceTransformer(self.embedding_model_name)
                return embedder.encode(text).tolist()
            raise
    
    def _retrieve_relevant_knowledge(
        self,
        area: str,
        issue: str,
        swing_type: Optional[str] = None,
        all_weaknesses: Optional[List[Dict]] = None
    ) -> str:
        """Retrieve relevant golf instruction knowledge"""
        if not self.use_rag or not self.collection:
            return ""
        
        # Build query
        query_parts = [area.replace('_', ' '), issue]
        if swing_type:
            query_parts.append(swing_type.replace('_', ' '))
        if all_weaknesses:
            related_areas = [w['area'].replace('_', ' ') for w in all_weaknesses[:2]]
            query_parts.extend(related_areas)
        
        query = " ".join(query_parts) + " golf instruction technique drill"
        
        try:
            # Get query embedding
            if self.use_ollama_embeddings and OLLAMA_AVAILABLE:
                query_embedding = self._get_ollama_embedding(query)
            else:
                query_embedding = self.embedder.encode(query).tolist()
            
            # Retrieve similar chunks
            if CHROMADB_AVAILABLE and hasattr(self.collection, 'query'):
                results = self.collection.query(
                    query_embeddings=[query_embedding],
                    n_results=self.retrieval_k
                )
                
                # Format results
                if results['documents'] and len(results['documents'][0]) > 0:
                    context = "\n\n=== Relevant Golf Instruction Knowledge ===\n\n"
                    for i, (doc, metadata) in enumerate(zip(
                        results['documents'][0],
                        results['metadatas'][0]
                    ), 1):
                        source = metadata.get('source', 'Unknown')
                        context += f"[{i}] {doc}\n"
                        context += f"    Source: {source}\n\n"
                    return context
            else:
                # Simple similarity search
                if self.collection.get('embeddings') and len(self.collection['embeddings']) > 0:
                    # Calculate cosine similarity
                    import numpy as np
                    query_vec = np.array(query_embedding)
                    similarities = []
                    
                    for emb in self.collection['embeddings']:
                        emb_vec = np.array(emb)
                        similarity = np.dot(query_vec, emb_vec) / (
                            np.linalg.norm(query_vec) * np.linalg.norm(emb_vec)
                        )
                        similarities.append(similarity)
                    
                    # Get top k
                    top_k_indices = np.argsort(similarities)[-self.retrieval_k:][::-1]
                    
                    context = "\n\n=== Relevant Golf Instruction Knowledge ===\n\n"
                    for i, idx in enumerate(top_k_indices, 1):
                        doc = self.collection['chunks'][idx]
                        source = self.collection['metadata'][idx].get('source', 'Unknown')
                        context += f"[{i}] {doc}\n"
                        context += f"    Source: {source}\n\n"
                    return context
        
        except Exception as e:
            print(f"⚠️  Knowledge retrieval failed: {e}")
            return ""
        
        return ""
    
    def _build_prompt(
        self,
        area: str,
        issue: str,
        current_score: float,
        target_score: float,
        gap: float,
        impact_level: str,
        swing_type: Optional[str] = None,
        golfer_context: Optional[Dict] = None,
        all_quality_scores: Optional[np.ndarray] = None,
        all_weaknesses: Optional[List[Dict]] = None,
        all_strengths: Optional[List[Dict]] = None
    ) -> str:
        """Build prompt with RAG-retrieved knowledge"""
        # Get base prompt from parent
        prompt = super()._build_prompt(
            area, issue, current_score, target_score, gap, impact_level,
            swing_type, golfer_context, all_quality_scores,
            all_weaknesses, all_strengths
        )
        
        # Add retrieved knowledge if using RAG
        if self.use_rag:
            knowledge = self._retrieve_relevant_knowledge(
                area, issue, swing_type, all_weaknesses
            )
            
            if knowledge:
                # Insert knowledge before the final instruction
                prompt = prompt.replace(
                    "\nGenerate personalized feedback for this golfer:",
                    f"{knowledge}\n\nGenerate personalized feedback for this golfer:"
                )
        
        return prompt


# Convenience function
def create_rag_feedback_generator(
    knowledge_base_path: str = 'knowledge/golf_instruction/',
    use_rag: bool = True,
    llm_backend: str = 'ollama',
    model_name: str = 'llama3.2:1b',
    **kwargs
) -> RAGFeedbackGenerator:
    """Create RAG-enhanced feedback generator"""
    return RAGFeedbackGenerator(
        knowledge_base_path=knowledge_base_path,
        use_rag=use_rag,
        llm_backend=llm_backend,
        model_name=model_name,
        **kwargs
    )

