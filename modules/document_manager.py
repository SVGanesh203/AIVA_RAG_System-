# modules/document_manager.py
"""
Document loading and management module.
Handles document loading, FAISS operations, and document metadata.
"""

import os
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class AdvancedDocumentManager:
    """Document manager for handling document loading and search"""
    def __init__(self, embeddings_dir, embedding_model=None, load_on_init=False):
        self.embeddings_dir = embeddings_dir
        self.embedding_model = embedding_model
        self.loaded_documents = {}
        self.search_cache = {}
        self.dimension = None  # Will be set when loading documents
        
        # Load documents on initialization if requested
        if load_on_init:
            self.load_all_documents()

    def load_all_documents(self):
        """Load all available documents"""
        if not os.path.exists(self.embeddings_dir):
            logger.warning(f"Embeddings directory not found: {self.embeddings_dir}")
            return False
            
        # Clear existing documents to avoid duplicates
        self.loaded_documents.clear()
        self.search_cache.clear()

        folders = [f for f in os.listdir(self.embeddings_dir) 
                  if os.path.isdir(os.path.join(self.embeddings_dir, f))]

        if not folders:
            print("No document embeddings found.")
            return False

        print(f"Loading {len(folders)} documents...")
        loaded_count = 0
        
        for folder in folders:
            folder_path = os.path.join(self.embeddings_dir, folder)
            try:
                print(f"  Loading {folder}...")
                db, metadata, stats = self._load_single_document(folder_path)
                if db is not None:
                    self.loaded_documents[folder] = {
                        'db': db,
                        'metadata': metadata,
                        'folder_path': folder_path,
                        'statistics': stats
                    }
                    loaded_count += 1
                    print(f"    Loaded: {stats.get('chunks', 0)} chunks")
            except Exception as e:
                print(f"    Failed to load {folder}: {e}")

        print(f"Successfully loaded {loaded_count} documents")
        return loaded_count > 0

    def _load_single_document(self, folder_path):
        """Load a single document with error handling"""
        try:
            import pickle
            from langchain_community.vectorstores import FAISS
            
            # Check for required files
            faiss_path = os.path.join(folder_path, 'index.faiss')
            pkl_path = os.path.join(folder_path, 'index.pkl')
            
            if not os.path.exists(faiss_path) or not os.path.exists(pkl_path):
                print(f"    Missing index files in {os.path.basename(folder_path)}")
                return None, None, None
                
            # Initialize metadata and stats
            metadata = {'author': 'Unknown', 'title': 'Unknown', 'total_pages': 0}
            stats = {'chunks': 0, 'images': 0, 'tables': 0}
            
            # Load the FAISS index
            db = FAISS.load_local(
                folder_path,
                self.embedding_model,
                allow_dangerous_deserialization=True
            )
            print(f"    ✅ Loaded FAISS database successfully")
            
            # Load metadata file if it exists
            metadata_file = os.path.join(folder_path, "metadata.txt")
            if os.path.exists(metadata_file):
                try:
                    with open(metadata_file, "r", encoding="utf-8") as f:
                        for line in f:
                            line = line.strip()
                            if line.startswith("Author: "):
                                metadata['author'] = line.replace("Author: ", "")
                            elif line.startswith("Title: "):
                                metadata['title'] = line.replace("Title: ", "")
                            elif line.startswith("Total Pages: "):
                                try:
                                    metadata['total_pages'] = int(line.replace("Total Pages: ", ""))
                                except ValueError:
                                    metadata['total_pages'] = 0
                            elif line.startswith("Total Chunks: "):
                                try:
                                    stats['chunks'] = int(line.replace("Total Chunks: ", ""))
                                except ValueError:
                                    stats['chunks'] = 0
                except Exception as e:
                    print(f"    ⚠️ Error loading metadata: {e}")
            
            # Update stats with actual values if available
            if hasattr(db, 'index_to_docstore_id'):
                stats['chunks'] = len(db.index_to_docstore_id)
            
            return db, metadata, stats
            
        except Exception as e:
            import traceback
            print(f"    Error loading {os.path.basename(folder_path)}: {str(e)}")
            print(traceback.format_exc())
            return None, None, None

    def intelligent_search(self, query, k=18, search_type="hybrid", use_cache=True, use_mmr=True):
        """Simple search with optional cache and MMR for speed"""
        cache_key = (query, k, search_type)
        if use_cache and cache_key in self.search_cache:
            return self.search_cache[cache_key], self.loaded_documents

        all_results = []
        for doc_name, doc_data in self.loaded_documents.items():
            try:
                per_doc_k = max(1, k // max(1, len(self.loaded_documents)) + 2)
                docs = doc_data['db'].similarity_search(query, k=per_doc_k)
                print(f"✅ Search successful for {doc_name}: found {len(docs)} results")
                
                for doc in docs:
                    doc.metadata['source_document'] = doc_name
                    doc.metadata['document_title'] = doc_data['metadata']['title']
                    doc.metadata['document_author'] = doc_data['metadata']['author']
                all_results.extend(docs)
            except Exception as e:
                print(f"❌ Search error in {doc_name}: {e}")
                print(f"🔍 Debug info for {doc_name}:")
                print(f"   - DB type: {type(doc_data['db'])}")
                print(f"   - Has similarity_search: {hasattr(doc_data['db'], 'similarity_search')}")
                if hasattr(doc_data['db'], 'index'):
                    print(f"   - Index type: {type(doc_data['db'].index)}")
                    if hasattr(doc_data['db'].index, 'd'):
                        print(f"   - Index dimensions: {doc_data['db'].index.d}")

        results = all_results[:k]
        if use_cache:
            self.search_cache[cache_key] = results
        return results, self.loaded_documents


def get_compatible_embedding_model(faiss_dim):
    """Get an embedding model compatible with the FAISS index dimensions"""
    from langchain_huggingface import HuggingFaceEmbeddings
    from config import config
    
    try:
        dimension_to_model = {
            384: "sentence-transformers/all-MiniLM-L6-v2",
            768: "sentence-transformers/all-mpnet-base-v2", 
            512: "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
        }
        
        if faiss_dim in dimension_to_model:
            model_name = dimension_to_model[faiss_dim]
            print(f"🔄 Creating compatible embedding model: {model_name}")
            compatible_model = HuggingFaceEmbeddings(
                model_name=model_name,
                model_kwargs={'device': config.DEVICE},
                encode_kwargs={'normalize_embeddings': True}
            )
            return compatible_model
        else:
            print(f"⚠️ Unknown FAISS dimension {faiss_dim}, using default model")
            from app import embedding_model
            return embedding_model
    except Exception as e:
        print(f"⚠️ Could not create compatible embedding model: {e}")
        from app import embedding_model
        return embedding_model


def check_embedding_compatibility(db, embedding_model):
    """Check if the FAISS database is compatible with the current embedding model"""
    try:
        if hasattr(db, 'index') and hasattr(db.index, 'd'):
            faiss_dim = db.index.d
        else:
            return True
        
        test_text = "test"
        test_embedding = embedding_model.embed_query(test_text)
        current_dim = len(test_embedding)
        
        if faiss_dim != current_dim:
            print(f"⚠️ Embedding dimension mismatch: FAISS index has {faiss_dim} dimensions, current model has {current_dim}")
            return False
        
        return True
    except Exception as e:
        print(f"⚠️ Could not check embedding compatibility: {e}")
        return True