"""
Application state management.
This module holds the global application state to avoid circular imports.
"""
import os
from pathlib import Path
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import Ollama
from modules.document_manager import AdvancedDocumentManager
from modules.conversation import EnhancedConversationContext
from config import config as app_config

# Initialize system state
system_initialized = False
initialization_error = None
embedding_model = None
llm = None
doc_manager = None
context_manager = None
streaming_callback = None
agent_manager = None          # ← Stage 2: AIVA Agent Manager

def get_system_state():
    """Get the current system state"""
    return system_initialized, initialization_error, doc_manager, context_manager, app_config, embedding_model, llm

def get_agent_manager():
    """Return the global AgentManager instance (or None if not ready)."""
    return agent_manager

def initialize_system():
    """Initialize the RAG system on startup"""
    global system_initialized, initialization_error, doc_manager, context_manager, embedding_model, llm, streaming_callback
    
    try:
        print("Initializing RAG system...")
        
        # Initialize embedding model - use the same model that was used to create embeddings
        print(f"Loading embedding model on {app_config.DEVICE}...")
        try:
            # Use the first model from EMBED_MODEL_OPTIONS (same as embedding creator)
            embedding_model = HuggingFaceEmbeddings(
                model_name=app_config.EMBED_MODEL_OPTIONS[0],
                model_kwargs={'device': app_config.DEVICE},
                encode_kwargs={'normalize_embeddings': True}
            )
            print(f"✅ Embedding model loaded: {app_config.EMBED_MODEL_OPTIONS[0]} on {app_config.DEVICE}")
        except Exception as e1:
            try:
                # Fallback to the second model
                print(f"Warning: Failed to load primary model, trying fallback {app_config.EMBED_MODEL}...")
                embedding_model = HuggingFaceEmbeddings(
                    model_name=app_config.EMBED_MODEL,
                    model_kwargs={'device': app_config.DEVICE},
                    encode_kwargs={'normalize_embeddings': True}
                )
                print(f"✅ Fallback embedding model loaded: {app_config.EMBED_MODEL} on {app_config.DEVICE}")
            except Exception as e2:
                raise Exception(f"Failed to load any embedding model: {e2}")
        
        # Initialize LLM (optional - make it non-blocking)
        print("Initializing language model...")
        try:
            # Use local default if not in docker
            ollama_base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

            llm = Ollama(
                model=app_config.LLM_MODEL,
                temperature=0.1,
                base_url=ollama_base_url,
                num_ctx=getattr(app_config, "OLLAMA_NUM_CTX", 2048),
            )

            # Test connection disabled to prevent 50-second startup block
            # test_response = llm.invoke("Hello")
            print(f"✅ Language model initialized: {app_config.LLM_MODEL} @ {ollama_base_url}")

        except Exception as e:
            print(f"⚠️  Warning: LLM initialization failed: {e}")
            print("   The system will continue with document search only (no LLM responses)")
            llm = None  # Continue without LLM
        
        # Initialize managers
        os.makedirs(app_config.EMBEDDINGS_DIR, exist_ok=True)
        doc_manager = AdvancedDocumentManager(app_config.EMBEDDINGS_DIR, embedding_model=embedding_model)
        context_manager = EnhancedConversationContext()
        
        # Auto-load documents if available (with timeout to prevent blocking)
        print("Loading documents...")
        from threading import Thread
        import time
        
        def load_docs_with_timeout(timeout_seconds=60):
            """Load documents with a timeout to prevent blocking"""
            try:
                doc_manager.load_all_documents()
            except Exception as e:
                print(f"⚠️  Warning: Document loading failed: {e}")
        
        # Try to load documents with a timeout
        doc_thread = Thread(target=load_docs_with_timeout, daemon=True)
        doc_thread.start()
        doc_thread.join(timeout=60)  # Wait max 60 seconds for document loading
        
        if doc_thread.is_alive():
            print("⚠️  Warning: Document loading timed out (60s), continuing without full documents")
        
        system_initialized = True
        print("✅ System initialization complete")

        # ── Stage 2: Start AIVA Agent Manager & scheduler ──────────────────
        try:
            from agents.agent_manager import AgentManager
            from agents.scheduler import start_scheduler
            agent_manager = AgentManager(embedding_model)
            start_scheduler(agent_manager)
            print("✅ Agent Manager started (Stage 2 active)")
        except Exception as e:
            print(f"⚠️  Agent Manager failed to start: {e}")
            agent_manager = None
        
    except Exception as e:
        system_initialized = False
        initialization_error = str(e)
        print(f"❌ Error initializing system: {e}")
        raise