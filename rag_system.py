# rag_system.py - Main RAG Query Interface & Context Management

import os
import re
import time
import json
import hashlib
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from collections import defaultdict
from dataclasses import dataclass

# Import statements that were missing
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.llms import Ollama
from langchain_community.vectorstores import FAISS
from langchain.callbacks.base import BaseCallbackHandler

from config import config, templates

# Initialize global variables
embedding_model = None
llm = None

def ensure_directory(path: str):
    """Utility function to ensure directory exists"""
    os.makedirs(path, exist_ok=True)

def initialize_rag_system():
    """Initialize the RAG system components"""
    global embedding_model, llm
    
    print("Initializing System...")
    
    # Initialize embedding model
    print("Loading embedding model...")
    for i, model_name in enumerate(config.EMBED_MODEL_OPTIONS):
        try:
            print(f"  Trying {model_name}...")
            embedding_model = HuggingFaceEmbeddings(
                model_name=model_name,
                model_kwargs={'device': 'cpu'},
                encode_kwargs={'normalize_embeddings': True}
            )
            print(f"  Successfully loaded embedding model: {model_name}")
            break
        except Exception as e:
            print(f"  Failed to load {model_name}: {e}")
            if i == len(config.EMBED_MODEL_OPTIONS) - 1:
                raise Exception("Could not load any embedding model")
            continue
    
    # Initialize language model
    print("Initializing language model...")
    try:
        streaming_callback = EnhancedStreamingCallback()
        llm = Ollama(
            model=config.LLM_MODEL, 
            callbacks=[streaming_callback],
            temperature=0.3,
            num_predict=700,
            top_k=40,
            top_p=0.9
        )
        
        # Test the LLM connection
        test_response = llm.invoke("Hello")
        print(f"  Language model {config.LLM_MODEL} initialized and tested successfully")
        
    except Exception as e:
        print(f"  Language model initialization failed: {e}")
        print("  Please ensure Ollama is running and the model is installed.")
        print(f"  Try: ollama pull {config.LLM_MODEL}")
        raise Exception(f"Could not initialize language model: {e}")
    
    print("System initialization complete!")
    return True

# ---------- CONVERSATION CONTEXT MANAGEMENT ----------

@dataclass
class ConversationExchange:
    question: str
    answer: str
    timestamp: float
    sources: List[Dict]
    context_used: str = ""
    query_type: str = "general"
    confidence: float = 0.0

class EnhancedConversationContext:
    """Advanced conversation context with better reference tracking"""
    
    def __init__(self, max_history: int = 10):
        self.conversation_history: List[ConversationExchange] = []
        self.max_history = max_history
        self.document_context = {}
        self.entity_memory = {}
        self.topic_tracking = {}
        self.reference_chains = []
        
    def add_exchange(self, question: str, answer: str, sources: List = None, 
                    context_used: str = "", query_type: str = "general", 
                    confidence: float = 0.0):
        """Enhanced exchange tracking"""
        sources_info = []
        if sources:
            for source in sources[:6]:
                if hasattr(source, 'metadata'):
                    source_info = {
                        "type": source.metadata.get('type', 'text'),
                        "page": source.metadata.get('display_page', 0),
                        "source_doc": source.metadata.get('source_document', 'unknown'),
                        "has_table": source.metadata.get('has_table', False),
                        "image_type": source.metadata.get('image_type', None),
                        "chart_analysis": source.metadata.get('chart_analysis', {}),
                    }
                    sources_info.append(source_info)
        
        # Enhanced entity and topic tracking
        self._extract_and_store_entities(question, answer)
        self._track_topics(question, answer)
        self._update_reference_chains(question, len(self.conversation_history))
        
        exchange = ConversationExchange(
            question=question,
            answer=answer,
            timestamp=time.time(),
            sources=sources_info,
            context_used=context_used,
            query_type=query_type,
            confidence=confidence
        )
        
        self.conversation_history.append(exchange)
        
        if len(self.conversation_history) > self.max_history:
            removed = self.conversation_history.pop(0)
            self._cleanup_references(0)
    
    def _extract_and_store_entities(self, question: str, answer: str):
        """Advanced entity extraction and storage"""
        patterns = {
            'person_names': r'\b[A-Z][a-z]+\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?\b',
            'job_titles': r'\b(?:CEO|CTO|CFO|SVP|VP|President|Manager|Director|Lead|Senior|Junior|Associate)\b',
            'departments': r'\b(?:Engineering|Marketing|Sales|HR|Finance|Operations|IT|Legal|R&D)\b',
            'companies': r'\b[A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+)*(?:\s+(?:Inc|LLC|Ltd|Corp|Corporation))\b',
            'numbers': r'\b\d+(?:\.\d+)?(?:%|\$|k|M|B)?\b',
            'dates': r'\b(?:\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\w+\s+\d{1,2},?\s+\d{4})\b'
        }
        
        current_time = time.time()
        
        for text in [question, answer]:
            for entity_type, pattern in patterns.items():
                matches = re.findall(pattern, text, re.IGNORECASE)
                for match in matches:
                    entity_key = f"{entity_type}:{match.lower()}"
                    self.entity_memory[entity_key] = {
                        'value': match,
                        'type': entity_type,
                        'last_mentioned': current_time,
                        'mention_count': self.entity_memory.get(entity_key, {}).get('mention_count', 0) + 1
                    }
    
    def _track_topics(self, question: str, answer: str):
        """Track conversation topics for better context"""
        topic_keywords = {
            'organizational': ['organization', 'structure', 'hierarchy', 'team', 'department'],
            'financial': ['budget', 'cost', 'revenue', 'profit', 'expense', 'financial'],
            'personnel': ['staff', 'employee', 'person', 'people', 'individual', 'team member'],
            'technical': ['system', 'process', 'procedure', 'method', 'technology'],
            'temporal': ['when', 'time', 'date', 'period', 'duration', 'schedule']
        }
        
        combined_text = f"{question} {answer}".lower()
        current_time = time.time()
        
        for topic, keywords in topic_keywords.items():
            relevance_score = sum(1 for keyword in keywords if keyword in combined_text)
            
            if relevance_score > 0:
                if topic not in self.topic_tracking:
                    self.topic_tracking[topic] = {
                        'first_mentioned': current_time,
                        'last_mentioned': current_time,
                        'relevance_score': relevance_score,
                        'mention_count': 1
                    }
                else:
                    self.topic_tracking[topic]['last_mentioned'] = current_time
                    self.topic_tracking[topic]['relevance_score'] += relevance_score
                    self.topic_tracking[topic]['mention_count'] += 1
    
    def _update_reference_chains(self, question: str, exchange_index: int):
        """Track reference relationships between questions"""
        reference_patterns = [
            r'\b(?:this|that|these|those|it|they|them)\b',
            r'\b(?:previous|earlier|above|before|mentioned|discussed)\b',
            r'\b(?:same|similar|related|also|additionally)\b'
        ]
        
        question_lower = question.lower()
        has_references = any(re.search(pattern, question_lower) for pattern in reference_patterns)
        
        if has_references and self.conversation_history:
            self.reference_chains.append({
                'current_index': exchange_index,
                'references_index': len(self.conversation_history) - 1,
                'reference_strength': sum(1 for pattern in reference_patterns 
                                        if re.search(pattern, question_lower)),
                'timestamp': time.time()
            })
    
    def _cleanup_references(self, removed_index: int):
        """Clean up reference chains when removing old exchanges"""
        self.reference_chains = [
            ref for ref in self.reference_chains 
            if ref['current_index'] > removed_index and ref['references_index'] > removed_index
        ]
        
        for ref in self.reference_chains:
            if ref['current_index'] > removed_index:
                ref['current_index'] -= 1
            if ref['references_index'] > removed_index:
                ref['references_index'] -= 1
    
    def get_enhanced_context_for_query(self, query: str) -> str:
        """Generate sophisticated context with smart reference resolution"""
        if not self.conversation_history:
            return ""
        
        context_analysis = self._analyze_context_needs(query)
        
        if not context_analysis['needs_context']:
            return ""
        
        context_parts = ["=== ENHANCED CONVERSATION CONTEXT ==="]
        
        relevant_exchanges = self._select_relevant_exchanges(query, context_analysis)
        
        for i, exchange in enumerate(relevant_exchanges):
            context_parts.append(f"Previous Exchange {i + 1} ({exchange.query_type}):")
            context_parts.append(f"Q: {exchange.question}")
            
            if len(exchange.answer) > 300:
                if context_analysis['query_type'] == exchange.query_type:
                    answer_excerpt = exchange.answer[:400] + "..."
                else:
                    answer_excerpt = exchange.answer[:200] + "..."
            else:
                answer_excerpt = exchange.answer
            
            context_parts.append(f"A: {answer_excerpt}")
            
            if exchange.sources:
                source_types = set(s.get('type', 'unknown') for s in exchange.sources)
                context_parts.append(f"   (Referenced: {', '.join(source_types)})")
        
        relevant_entities = self._get_relevant_entities(query)
        if relevant_entities:
            context_parts.append(f"Previously mentioned entities: {', '.join(relevant_entities[:5])}")
        
        active_topics = self._get_active_topics()
        if active_topics:
            context_parts.append(f"Active discussion topics: {', '.join(active_topics[:3])}")
        
        context_parts.append("=== END CONTEXT ===\n")
        context_parts.append("Current question:")
        
        return "\n".join(context_parts)
    
    def _analyze_context_needs(self, query: str) -> Dict:
        """Analyze what type of context the query needs"""
        query_lower = query.lower()
        
        reference_patterns = [
            r'\b(?:it|this|that|these|those)\b',
            r'\b(?:he|she|they|him|her|them)\b',
            r'\b(?:previous|earlier|above|before|mentioned|discussed)\b',
            r'\b(?:same|similar|related|also|additionally)\b'
        ]
        
        has_references = any(re.search(pattern, query_lower) for pattern in reference_patterns)
        
        query_type = "general"
        type_indicators = {
            'table': ['who is', 'list', 'show me', 'table', 'names', 'staff', 'employees'],
            'chart': ['chart', 'graph', 'data', 'trend', 'analysis', 'statistics'],
            'image': ['image', 'picture', 'photo', 'visual', 'diagram'],
            'comparison': ['compare', 'difference', 'similar', 'versus', 'vs']
        }
        
        for qtype, indicators in type_indicators.items():
            if any(indicator in query_lower for indicator in indicators):
                query_type = qtype
                break
        
        mentioned_entities = [
            entity_data['value'] for entity_key, entity_data in self.entity_memory.items() 
            if any(word in query_lower for word in entity_data['value'].lower().split())
        ]
        
        return {
            'needs_context': has_references or mentioned_entities,
            'query_type': query_type,
            'has_references': has_references,
            'mentioned_entities': mentioned_entities,
            'reference_strength': sum(1 for pattern in reference_patterns 
                                    if re.search(pattern, query_lower))
        }
    
    def _select_relevant_exchanges(self, query: str, context_analysis: Dict) -> List[ConversationExchange]:
        """Intelligently select most relevant previous exchanges"""
        if not self.conversation_history:
            return []
        
        scored_exchanges = []
        query_words = set(query.lower().split())
        
        for i, exchange in enumerate(self.conversation_history[-5:], start=len(self.conversation_history)-5):
            score = 0
            
            # Recency bonus (more recent = higher score)
            score += (i + 1) * 10
            
            # Same query type bonus
            if exchange.query_type == context_analysis['query_type']:
                score += 20
            
            # Word overlap bonus
            exchange_words = set((exchange.question + " " + exchange.answer).lower().split())
            overlap = len(query_words & exchange_words)
            score += overlap * 2
            
            # Entity mention bonus
            for entity in context_analysis.get('mentioned_entities', []):
                if entity.lower() in exchange.answer.lower():
                    score += 15
            
            # Reference chain bonus
            for ref_chain in self.reference_chains:
                if ref_chain['references_index'] == i:
                    score += ref_chain['reference_strength'] * 10
            
            scored_exchanges.append((score, exchange))
        
        # Sort by score and return top exchanges
        scored_exchanges.sort(key=lambda x: x[0], reverse=True)
        return [exchange for score, exchange in scored_exchanges[:3]]
    
    def _get_relevant_entities(self, query: str) -> List[str]:
        """Get entities relevant to current query"""
        query_lower = query.lower()
        relevant = []
        
        current_time = time.time()
        for entity_key, entity_data in self.entity_memory.items():
            entity_words = entity_data['value'].lower().split()
            if any(word in query_lower for word in entity_words):
                relevant.append(entity_data['value'])
            elif (current_time - entity_data['last_mentioned'] < 300 and  # 5 minutes
                  entity_data['mention_count'] > 1):
                relevant.append(entity_data['value'])
        
        return relevant[:20]
    
    def _get_active_topics(self) -> List[str]:
        """Get currently active discussion topics"""
        current_time = time.time()
        active_topics = []
        
        for topic, data in self.topic_tracking.items():
            if (current_time - data['last_mentioned'] < 600 or  # 10 minutes
                data['mention_count'] > 2):
                active_topics.append(topic)
        
        return active_topics
    
    def clear_history(self):
        """Clear all conversation data"""
        self.conversation_history = []
        self.entity_memory = {}
        self.topic_tracking = {}
        self.reference_chains = []

# ---------- STREAMING CALLBACK ----------

class EnhancedStreamingCallback(BaseCallbackHandler):
    def __init__(self):
        self.text = ""
        self.is_streaming = False
        self.start_time = None
    
    def on_llm_start(self, serialized, prompts, **kwargs):
        self.start_time = time.time()
    
    def on_llm_new_token(self, token: str, **kwargs) -> None:
        if not self.is_streaming:
            print("\nAnswer: ", end="", flush=True)
            self.is_streaming = True
        print(token, end="", flush=True)
        self.text += token
        #time.sleep(0.008)
    
    def on_llm_end(self, response, **kwargs) -> None:
        if self.is_streaming:
            print()
            if self.start_time:
                duration = time.time() - self.start_time
                print(f"(Response generated in {duration:.1f}s)")
        self.is_streaming = False
    
    def on_llm_error(self, error, **kwargs) -> None:
        print(f"\nLLM Error: {error}")
        self.is_streaming = False
    
    def reset(self):
        self.text = ""
        self.is_streaming = False
        self.start_time = None

# ---------- DOCUMENT MANAGER ----------

class AdvancedDocumentManager:
    """Advanced document management with intelligent search and caching"""
    
    def __init__(self, embeddings_dir: str):
        self.embeddings_dir = embeddings_dir
        self.loaded_documents = {}
        self.search_cache = {}
        self.document_statistics = {}
        
    def load_all_documents(self) -> bool:
        """Load all available documents with enhanced metadata"""
        if not os.path.exists(self.embeddings_dir):
            print(f"Embeddings directory not found: {self.embeddings_dir}")
            return False
        
        folders = [f for f in os.listdir(self.embeddings_dir) 
                  if os.path.isdir(os.path.join(self.embeddings_dir, f))]
        
        if not folders:
            print("No document embeddings found.")
            return False
        
        print(f"\nLoading {len(folders)} documents...")
        
        loaded_count = 0
        for folder in folders:
            folder_path = os.path.join(self.embeddings_dir, folder)
            try:
                print(f"  Loading {folder}...")
                db, metadata, stats = self._load_single_document_enhanced(folder_path)
                if db is not None:
                    self.loaded_documents[folder] = {
                        'db': db,
                        'metadata': metadata,
                        'folder_path': folder_path,
                        'statistics': stats
                    }
                    self.document_statistics[folder] = stats
                    loaded_count += 1
                    print(f"    ✅ Loaded: {stats['chunks']} chunks, {stats['images']} images, {stats['tables']} tables")
            except Exception as e:
                print(f"    ❌ Failed to load {folder}: {e}")
        
        print(f"\n✅ Successfully loaded {loaded_count} documents")
        self._print_loading_summary()
        return loaded_count > 0
    
    def _load_single_document_enhanced(self, folder_path: str) -> Tuple[Any, Dict, Dict]:
        """Load document with comprehensive metadata and statistics"""
        try:
            # Load FAISS database
            db = FAISS.load_local(folder_path, embedding_model, allow_dangerous_deserialization=True)
            
            # Initialize metadata and statistics
            metadata = {'author': 'Unknown', 'title': 'Unknown', 'images': [], 'total_pages': 0}
            stats = {'chunks': 0, 'images': 0, 'tables': 0, 'text_chunks': 0, 'image_chunks': 0, 'table_chunks': 0}
            
            # Load basic metadata
            metadata_file = os.path.join(folder_path, "metadata.txt")
            if os.path.exists(metadata_file):
                with open(metadata_file, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if line.startswith("Author: "):
                            metadata['author'] = line.replace("Author: ", "")
                        elif line.startswith("Title: "):
                            metadata['title'] = line.replace("Title: ", "")
                        elif line.startswith("Total Pages: "):
                            metadata['total_pages'] = int(line.replace("Total Pages: ", ""))
                        elif line.startswith("Total Chunks: "):
                            stats['chunks'] = int(line.replace("Total Chunks: ", ""))
                        elif line.startswith("Images: "):
                            stats['images'] = int(line.replace("Images: ", ""))
                        elif line.startswith("Tables: "):
                            stats['tables'] = int(line.replace("Tables: ", ""))
            
            # Load image metadata
            images_metadata_file = os.path.join(folder_path, "images", "images_metadata.json")
            if os.path.exists(images_metadata_file):
                try:
                    with open(images_metadata_file, 'r', encoding='utf-8') as f:
                        metadata['images'] = json.load(f)
                        stats['images'] = len(metadata['images'])
                except Exception as e:
                    print(f"    Warning: Could not load image metadata: {e}")
            
            # Analyze document content for better statistics
            if hasattr(db, 'docstore') and hasattr(db.docstore, '_dict'):
                docs = list(db.docstore._dict.values())
                stats['chunks'] = len(docs)
                
                for doc in docs:
                    doc_type = doc.metadata.get('type', 'text')
                    if doc_type == 'text':
                        stats['text_chunks'] += 1
                    elif doc_type == 'image':
                        stats['image_chunks'] += 1
                    elif doc_type == 'table':
                        stats['table_chunks'] += 1
            
            return db, metadata, stats
            
        except Exception as e:
            print(f"    Error loading {folder_path}: {e}")
            return None, {}, {}
    
    def _print_loading_summary(self):
        """Print comprehensive loading summary"""
        if not self.document_statistics:
            return
        
        total_chunks = sum(stats['chunks'] for stats in self.document_statistics.values())
        total_images = sum(stats['images'] for stats in self.document_statistics.values())
        total_tables = sum(stats['tables'] for stats in self.document_statistics.values())
        
        print(f"\n📊 Loading Summary:")
        print(f"   Total chunks: {total_chunks}")
        print(f"   Total images: {total_images}")  
        print(f"   Total tables: {total_tables}")
        print(f"   Documents with images: {sum(1 for stats in self.document_statistics.values() if stats['images'] > 0)}")
        print(f"   Documents with tables: {sum(1 for stats in self.document_statistics.values() if stats['tables'] > 0)}")
    
    def intelligent_search(self, query: str, k: int = 25, search_type: str = "hybrid") -> Tuple[List, Dict]:
        """Enhanced search with multiple strategies"""
        # Check cache first
        cache_key = f"{query}_{k}_{search_type}"
        if cache_key in self.search_cache:
            return self.search_cache[cache_key]
        
        all_results = []
        query_analysis = self._analyze_query(query)
        
        # Determine search strategy based on query type
        if query_analysis['is_ui_query']:
            # UI element query - prioritize images with UI metadata
            k_per_doc = max(12, k // len(self.loaded_documents)) if self.loaded_documents else k
            if query_analysis['is_popup_query']:
                search_query = f"{query} popup dialog window modal snapshot POPUP DIALOG SCREEN image screenshot"
            elif query_analysis['is_screen_query']:
                search_query = f"{query} screen page appears SCREEN PAGE image screenshot interface"
            elif query_analysis['is_tab_query']:
                search_query = f"{query} tab tabs TABS interface navigation image screenshot"
            elif query_analysis['is_button_query']:
                search_query = f"{query} button buttons BUTTONS click interface image screenshot"
            else:
                search_query = f"{query} UI interface popup dialog screen tab button screenshot image"
        elif query_analysis['is_page_query'] and query_analysis['has_specific_pages']:
            # Page-specific query - prioritize images and content from specific pages
            k_per_doc = max(10, k // len(self.loaded_documents)) if self.loaded_documents else k
            page_nums = query_analysis['page_numbers']
            page_refs = " ".join([f"page {p}" for p in page_nums])
            search_query = f"{query} {page_refs} page-specific content images visual"
        elif query_analysis['is_summary_query']:
            k_per_doc = max(15, k // len(self.loaded_documents)) if self.loaded_documents else k
            search_query = f"{query} overview main points key concepts introduction conclusion summary"
        elif query_analysis['is_experiment_query']:
            k_per_doc = max(20, k // len(self.loaded_documents)) if self.loaded_documents else k
            search_query = f"{query} experiment aim objective procedure steps tutorial lab practical exercise"
        elif query_analysis['is_table_query']:
            k_per_doc = max(8, k // len(self.loaded_documents)) if self.loaded_documents else k
            search_query = f"{query} table name title role signature department"
        elif query_analysis['is_chart_query']:
            k_per_doc = max(6, k // len(self.loaded_documents)) if self.loaded_documents else k
            search_query = f"{query} chart graph data visualization trend analysis"
        elif query_analysis['is_image_query']:
            k_per_doc = max(6, k // len(self.loaded_documents)) if self.loaded_documents else k
            search_query = f"{query} image visual diagram figure"
        else:
            k_per_doc = max(4, k // len(self.loaded_documents)) if self.loaded_documents else k
            search_query = query
        
        # Search each document
        for doc_name, doc_data in self.loaded_documents.items():
            try:
                docs = doc_data['db'].similarity_search(search_query, k=k_per_doc)
                
                for doc in docs:
                    doc.metadata['source_document'] = doc_name
                    doc.metadata['document_title'] = doc_data['metadata']['title']
                    doc.metadata['document_author'] = doc_data['metadata']['author']
                
                all_results.extend(docs)
                
                # If table query, also search with entity-specific queries
                if query_analysis['is_table_query'] and query_analysis['entities']:
                    for entity in query_analysis['entities'][:3]:
                        entity_docs = doc_data['db'].similarity_search(f"{entity} name title role", k=2)
                        for doc in entity_docs:
                            if doc not in all_results:
                                doc.metadata['source_document'] = doc_name
                                doc.metadata['document_title'] = doc_data['metadata']['title']
                                doc.metadata['document_author'] = doc_data['metadata']['author']
                                all_results.append(doc)
                
            except Exception as e:
                print(f"Search error in {doc_name}: {e}")
        
        # Enhanced post-processing and ranking
        processed_results = self._post_process_search_results(all_results, query_analysis, k)
        
        # Cache results
        self.search_cache[cache_key] = (processed_results, self.loaded_documents)
        
        # Cleanup cache if too large
        if len(self.search_cache) > 100:
            oldest_keys = list(self.search_cache.keys())[:50]
            for key in oldest_keys:
                del self.search_cache[key]
        
        return processed_results, self.loaded_documents
    
    def _analyze_query(self, query: str) -> Dict:
        """Analyze query to determine optimal search strategy"""
        query_lower = query.lower()
        
        table_indicators = ['who is', 'list', 'show me', 'table', 'names', 'staff', 'employees', 'people', 'personnel', 'directory']
        chart_indicators = ['chart', 'graph', 'data', 'trend', 'analysis', 'statistics', 'visualization', 'plot']
        image_indicators = ['image', 'picture', 'photo', 'visual', 'diagram', 'figure']
        summary_indicators = ['summarize', 'summary', 'overview', 'brief', 'outline', 'main points', 'key points', 'what is this about', 'tell me about']
        experiment_indicators = ['experiment', 'lab', 'practical', 'exercise', 'tutorial', 'step by step', 'procedure', 'aim', 'objective']
        page_indicators = ['page', 'on page', 'in page', 'page number', 'what is on page', 'explain page', 'show page', 'page content']
        # NEW: UI element indicators
        popup_indicators = ['popup', 'dialog', 'pop up', 'pop-up', 'modal', 'window']
        screen_indicators = ['screen', 'page appears', 'which screen', 'what screen', 'screen name']
        tab_indicators = ['tab', 'tabs', 'which tab', 'available tabs', 'tab names']
        button_indicators = ['button', 'buttons', 'click', 'which button', 'button text']
        
        is_table_query = any(indicator in query_lower for indicator in table_indicators)
        is_chart_query = any(indicator in query_lower for indicator in chart_indicators)
        is_image_query = any(indicator in query_lower for indicator in image_indicators)
        is_summary_query = any(indicator in query_lower for indicator in summary_indicators)
        is_experiment_query = any(indicator in query_lower for indicator in experiment_indicators)
        is_page_query = any(indicator in query_lower for indicator in page_indicators)
        # NEW: UI element query detection
        is_popup_query = any(indicator in query_lower for indicator in popup_indicators)
        is_screen_query = any(indicator in query_lower for indicator in screen_indicators)
        is_tab_query = any(indicator in query_lower for indicator in tab_indicators)
        is_button_query = any(indicator in query_lower for indicator in button_indicators)
        is_ui_query = is_popup_query or is_screen_query or is_tab_query or is_button_query
        
        # Entity extraction for targeted search
        entities = []
        entity_patterns = [
            r'\b[A-Z][a-z]+\s+[A-Z][a-z]+\b',
            r'\b(?:CEO|CTO|CFO|SVP|VP|President|Manager|Director|Lead)\b'
        ]
        
        for pattern in entity_patterns:
            matches = re.findall(pattern, query)
            entities.extend(matches)
        
        # Extract page numbers from query
        page_numbers = []
        page_patterns = [
            r'page\s+(\d+)',
            r'on\s+page\s+(\d+)',
            r'in\s+page\s+(\d+)',
            r'page\s+number\s+(\d+)',
            r'what\s+is\s+on\s+page\s+(\d+)',
            r'explain\s+page\s+(\d+)',
            r'show\s+page\s+(\d+)'
        ]
        
        for pattern in page_patterns:
            matches = re.findall(pattern, query_lower)
            page_numbers.extend([int(match) for match in matches])
        
        return {
            'is_table_query': is_table_query,
            'is_chart_query': is_chart_query,
            'is_image_query': is_image_query,
            'is_summary_query': is_summary_query,
            'is_experiment_query': is_experiment_query,
            'is_page_query': is_page_query,
            'page_numbers': page_numbers,
            'entities': entities,
            'query_complexity': len(query.split()),
            'has_specific_entities': len(entities) > 0,
            'has_specific_pages': len(page_numbers) > 0,
            # NEW: UI element query flags
            'is_popup_query': is_popup_query,
            'is_screen_query': is_screen_query,
            'is_tab_query': is_tab_query,
            'is_button_query': is_button_query,
            'is_ui_query': is_ui_query
        }
    
    def _post_process_search_results(self, results: List, query_analysis: Dict, target_k: int) -> List:
        """Advanced post-processing of search results"""
        if not results:
            return []
        
        scored_results = []
        
        for doc in results:
            score = 0
            doc_type = doc.metadata.get('type', 'text')
            
            score += 1
            
            # Type matching bonus
            if query_analysis['is_ui_query']:
                # NEW: UI element query bonuses
                if doc_type == 'image':
                    score += 2  # Base bonus for images in UI queries
                    if query_analysis['is_popup_query'] and (doc.metadata.get('has_popup') or doc.metadata.get('popup_names')):
                        score += 5  # High bonus for images with popups
                    if query_analysis['is_screen_query'] and doc.metadata.get('screen_names'):
                        score += 5  # High bonus for images with screen names
                    if query_analysis['is_tab_query'] and doc.metadata.get('detected_tabs'):
                        score += 5  # High bonus for images with tabs
                    if query_analysis['is_button_query'] and doc.metadata.get('detected_buttons'):
                        score += 5  # High bonus for images with buttons
                    if doc.metadata.get('is_screenshot'):
                        score += 3  # Bonus for detected screenshots
                    if doc.metadata.get('has_annotations') or doc.metadata.get('has_callouts'):
                        score += 2  # Bonus for annotated images
            elif query_analysis['is_page_query'] and query_analysis['has_specific_pages']:
                # High bonus for page-specific queries matching requested pages
                doc_page = doc.metadata.get('display_page', doc.metadata.get('page_number', 0))
                if doc_page in query_analysis['page_numbers']:
                    score += 5  # Highest bonus for exact page match
                    if doc_type == 'image':
                        score += 3  # Extra bonus for images on requested pages
            elif query_analysis['is_table_query'] and (doc_type == 'table' or doc.metadata.get('has_table')):
                score += 3
            elif query_analysis['is_chart_query'] and doc_type == 'image' and 'chart' in doc.page_content.lower():
                score += 3
            elif query_analysis['is_image_query'] and doc_type == 'image':
                score += 2
            elif query_analysis['is_experiment_query'] and ('experiment' in doc.page_content.lower() or doc.metadata.get('is_experiment')):
                score += 4  # High bonus for experiment queries
            
            # Entity matching bonus
            if query_analysis['entities']:
                content_lower = doc.page_content.lower()
                for entity in query_analysis['entities']:
                    if entity.lower() in content_lower:
                        score += 2
            
            # Content quality bonus
            content_length = len(doc.page_content)
            if 100 < content_length < 3000:
                score += 1
            
            if doc.metadata.get('is_enhanced_table'):
                score += 1
            
            scored_results.append((score, doc))
        
        scored_results.sort(key=lambda x: x[0], reverse=True)
        
        # Deduplicate while preserving top results
        seen_content = set()
        final_results = []
        
        for score, doc in scored_results:
            content_hash = hashlib.md5(doc.page_content.encode()).hexdigest()
            if content_hash not in seen_content:
                seen_content.add(content_hash)
                final_results.append(doc)
                
                if len(final_results) >= target_k:
                    break
        
        return final_results

# ---------- QUERY PROCESSING ----------

def process_query_with_advanced_context(query: str, doc_manager: AdvancedDocumentManager, 
                                       context_manager: EnhancedConversationContext, 
                                       streaming_callback: EnhancedStreamingCallback) -> Tuple[str, List, str]:
    """Advanced query processing with intelligent context management"""
    
    start_time = time.time()
    
    # Get enhanced conversation context
    conversation_context = context_manager.get_enhanced_context_for_query(query)
    
    # Analyze query type for optimal processing
    query_analysis = doc_manager._analyze_query(query)
    query_type = "general"
    
    if query_analysis['is_ui_query']:
        query_type = "ui_element"
        if query_analysis['is_popup_query']:
            query_type = "popup"
        elif query_analysis['is_screen_query']:
            query_type = "screen"
        elif query_analysis['is_tab_query']:
            query_type = "tab"
        elif query_analysis['is_button_query']:
            query_type = "button"
    elif query_analysis['is_page_query'] and query_analysis['has_specific_pages']:
        query_type = "page_specific"
    elif query_analysis['is_summary_query']:
        query_type = "summary"
    elif query_analysis['is_table_query']:
        query_type = "table"
    elif query_analysis['is_chart_query']:
        query_type = "chart"
    elif query_analysis['is_image_query']:
        query_type = "image"
    
    # Intelligent search with type-specific optimization
    if query_type in ["popup", "screen", "tab", "button", "ui_element"]:
        search_k = 15  # More results for UI queries to find screenshots with UI elements
    elif query_type == "page_specific":
        search_k = 12  # More results for page-specific queries to ensure we get images
    elif query_type == "summary":
        search_k = 15
    else:
        search_k = 5
    search_results, source_docs = doc_manager.intelligent_search(query, k=search_k, search_type="hybrid")
    
    if not search_results:
        error_msg = "No relevant information found in the loaded documents."
        context_manager.add_exchange(query, error_msg, [], "", query_type, 0.0)
        return error_msg, [], query_type
    
    # Enhanced result prioritization
    prioritized_results = search_results
    
    if query_type == "summary":
        # For summary queries, prioritize diverse content from different parts of documents
        prioritized_results = search_results[:15]  # Use more results for comprehensive summary
        
    elif query_type == "table":
        table_results = [doc for doc in search_results 
                        if (doc.metadata.get('has_table', False) or 
                            doc.metadata.get('type') == 'table' or
                            doc.metadata.get('is_enhanced_table', False))]
        other_results = [doc for doc in search_results if doc not in table_results]
        prioritized_results = table_results[:10] + other_results[:5]
        
    elif query_type == "chart":
        chart_results = [doc for doc in search_results 
                        if (doc.metadata.get('type') == 'image' and 
                            'chart' in doc.page_content.lower())]
        other_results = [doc for doc in search_results if doc not in chart_results]
        prioritized_results = chart_results[:8] + other_results[:7]
    
    elif query_type in ["popup", "screen", "tab", "button", "ui_element"]:
        # NEW: For UI element queries, prioritize images with UI metadata
        ui_results = []
        other_results = []
        
        for doc in search_results:
            if doc.metadata.get('type') == 'image':
                has_ui = False
                if query_type == "popup" and (doc.metadata.get('has_popup') or doc.metadata.get('popup_names')):
                    has_ui = True
                elif query_type == "screen" and doc.metadata.get('screen_names'):
                    has_ui = True
                elif query_type == "tab" and doc.metadata.get('detected_tabs'):
                    has_ui = True
                elif query_type == "button" and doc.metadata.get('detected_buttons'):
                    has_ui = True
                elif query_type == "ui_element" and doc.metadata.get('is_screenshot'):
                    has_ui = True
                
                if has_ui:
                    ui_results.append(doc)
                else:
                    other_results.append(doc)
            else:
                other_results.append(doc)
        
        # Prioritize: UI images first, then other images, then text
        prioritized_results = ui_results[:10] + other_results[:5]
    
    elif query_type == "page_specific":
        # For page-specific queries, prioritize content from the requested pages
        requested_pages = query_analysis['page_numbers']
        page_specific_results = []
        image_results = []
        other_results = []
        
        for doc in search_results:
            doc_page = doc.metadata.get('display_page', doc.metadata.get('page_number', 0))
            if doc_page in requested_pages:
                if doc.metadata.get('type') == 'image':
                    image_results.append(doc)
                else:
                    page_specific_results.append(doc)
            else:
                other_results.append(doc)
        
        # Prioritize: images from requested pages, then other content from requested pages, then other content
        prioritized_results = image_results[:6] + page_specific_results[:4] + other_results[:2]
    
    # Build enhanced context
    context_parts = []
    if query_type in ["popup", "screen", "tab", "button", "ui_element"]:
        max_context_docs = 10  # More context for UI queries
    elif query_type == "page_specific":
        max_context_docs = 8  # More context for page-specific queries
    elif query_type == "summary":
        max_context_docs = 10
    else:
        max_context_docs = 5
    for doc in prioritized_results[:max_context_docs]:
        doc_name = doc.metadata.get('source_document', 'Unknown')
        doc_title = doc.metadata.get('document_title', 'Unknown')
        doc_author = doc.metadata.get('document_author', 'Unknown')
        page_num = doc.metadata.get('display_page', 'Unknown')
        doc_type = doc.metadata.get('type', 'text')
        
        source_header = f"[Source: {doc_name}]"
        
        if doc.metadata.get('is_enhanced_table'):
            source_header += " [ENHANCED TABLE DATA]"
        if doc.metadata.get('chart_analysis'):
            source_header += " [CHART ANALYSIS AVAILABLE]"
        
        # NEW: Add UI element metadata to context
        if doc_type == 'image' and query_type in ["popup", "screen", "tab", "button", "ui_element"]:
            ui_info = []
            if doc.metadata.get('popup_names'):
                ui_info.append(f"POPUP: {', '.join(doc.metadata['popup_names'])}")
            if doc.metadata.get('screen_names'):
                ui_info.append(f"SCREEN: {', '.join(doc.metadata['screen_names'])}")
            if doc.metadata.get('detected_tabs'):
                ui_info.append(f"TABS: {', '.join(doc.metadata['detected_tabs'])}")
            if doc.metadata.get('detected_buttons'):
                ui_info.append(f"BUTTONS: {', '.join(doc.metadata['detected_buttons'])}")
            if doc.metadata.get('annotations'):
                ui_info.append(f"ANNOTATIONS: {doc.metadata['annotations']}")
            if doc.metadata.get('callouts'):
                ui_info.append(f"CALLOUTS: {doc.metadata['callouts']}")
            
            if ui_info:
                source_header += f" [UI ELEMENTS: {' | '.join(ui_info)}]"
        
        context_parts.append(f"{source_header}\n{doc.page_content}")
    
    combined_context = "\n\n".join(context_parts)
    
    # Select appropriate template based on query type
    if query_type in ["popup", "screen", "tab", "button", "ui_element"]:
        template = templates.UI_ELEMENT_TEMPLATE
        formatted_prompt = template.format(
            context=combined_context,
            conversation_context=conversation_context,
            question=query
        )
    elif query_type == "page_specific":
        template = templates.PAGE_SPECIFIC_TEMPLATE
        formatted_prompt = template.format(
            context=combined_context,
            conversation_context=conversation_context,
            question=query
        )
    elif query_type == "summary":
        template = templates.DOCUMENT_SUMMARY_TEMPLATE
        formatted_prompt = template.format(
            context=combined_context,
            conversation_context=conversation_context,
            question=query
        )
    elif query_type == "table":
        template = templates.TABLE_ANALYSIS_TEMPLATE
        formatted_prompt = template.format(
            table_content=combined_context,
            conversation_context=conversation_context,
            question=query
        )
    elif query_type == "chart":
        template = templates.CHART_ANALYSIS_TEMPLATE
        formatted_prompt = template.format(
            chart_content=combined_context,
            conversation_context=conversation_context,
            question=query
        )
    else:
        template = templates.ADVANCED_CONTEXT_QA_TEMPLATE
        formatted_prompt = template.format(
            context=combined_context,
            conversation_context=conversation_context,
            question=query
        )
    
    # Generate response with error handling
    streaming_callback.reset()
    try:
        print("🤖 Analyzing documents and generating response...")
        response = llm.invoke(formatted_prompt)
        final_answer = streaming_callback.text if streaming_callback.text else response
        
        processing_time = time.time() - start_time
        confidence = min(1.0, len(prioritized_results) / 10.0)
        
        context_manager.add_exchange(
            query, final_answer, prioritized_results, 
            conversation_context, query_type, confidence
        )
        
        return final_answer, prioritized_results, query_type
        
    except Exception as e:
        error_msg = f"Error generating response: {e}. Please try rephrasing your question."
        print(f"\n❌ LLM Error: {e}")
        context_manager.add_exchange(query, error_msg, prioritized_results, "", query_type, 0.0)
        return error_msg, prioritized_results, query_type

# ---------- MAIN INTERFACE ----------

def main():
    """Enhanced main application with comprehensive features"""
    ensure_directory(str(config.EMBEDDINGS_DIR))
    
    # Initialize system
    try:
        if not initialize_rag_system():
            print("❌ Failed to initialize RAG system")
            return
    except Exception as e:
        print(f"❌ RAG system initialization failed: {e}")
        return
    
    # Initialize components
    doc_manager = AdvancedDocumentManager(str(config.EMBEDDINGS_DIR))
    context_manager = EnhancedConversationContext()
    streaming_callback = EnhancedStreamingCallback()
    
    print("=" * 70)
    print("🚀 ENHANCED RAG QUERY SYSTEM")
    print("    Advanced Document Q&A with Context Management")
    print("=" * 70)
    
    # Display system capabilities
    print("\n🔧 System Capabilities:")
    status = config.get_capability_status()
    for capability, status_text in status.items():
        print(f"   📊 {capability.title()}: {status_text}")
    
    print("\n📋 Available Operations:")
    print("1. 📚 Load all documents")
    print("2. 📖 Load specific document")
    print("3. 🔄 Refresh document statistics")
    
    choice = input("\n➤ Select option (1-3): ").strip()
    
    if choice == "1":
        print("\n📚 Loading all available documents...")
        if not doc_manager.load_all_documents():
            print("❌ No documents could be loaded!")
            return
    
    elif choice == "2":
        folders = [f for f in os.listdir(str(config.EMBEDDINGS_DIR)) 
                  if os.path.isdir(os.path.join(str(config.EMBEDDINGS_DIR), f))]
        if not folders:
            print("❌ No embeddings found!")
            return
        
        print("\n📖 Available documents:")
        for i, folder in enumerate(folders, 1):
            metadata_file = os.path.join(str(config.EMBEDDINGS_DIR), folder, "metadata.txt")
            info = ""
            if os.path.exists(metadata_file):
                try:
                    with open(metadata_file, 'r', encoding='utf-8') as f:
                        lines = f.read()
                        if 'Author:' in lines:
                            author = lines.split('Author: ')[1].split('\n')[0]
                            info += f" (Author: {author})"
                except:
                    pass
            
            print(f"{i:2d}. {folder}{info}")
        
        try:
            doc_idx = int(input(f"\n➤ Select document number (1-{len(folders)}): ")) - 1
            if 0 <= doc_idx < len(folders):
                selected_folder = folders[doc_idx]
                folder_path = os.path.join(str(config.EMBEDDINGS_DIR), selected_folder)
                
                print(f"📖 Loading {selected_folder}...")
                db, metadata, stats = doc_manager._load_single_document_enhanced(folder_path)
                
                if db:
                    doc_manager.loaded_documents[selected_folder] = {
                        'db': db, 
                        'metadata': metadata, 
                        'folder_path': folder_path,
                        'statistics': stats
                    }
                    print(f"✅ Successfully loaded {selected_folder}")
                    print(f"   📊 {stats['chunks']} chunks, {stats['images']} images, {stats['tables']} tables")
                else:
                    print("❌ Failed to load document!")
                    return
            else:
                print("❌ Invalid selection!")
                return
        except ValueError:
            print("❌ Invalid input!")
            return
    
    elif choice == "3":
        print("\n🔄 Refreshing document statistics...")
        if doc_manager.loaded_documents:
            doc_manager._print_loading_summary()
        else:
            print("📊 No documents currently loaded.")
        input("\nPress Enter to continue...")
        main()
        return
    
    else:
        print("❌ Invalid choice!")
        return
    
    # Interactive query loop
    print("\n" + "=" * 70)
    print("🤖 ADVANCED RAG SYSTEM READY")
    print("=" * 70)
    print("💡 Available Commands:")
    print("   • Ask questions about your documents")
    print("   • 'context' - Show conversation history")
    print("   • 'clear' - Clear conversation history")
    print("   • 'docs' - List loaded documents")
    print("   • 'stats' - Show detailed system statistics")
    print("   • 'quit' - Exit system")
    print("=" * 70)
    
    query_count = 0
    
    while True:
        try:
            query = input(f"\n[{query_count + 1}] ➤ Your question: ").strip()
            
            if query.lower() in ['quit', 'exit', 'q']:
                print("👋 Thank you for using the Enhanced RAG System!")
                break
                
            elif query.lower() == 'context':
                if context_manager.conversation_history:
                    print(f"\n💬 Conversation History ({len(context_manager.conversation_history)} exchanges):")
                    for i, exchange in enumerate(context_manager.conversation_history, 1):
                        print(f"\n{i}. [{exchange.query_type.upper()}] Q: {exchange.question}")
                        preview = exchange.answer[:200] + "..." if len(exchange.answer) > 200 else exchange.answer
                        print(f"   A: {preview}")
                        if exchange.confidence > 0:
                            print(f"   🎯 Confidence: {exchange.confidence:.2f}")
                else:
                    print("📝 No conversation history yet")
                continue
                
            elif query.lower() == 'clear':
                context_manager.clear_history()
                doc_manager.search_cache.clear()
                print("🧹 Conversation history and search cache cleared")
                query_count = 0
                continue
                
            elif query.lower() == 'docs':
                if doc_manager.loaded_documents:
                    print(f"\n📚 Loaded Documents ({len(doc_manager.loaded_documents)}):")
                    for name, data in doc_manager.loaded_documents.items():
                        metadata = data['metadata']
                        stats = data['statistics']
                        print(f"\n📄 {name}")
                        print(f"   📝 Title: {metadata.get('title', 'Unknown')}")
                        print(f"   👤 Author: {metadata.get('author', 'Unknown')}")
                        print(f"   🧩 Chunks: {stats.get('chunks', 0)}")
                        print(f"   🖼️ Images: {stats.get('images', 0)}")
                        print(f"   📊 Tables: {stats.get('tables', 0)}")
                else:
                    print("📚 No documents currently loaded")
                continue
                
            elif query.lower() == 'stats':
                if doc_manager.loaded_documents:
                    total_chunks = sum(data['statistics'].get('chunks', 0) for data in doc_manager.loaded_documents.values())
                    total_images = sum(data['statistics'].get('images', 0) for data in doc_manager.loaded_documents.values())
                    total_tables = sum(data['statistics'].get('tables', 0) for data in doc_manager.loaded_documents.values())
                    
                    print(f"\n📊 SYSTEM STATISTICS")
                    print(f"━━━━━━━━━━━━━━━━━━━")
                    print(f"📚 Documents loaded: {len(doc_manager.loaded_documents)}")
                    print(f"🧩 Total chunks: {total_chunks}")
                    print(f"🖼️ Total images: {total_images}")
                    print(f"📊 Total tables: {total_tables}")
                    print(f"🔍 Search queries cached: {len(doc_manager.search_cache)}")
                    print(f"💬 Conversation exchanges: {len(context_manager.conversation_history)}")
                else:
                    print("📊 No documents loaded for statistics")
                continue
            
            if not query:
                continue
            
            # Process the query
            print("🔍 Analyzing your question...")
            start_time = time.time()
            
            answer, sources, query_type = process_query_with_advanced_context(
                query, doc_manager, context_manager, streaming_callback
            )
            
            processing_time = time.time() - start_time
            query_count += 1
            
            # Enhanced source summary
            if sources:
                source_docs = set()
                source_types = defaultdict(int)
                pages_referenced = set()
                
                for source in sources[:8]:
                    if hasattr(source, 'metadata'):
                        source_docs.add(source.metadata.get('source_document', 'Unknown'))
                        source_type = source.metadata.get('type', 'text')
                        source_types[source_type] += 1
                        pages_referenced.add(source.metadata.get('display_page', 'Unknown'))
                
                print(f"\n📋 Sources Used:")
                print(f"   📖 Documents: {len(source_docs)} ({', '.join(list(source_docs)[:3])}{'...' if len(source_docs) > 3 else ''})")
                # Filter out 'Unknown' strings and sort only numeric pages
                numeric_pages = [p for p in pages_referenced if isinstance(p, int)]
                unknown_pages = [p for p in pages_referenced if isinstance(p, str)]
                sorted_pages = sorted(numeric_pages) + unknown_pages
                print(f"   📄 Pages: {', '.join(map(str, sorted_pages[:8]))}")
                print(f"   📊 Content types: {', '.join([f'{count} {type_name}' for type_name, count in source_types.items()])}")
                print(f"   ⚡ Query type: {query_type.upper()}")
                print(f"   ⏱️ Processing time: {processing_time:.1f}s")
        
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ An error occurred: {e}")
            print("Please try rephrasing your question or check the system status.")
            continue

if __name__ == "__main__":
    main()
