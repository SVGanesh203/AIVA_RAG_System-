# modules/conversation.py
"""
Conversation context and memory management.
Handles conversation history, user memory, and context building.
"""

import time
import re
import random


class ConversationExchange:
    """Single conversation exchange"""
    def __init__(self, question, answer, timestamp, sources=None, context_used="", 
                 query_type="general", confidence=0.0):
        self.question = question
        self.answer = answer
        self.timestamp = timestamp
        self.sources = sources or []
        self.context_used = context_used
        self.query_type = query_type
        self.confidence = confidence


class EnhancedConversationContext:
    """Advanced conversation context with document tracking and entity memory"""
    def __init__(self, max_history=10):
        self.conversation_history = []
        self.max_history = max_history
        self.user_memory = {
            'bot_name': 'Catapult',
            'facts': {},
            'notes': []
        }
        self.document_context = {}
        self.entity_memory = {}
        self.query_patterns = []
        self.last_document_used = None

    def add_exchange(self, question, answer, sources=None, context_used="", 
                    query_type="general", confidence=0.0):
        """Add conversation exchange to history"""
        sources_info = []
        if sources:
            for source in sources[:6]:
                if hasattr(source, 'metadata'):
                    source_info = {
                        "type": source.metadata.get('type', 'text'),
                        "page": source.metadata.get('display_page', 0),
                        "source_doc": source.metadata.get('source_document', 'unknown'),
                        "has_table": source.metadata.get('has_table', False),
                        "image_type": source.metadata.get('image_type', None)
                    }
                    sources_info.append(source_info)
                    
                    doc_name = source.metadata.get('source_document', 'unknown')
                    if doc_name not in self.document_context:
                        self.document_context[doc_name] = {
                            'last_used': time.time(),
                            'usage_count': 0,
                            'entities': set(),
                            'topics': set()
                        }
                    self.document_context[doc_name]['last_used'] = time.time()
                    self.document_context[doc_name]['usage_count'] += 1
                    self.last_document_used = doc_name

        if query_type in ['general', 'document_query'] and answer:
            self.extract_and_store_facts(question, answer)

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
        
        self.query_patterns.append({
            'query': question,
            'type': query_type,
            'timestamp': time.time(),
            'sources_count': len(sources_info)
        })
        
        if len(self.conversation_history) > self.max_history:
            self.conversation_history.pop(0)
            
        if len(self.query_patterns) > 50:
            self.query_patterns.pop(0)

    def get_enhanced_context_for_query(self, query, force: bool = False):
        """Return conversation context with intelligent follow-up detection"""
        try:
            if not force:
                if len(self.conversation_history) >= 1:
                    pass
        except Exception:
            pass

        if not self.conversation_history:
            if self.user_memory:
                mem_lines = ["=== MEMORY ==="]
                for k, v in self.user_memory.items():
                    mem_lines.append(f"{k}: {v}")
                mem_lines.append("=== END MEMORY ===")
                return "\n".join(mem_lines)
            return ""

        follow_up_patterns = [
            r'\b(?:explain\s+more|tell\s+me\s+more|elaborate|expand|go\s+deeper|more\s+details?)\b',
            r'\b(?:what\s+about|how\s+about|can\s+you\s+explain|can\s+you\s+tell)\b',
            r'\b(?:this|that|it|they|them|these|those)\b',
            r'\b(?:previous|earlier|above|before|mentioned|discussed|said)\b',
            r'\b(?:same|similar|related|also|additionally|furthermore)\b'
        ]
        
        is_follow_up = any(re.search(pattern, query.lower()) for pattern in follow_up_patterns)

        context_parts = []
        if self.user_memory:
            context_parts.append("=== MEMORY ===")
            for k, v in self.user_memory.items():
                context_parts.append(f"{k}: {v}")
            context_parts.append("=== END MEMORY ===")

        context_parts.append("=== CONVERSATION CONTEXT ===")

        for i, exchange in enumerate(self.conversation_history[-3:]):
            context_parts.append(f"Previous question: {exchange.question}")
            
            if is_follow_up and i == len(self.conversation_history[-3:]) - 1:
                answer_text = exchange.answer
                if len(answer_text) > 1500:
                    answer_text = answer_text[:1500] + "..."
                context_parts.append(f"Previous answer: {answer_text}")
            elif exchange.query_type in ['system_info', 'memory_lookup']:
                context_parts.append(f"Previous answer: {exchange.answer[:100]}...")
            elif force:
                answer_text = exchange.answer
                if len(answer_text) > 200:
                    answer_text = answer_text[:200] + "..."
                context_parts.append(f"Previous answer: {answer_text}")

        if is_follow_up:
            context_parts.append("IMPORTANT: This is a follow-up question. Please build upon and expand the previous answer with additional details, examples, or explanations.")

        context_parts.append("=== END CONTEXT ===")
        return "\n".join(context_parts)

    def set_memory(self, key: str, value: str):
        """Set memory value"""
        self.user_memory[key] = value

    def get_memory(self, key: str, default=None):
        """Get memory value"""
        return self.user_memory.get(key, default)

    def add_fact(self, key: str, value: str):
        """Add fact to memory"""
        key_norm = key.strip().lower()
        if key_norm:
            self.user_memory['facts'][key_norm] = value.strip()

    def get_fact(self, key: str):
        """Get fact from memory"""
        return self.user_memory['facts'].get(key.strip().lower())
    
    def extract_and_store_facts(self, question: str, answer: str):
        """Automatically extract and store facts about people and roles from answers"""
        patterns = [
            (r'([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s+(?:is|:)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)', 'person_role'),
            (r'([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s*(?:is|:)\s*([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)', 'role_person'),
            (r'([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s*[\(-]\s*([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)', 'person_role'),
            (r'(?:the\s+)?([a-z]+(?:\s+[a-z]+)*)\'?s?\s+name\s+is\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)', 'role_person'),
            (r'([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s+is\s+(?:identified\s+as|mentioned\s+as)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)', 'person_role'),
        ]
        
        for pattern, pattern_type in patterns:
            matches = re.findall(pattern, answer, re.IGNORECASE)
            for match in matches:
                if pattern_type == 'person_role':
                    person, role = match
                    self.add_fact(f"who is {person.lower()}", role)
                    self.add_fact(f"{person.lower()} role", role)
                    self.add_fact(f"{role.lower()}", person)
                elif pattern_type == 'role_person':
                    role, person = match
                    self.add_fact(f"who is {person.lower()}", role)
                    self.add_fact(f"{person.lower()} role", role)
                    self.add_fact(f"{role.lower()}", person)
    
    def check_facts_before_search(self, query: str) -> str:
        """Check if we have stored facts that can answer the query"""
        query_lower = query.lower().strip()
        
        fact = self.get_fact(query_lower)
        if fact:
            return fact
        
        variations = [
            query_lower,
            query_lower.replace('?', '').strip(),
            query_lower.replace('who is', '').strip(),
            query_lower.replace('what is', '').strip(),
            query_lower.replace('tell me about', '').strip(),
            query_lower.replace('the ', '').strip(),
        ]
        
        if 'manager' in query_lower:
            fact = self.get_fact('manager')
            if fact:
                return f"The manager is {fact}"
        
        if 'svp' in query_lower or 'senior vice president' in query_lower:
            fact = self.get_fact('svp')
            if fact:
                return f"The SVP is {fact}"
        
        for variation in variations:
            if variation:
                fact = self.get_fact(variation)
                if fact:
                    return fact
        
        return None

    def clear_history(self):
        """Clear all conversation data"""
        self.conversation_history = []
        self.document_context = {}
        self.entity_memory = {}
        self.query_patterns = []
        self.last_document_used = None
        
    def get_document_context(self):
        """Get current document context for intelligent search"""
        return {
            'last_document': self.last_document_used,
            'document_usage': self.document_context,
            'recent_patterns': self.query_patterns[-3:] if self.query_patterns else []
        }
        
    def should_search_last_document_first(self, query: str) -> bool:
        """Determine if we should search the last document first"""
        if not self.last_document_used:
            return False
        
        follow_up_indicators = ['it', 'this', 'that', 'the above', 'previous', 'mentioned', 'same']
        query_lower = query.lower()
        
        if any(indicator in query_lower for indicator in follow_up_indicators):
            return True
            
        if len(query.split()) <= 3 and self.query_patterns:
            return True
        
        return False


def detect_casual_conversation(message: str, selected_documents: list = None) -> str:
    """Detect casual conversation patterns and respond appropriately"""
    if not message or len(message.strip()) == 0:
        return None
    
    if is_document_query(message):
        return None
    
    message_lower = message.lower().strip()
    message_words = message_lower.split()
    
    casual_patterns = {
        'greeting': ['hi', 'hello', 'hey', 'hiya', 'howdy'],
        'how_are_you': ['how are you', 'how are u', 'how r u', 'how\'re you'],
        'thank_you': ['thank you', 'thanks', 'thx', 'thank u'],
        'goodbye': ['bye', 'goodbye', 'see you', 'see ya', 'later', 'take care'],
        'who_are_you': ['who are you', 'what are you', 'what can you do', 'what do you do'],
        'help': ['help', 'can you help', 'i need help', 'how can you help']
    }
    
    for pattern_type, patterns in casual_patterns.items():
        for pattern in patterns:
            if message_lower == pattern:
                return generate_casual_response(pattern_type, message_lower, selected_documents)
    
    if len(message_words) == 1 and message_lower in ['hi', 'hello', 'hey', 'ok', 'yes', 'no', 'okay']:
        return generate_casual_response('greeting', message_lower, selected_documents)
    
    return None


def generate_casual_response(pattern_type: str, message: str, selected_documents: list = None) -> str:
    """Generate appropriate casual responses"""
    # This would need access to doc_manager, so we'll need to pass it or make it global
    # For now, simplified version
    responses = {
        'greeting': [
            "Hello! 👋 I'm your AI document assistant. I can help you search through and analyze your documents.",
            "Hi there! I'm ready to help you explore your documents. What would you like to know?",
            "Hello! I'm here to assist you with document analysis and Q&A."
        ],
        'how_are_you': [
            "I'm doing great, thank you for asking! I'm ready to help you with your documents. What would you like to search for?",
            "I'm doing well! I'm here and ready to assist you with document queries. How can I help?",
            "I'm excellent, thanks! I'm ready to analyze your documents. What information are you looking for?"
        ],
        'thank_you': [
            "You're very welcome! I'm happy to help. Is there anything else you'd like to know about your documents?",
            "My pleasure! I'm here whenever you need assistance with your documents.",
            "You're welcome! Feel free to ask me anything about your documents anytime."
        ],
        'goodbye': [
            "Goodbye! Feel free to come back anytime if you need help with your documents. 👋",
            "See you later! I'll be here whenever you need document assistance.",
            "Take care! Don't hesitate to return if you have more questions about your documents."
        ],
        'who_are_you': [
            "I'm an AI assistant specialized in document analysis and Q&A. I can help you search through your documents, answer questions, and extract information. What would you like to know about your documents?",
            "I'm your intelligent document assistant! I can analyze PDFs, search through content, and answer questions about your documents. How can I help you today?",
            "I'm an AI-powered document analysis tool. I can help you find information, answer questions, and explore your documents efficiently. What would you like to search for?"
        ],
        'help': [
            "I'm here to help! I can search through your documents, answer questions, and extract information. Try asking me about specific content in your documents, or use the search feature to find relevant documents first.",
            "I can assist you with document analysis! You can ask me questions about your loaded documents, search for specific information, or explore content. What would you like to know?",
            "I'm your document analysis assistant! I can help you find information, answer questions, and explore your documents. Start by loading some documents and then ask me anything about them!"
        ]
    }
    
    pattern_responses = responses.get(pattern_type, responses['greeting'])
    return random.choice(pattern_responses)


def is_document_query(message: str) -> bool:
    """Determine if a message is asking about documents/content"""
    if not message or len(message.strip()) == 0:
        return False
    
    message_lower = message.lower().strip()
    
    document_keywords = [
        'what', 'how', 'when', 'where', 'who', 'why', 'explain', 'describe', 'tell me',
        'find', 'search', 'look for', 'show me', 'list', 'get', 'retrieve',
        'document', 'file', 'pdf', 'content', 'information', 'data', 'details',
        'table', 'chart', 'graph', 'image', 'figure', 'diagram',
        'page', 'section', 'chapter', 'part', 'section',
        'name', 'title', 'author', 'date', 'number', 'value', 'amount',
        'process', 'procedure', 'method', 'step', 'workflow'
    ]
    
    has_document_keywords = any(keyword in message_lower for keyword in document_keywords)
    
    question_words = ['what', 'how', 'when', 'where', 'who', 'why', 'which', 'can', 'could', 'would', 'should']
    is_question = any(message_lower.startswith(word) for word in question_words)
    
    is_substantial = len(message_lower.split()) > 3
    
    return has_document_keywords or (is_question and is_substantial)