# modules/response_generator.py
"""
Response generation and streaming module.
Handles prompt generation and response streaming.
"""

import json
import time
import re


class EnhancedStreamingCallback:
    """Simple streaming callback for Flask"""
    def __init__(self):
        self.text = ""
        self.is_streaming = False

    def reset(self):
        self.text = ""
        self.is_streaming = False


def generate_prompt_based_on_mode(search_mode, bot_name, user_message, combined_context, 
                                  conversation_context, followup_force_context):
    """Generate prompt based on search mode"""
    
    if search_mode == 'general_plus_docs':
        if followup_force_context:
            return f"""You are {bot_name}, an AI assistant with access to both general knowledge and specific documents.

CONVERSATION CONTEXT:
{conversation_context}

DOCUMENT CONTENT (if available):
{combined_context if combined_context else "No relevant document content found."}

QUESTION: {user_message}

INSTRUCTIONS:
- This is a follow-up question. Build upon and expand the previous answer.
- First, check if the document content provides relevant information.
- If documents have relevant info, use it and cite the sources.
- If documents don't have sufficient info, use your general knowledge to provide a comprehensive answer.
- You can combine document information with general knowledge for a complete response.

ANSWER:"""
        else:
            return f"""You are {bot_name}, an AI assistant with access to both general knowledge and specific documents.

CONVERSATION CONTEXT:
{conversation_context}

DOCUMENT CONTENT (if available):
{combined_context if combined_context else "No relevant document content found."}

QUESTION: {user_message}

INSTRUCTIONS:
- First, check if the document content provides relevant information.
- If documents have relevant info, prioritize it and cite the sources.
- If documents don't have sufficient info or no documents are available, use your general knowledge to provide a comprehensive answer.
- For general knowledge questions (like "what are prime numbers"), feel free to answer using your training data.
- You can combine document information with general knowledge for a complete response.

ANSWER:"""
    
    else:  # documents_only mode (default)
        if followup_force_context:
            return f"""You are {bot_name}. This is a follow-up question that builds upon previous conversation.

CONVERSATION CONTEXT:
{conversation_context}

DOCUMENT CONTENT:
{combined_context}

QUESTION: {user_message}

IMPORTANT: 
- This is a follow-up question. Please build upon and expand the previous answer with additional details, examples, or explanations.
- Answer ONLY based on the information provided in the document content above.
- If the information is not in the documents, clearly state: "This information is not available in the loaded documents."

ANSWER:"""
        else:
            return f"""You are {bot_name}. Based on the following document content, answer the question comprehensively.

CONVERSATION CONTEXT:
{conversation_context}

DOCUMENT CONTENT:
{combined_context}

QUESTION: {user_message}

IMPORTANT:
- Provide a detailed answer based ONLY on the document content above.
- Include specific information, names, numbers, and details as available in the documents.
- If the information is not in the documents, clearly state: "This information is not available in the loaded documents."
- Do NOT use external knowledge or make assumptions beyond what's in the documents.

ANSWER:"""


def format_sources_for_response(search_results, max_sources=6):
    """Format search results into source information"""
    formatted_sources = []
    for source in search_results[:max_sources]:
        if hasattr(source, 'metadata'):
            formatted_sources.append({
                'document': source.metadata.get('source_document', 'Unknown'),
                'title': source.metadata.get('document_title', 'Unknown'),
                'page': source.metadata.get('display_page', 'Unknown'),
                'type': source.metadata.get('type', 'text'),
                'has_table': source.metadata.get('has_table', False),
                'image_type': source.metadata.get('image_type', None),
                'relevance_score': source.metadata.get('relevance_score', 0.0),
                'keywords': source.metadata.get('contextual_keywords', [])[:3],
                'entities': source.metadata.get('named_entities', [])[:3],
                'citations': source.metadata.get('citations', [])[:2]
            })
    return formatted_sources


def build_context_from_results(search_results, max_results=8, is_fast=False):
    """Build context string from search results"""
    context_parts = []
    for doc in search_results[:max_results]:
        doc_name = doc.metadata.get('source_document', 'Unknown')
        page_num = doc.metadata.get('display_page', 'Unknown')
        doc_type = doc.metadata.get('type', 'text')
        snippet = doc.page_content
        
        if is_fast and len(snippet) > 2000 // 4:
            snippet = snippet[:2000 // 4]
        elif (not is_fast) and len(snippet) > 8000 // 6:
            snippet = snippet[:8000 // 6]
        
        context_parts.append(f"[Source: {doc_name} | Page: {page_num} | Type: {doc_type}]\n{snippet}")
    
    return "\n\n".join(context_parts)


def generate_stream_response(llm, prompt, search_results, start_time, max_sources=6):
    """Generate streaming response tokens"""
    try:
        # Stream tokens from model
        for chunk in llm.stream(prompt):
            yield f"data: {json.dumps({'type': 'token', 'content': str(chunk)})}\n\n"
        
        # Format and send sources
        formatted_sources = format_sources_for_response(search_results, max_sources)
        total_time = round(time.time() - start_time, 2)
        yield f"data: {json.dumps({'type': 'done', 'sources': formatted_sources, 'processing_time': total_time})}\n\n"
        
    except Exception as e:
        yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"


def generate_casual_stream_response(response_text, start_time):
    """Generate streaming response for casual conversation"""
    try:
        yield ": ping\n\n"
        yield f"data: {json.dumps({'type': 'status', 'message': 'responding'})}\n\n"
        
        words = response_text.split()
        for i, word in enumerate(words):
            yield f"data: {json.dumps({'type': 'token', 'content': word + (' ' if i < len(words) - 1 else '')})}\n\n"
        
        yield f"data: {json.dumps({'type': 'done', 'sources': [], 'processing_time': round(time.time() - start_time, 2)})}\n\n"
        
    except Exception as e:
        yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"


def detect_follow_up_query(query: str) -> bool:
    """Detect if query is a follow-up question"""
    follow_up_patterns = [
        r'\b(?:explain\s+more|tell\s+me\s+more|elaborate|expand|go\s+deeper|more\s+details?)\b',
        r'\b(?:what\s+about|how\s+about|can\s+you\s+explain|can\s+you\s+tell)\b',
        r'\b(?:this|that|it|they|them|these|those)\b',
        r'\b(?:previous|earlier|above|before|mentioned|discussed|said)\b',
        r'\b(?:same|similar|related|also|additionally|furthermore)\b'
    ]
    
    query_lower = query.lower()
    return any(re.search(pattern, query_lower) for pattern in follow_up_patterns)