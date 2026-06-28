# modules/utils.py
"""
Utility functions module.
Contains helper functions used across the application.
"""

import difflib


def find_mentioned_document(message: str, candidates: list) -> str:
    """Return the candidate document name best matching the message"""
    if not message or not candidates:
        return ''
    
    msg = message.lower().strip()
    
    # Exact or substring match first
    for name in candidates:
        n = name.lower()
        if n in msg or msg in n:
            return name
    
    # Fuzzy match using difflib
    best = ''
    best_ratio = 0.0
    for name in candidates:
        ratio = difflib.SequenceMatcher(None, msg, name.lower()).ratio()
        if ratio > best_ratio:
            best_ratio = ratio
            best = name
    
    return best if best_ratio >= 0.6 else ''


def format_loaded_documents_response(doc_manager, selected: list = None) -> str:
    """Create a concise summary of loaded document names"""
    if not doc_manager or not doc_manager.loaded_documents:
        return "No documents are currently loaded."
    
    all_names = sorted(list(doc_manager.loaded_documents.keys()))
    names = all_names
    
    if selected:
        sel_set = set(selected)
        names = sorted([n for n in all_names if n in sel_set])
        if not names:
            names = all_names
    
    return "Loaded documents: " + ", ".join(names)


def format_document_info(doc_name: str, doc_manager) -> str:
    """Format detailed information about a document"""
    if not doc_manager or doc_name not in doc_manager.loaded_documents:
        return f"Document '{doc_name}' not found"
    
    info = doc_manager.loaded_documents[doc_name]
    meta = info.get('metadata', {})
    stats = info.get('statistics', {})
    
    title = meta.get('title', doc_name)
    author = meta.get('author', 'Unknown')
    pages = meta.get('total_pages', 0)
    chunks = stats.get('chunks', 0)
    images = stats.get('images', 0)
    tables = stats.get('tables', 0)
    
    return f"- {doc_name} (Title: {title}, Author: {author}, Pages: {pages}, Chunks: {chunks}, Images: {images}, Tables: {tables})"


def get_document_statistics(doc_manager):
    """Get statistics for all loaded documents"""
    if not doc_manager or not doc_manager.loaded_documents:
        return {
            'total_documents': 0,
            'total_chunks': 0,
            'total_images': 0,
            'total_tables': 0,
            'documents': []
        }
    
    total_chunks = sum(data['statistics'].get('chunks', 0) 
                      for data in doc_manager.loaded_documents.values())
    total_images = sum(data['statistics'].get('images', 0) 
                      for data in doc_manager.loaded_documents.values())
    total_tables = sum(data['statistics'].get('tables', 0) 
                      for data in doc_manager.loaded_documents.values())
    
    documents = [
        {
            'name': name,
            'title': data['metadata'].get('title', 'Unknown'),
            'author': data['metadata'].get('author', 'Unknown'),
            'chunks': data['statistics'].get('chunks', 0),
            'images': data['statistics'].get('images', 0),
            'tables': data['statistics'].get('tables', 0)
        }
        for name, data in doc_manager.loaded_documents.items()
    ]
    
    return {
        'total_documents': len(doc_manager.loaded_documents),
        'total_chunks': total_chunks,
        'total_images': total_images,
        'total_tables': total_tables,
        'documents': documents
    }


def is_query_about_documents(message: str) -> bool:
    """Check if message is asking about listing/counting documents"""
    lowered = message.lower().strip()
    
    doc_list_triggers = [
        'list docs', 'list of docs', 'list documents', 'list of documents',
        'which documents are loaded', 'which docs are loaded',
        'show loaded docs', 'show loaded documents', 'loaded docs', 'loaded documents',
        'all docs', 'all documents', 'show all docs', 'show all documents',
        'list out all docs', 'list out all documents', 'list out the docs', 
        'list out the documents', 'the list of docs', 'the list of documents',
        'list files', 'list of files', 'show loaded files', 'loaded files',
        'all files', 'show all files', 'list out all files'
    ]
    
    count_triggers = [
        'how many docs', 'how many documents', 'number of docs', 'number of documents',
        'how many are loaded', 'docs count', 'documents count', 'count docs',
        'count documents', 'how many files', 'files count', 'number of files', 'count files'
    ]
    
    list_heuristic = ('list' in lowered and ('doc' in lowered or 'document' in lowered or 'file' in lowered))
    
    return (any(trigger in lowered for trigger in doc_list_triggers) or 
            any(trigger in lowered for trigger in count_triggers) or 
            list_heuristic)


def is_referential_query(message: str) -> bool:
    """Check if message is a referential follow-up query"""
    lowered = message.lower().strip()
    
    referential_triggers = [
        'what are they', 'whar are they', 'wht are they', 'wat are they',
        'what are these', 'what are those', 'tell me about them', 'tell about them',
        'what are the selected documents', 'what are the selected docs',
        'give details about them', 'describe them', 'who are they', 
        'explain them', 'details about them', 'another document', 'next document',
        'the other document', 'the other one', 'another one'
    ]
    
    pronouns = [' they ', ' these ', ' those ', ' them ']
    wh_words = ['what', 'who', 'describe', 'detail', 'details', 'explain', 'tell']
    tokens = f" {lowered} "
    
    pronoun_present = any(p in tokens for p in pronouns)
    wh_present = any(w in lowered for w in wh_words)
    short_len = len(lowered.split()) <= 6
    
    pronoun_followup_heuristic = ((pronoun_present and (wh_present or ' are ' in tokens or '?' in lowered)) and short_len)
    
    return any(trigger in lowered for trigger in referential_triggers) or pronoun_followup_heuristic