
# modules/search.py
"""
Search logic and query processing module.
Handles intelligent document search, query analysis, and relevance scoring.
"""

import hashlib
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Tuple

# Threshold for "high confidence" matches (>= 70%)
HIGH_CONFIDENCE_THRESHOLD: float = 0.7


def intelligent_document_search(query: str, doc_manager, k: int = 10, fast: bool = False,
                               selected_documents: list = None, context_manager=None) -> tuple:
    """Intelligent document search with advanced metadata and context awareness"""
    if not doc_manager or not doc_manager.loaded_documents:
        return [], {}

    documents_to_search = doc_manager.loaded_documents
    if selected_documents:
        documents_to_search = {
            name: data for name, data in doc_manager.loaded_documents.items()
            if name in selected_documents
        }
        if not documents_to_search:
            print(f"⚠️ No selected documents found in loaded documents. Available: {list(doc_manager.loaded_documents.keys())}")
            return [], {}

    all_results = []
    query_lower = query.lower()

    query_analysis = analyze_query_intent(query)

    search_last_first = False
    if context_manager:
        search_last_first = context_manager.should_search_last_document_first(query)
    document_order = []

    if search_last_first and context_manager and context_manager.last_document_used:
        last_doc = context_manager.last_document_used
        if last_doc in documents_to_search:
            document_order.append((last_doc, documents_to_search[last_doc]))

        for doc_name, doc_data in documents_to_search.items():
            if doc_name != last_doc:
                document_order.append((doc_name, doc_data))
    else:
        document_order = list(documents_to_search.items())

        def calculate_doc_relevance(doc_item):
            doc_name, doc_data = doc_item
            doc_title = doc_data.get('metadata', {}).get('title', '').lower()
            doc_author = doc_data.get('metadata', {}).get('author', '').lower()

            relevance_score = 0
            query_words = set(query_lower.split())

            title_words = set(doc_title.split())
            title_overlap = len(query_words & title_words)
            relevance_score += title_overlap * 2

            author_words = set(doc_author.split())
            author_overlap = len(query_words & author_words)
            relevance_score += author_overlap

            doc_name_words = set(doc_name.lower().replace('_', ' ').split())
            name_overlap = len(query_words & doc_name_words)
            relevance_score += name_overlap * 1.5

            return relevance_score

        document_order.sort(key=calculate_doc_relevance, reverse=True)

    max_workers = min(4, len(document_order))

    def search_single_document(doc_item):
        doc_name, doc_data = doc_item
        try:
            if query_analysis['is_specific_search']:
                search_queries = [query, query_analysis['enhanced_query']]
            else:
                search_queries = [query]

            doc_results = []
            for search_query in search_queries:
                try:
                    # We keep k//2 per query to balance multi-query searches.
                    docs = doc_data['db'].similarity_search(search_query, k=k // 2)
                    print(f"✅ Search successful for {doc_name}: found {len(docs)} results")
                except Exception as search_error:
                    print(f"❌ Search error in {doc_name}: {search_error}")
                    continue

                for doc in docs:
                    try:
                        relevance_score = calculate_relevance_score(doc, query, query_analysis)
                        if relevance_score >= 0.1:
                            doc.metadata['relevance_score'] = relevance_score
                            doc.metadata['high_confidence_match'] = relevance_score >= HIGH_CONFIDENCE_THRESHOLD
                            doc.metadata['source_document'] = doc_name
                            doc.metadata['document_title'] = doc_data['metadata']['title']
                            doc.metadata['document_author'] = doc_data['metadata']['author']
                            doc_results.append(doc)
                    except Exception as score_error:
                        print(f"Scoring error for doc in {doc_name}: {score_error}")
                        doc.metadata['relevance_score'] = 0.5
                        doc.metadata['high_confidence_match'] = False
                        doc.metadata['source_document'] = doc_name
                        doc.metadata['document_title'] = doc_data['metadata']['title']
                        doc.metadata['document_author'] = doc_data['metadata']['author']
                        doc_results.append(doc)

            return doc_results

        except Exception as e:
            print(f"Search error in {doc_name}: {str(e)}")
            import traceback
            print(f"Full error traceback: {traceback.format_exc()}")
            return []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_doc = {
            executor.submit(search_single_document, doc_item): doc_item[0]
            for doc_item in document_order
        }

        for future in as_completed(future_to_doc):
            doc_name = future_to_doc[future]
            try:
                doc_results = future.result()
                all_results.extend(doc_results)
            except Exception as e:
                print(f"Error processing {doc_name}: {e}")

    unique_results = remove_duplicate_results(all_results)
    sorted_results = sorted(unique_results, key=lambda x: x.metadata.get('relevance_score', 0), reverse=True)

    return sorted_results[:k], documents_to_search


def analyze_query_intent(query: str) -> dict:
    """Analyze query to determine search strategy"""
    query_lower = query.lower()

    casual_indicators = ['hi', 'hello', 'hey', 'how are you', 'thank you', 'bye', 'goodbye']
    if any(indicator in query_lower for indicator in casual_indicators):
        return {
            'type': 'casual_conversation',
            'is_specific_search': False,
            'is_document_query': False,
            'enhanced_query': query,
            'original_query': query
        }

    specific_patterns = {
        'person_search': ['who is', 'who are', 'manager', 'svp', 'director', 'employee', 'staff'],
        'table_search': ['table', 'list', 'show me', 'names', 'details'],
        'number_search': ['how many', 'count', 'number of', 'total'],
        'date_search': ['when', 'date', 'time', 'schedule'],
        'location_search': ['where', 'location', 'place', 'office'],
        'experiment_search': ['experiment', 'lab', 'practical', 'exercise', 'tutorial', 'step by step', 'procedure', 'aim', 'objective']
    }

    query_type = 'general'
    enhanced_query = query

    document_indicators = [
        'document', 'pdf', 'file', 'page', 'section', 'chapter',
        'table', 'figure', 'chart', 'graph', 'image',
        'author', 'title', 'content', 'text', 'data'
    ]

    is_document_query = any(indicator in query_lower for indicator in document_indicators)

    if not is_document_query and len(query.split()) <= 3:
        return {
            'type': 'casual_conversation',
            'is_specific_search': False,
            'is_document_query': False,
            'enhanced_query': query,
            'original_query': query
        }

    for pattern_type, keywords in specific_patterns.items():
        if any(keyword in query_lower for keyword in keywords):
            query_type = pattern_type
            if pattern_type == 'person_search':
                enhanced_query = f"{query} name title role position manager director"
            elif pattern_type == 'table_search':
                enhanced_query = f"{query} table list data information"
            elif pattern_type == 'number_search':
                enhanced_query = f"{query} count number total amount quantity"
            break

    return {
        'type': query_type,
        'is_specific_search': query_type != 'general',
        'is_document_query': is_document_query,
        'enhanced_query': enhanced_query,
        'original_query': query
    }


def calculate_relevance_score(doc, query: str, query_analysis: dict) -> float:
    """Calculate enhanced relevance score"""
    try:
        score = 0.0
        content_lower = doc.page_content.lower()
        query_lower = query.lower()

        score += 0.2

        if query_lower in content_lower:
            score += 0.3

        query_words = set(query_lower.split())
        content_words = set(content_lower.split())
        overlap = len(query_words & content_words)
        if len(query_words) > 0:
            score += (overlap / len(query_words)) * 0.2

        metadata = doc.metadata

        try:
            from keybert import KeyBERT  # noqa: F401
            HAS_KEYBERT = True
        except ImportError:
            HAS_KEYBERT = False

        try:
            import spacy  # noqa: F401
            HAS_SPACY = True
        except ImportError:
            HAS_SPACY = False

        if HAS_KEYBERT and 'contextual_keywords' in metadata:
            keywords = metadata.get('contextual_keywords', [])
            keyword_matches = 0
            for keyword_data in keywords:
                if isinstance(keyword_data, tuple) and len(keyword_data) == 2:
                    keyword, confidence = keyword_data
                    try:
                        confidence = float(confidence)
                    except (ValueError, TypeError):
                        confidence = 0.5

                    if any(word in keyword.lower() for word in query_words):
                        keyword_matches += confidence
                elif isinstance(keyword_data, str):
                    if any(word in keyword_data.lower() for word in query_words):
                        keyword_matches += 0.5
            score += min(keyword_matches * 0.1, 0.3)

        if HAS_SPACY and 'named_entities' in metadata:
            entities = metadata.get('named_entities', [])
            entity_matches = 0
            for entity_text, entity_type in entities:
                if any(word in entity_text.lower() for word in query_words):
                    entity_matches += 1
            score += min(entity_matches * 0.1, 0.3)

        if 'sections' in metadata:
            sections = metadata.get('sections', [])
            for section in sections:
                if any(word in section.lower() for word in query_words):
                    score += 0.05

        # Content-type bonuses (tables/images etc.)
        doc_type = metadata.get('type', 'text')
        if query_analysis['type'] == 'person_search' and doc_type in ['table', 'image']:
            score += 0.2
        elif query_analysis['type'] == 'table_search' and doc_type == 'table':
            score += 0.2
        elif query_analysis['type'] == 'experiment_search' and doc_type == 'image':
            score += 0.3
        elif doc_type == 'image':
            score += 0.1
        elif query_analysis['type'] == 'number_search' and any(char.isdigit() for char in doc.page_content):
            score += 0.1

        if any(word in content_lower for word in ['badarinadh', 'gopal', 'sudha', 'veronica']):
            score += 0.2

        return min(score, 1.0)

    except Exception as e:
        print(f"Error calculating relevance score: {e}")
        return 0.5


def remove_duplicate_results(results: list) -> list:
    """Remove duplicate results while preserving highest relevance scores"""
    seen_content = {}

    for doc in results:
        content_hash = hashlib.md5(doc.page_content.encode()).hexdigest()
        relevance_score = doc.metadata.get('relevance_score', 0)

        if content_hash not in seen_content or seen_content[content_hash]['score'] < relevance_score:
            seen_content[content_hash] = {'doc': doc, 'score': relevance_score}

    return [item['doc'] for item in seen_content.values()]


def analyze_query_for_page_number(query: str) -> dict:
    """Detects if a query is asking for a specific page or slide"""
    patterns = [
        r'\b(page|slide|pg\.?|p\.?)\s+(\d+)\b',
        r'\b(\d+)\s*(?:st|nd|rd|th)?\s+(page|slide)\b'
    ]

    query_lower = query.lower()
    for pattern in patterns:
        match = re.search(pattern, query_lower)
        if match:
            # Pattern 1: "page 3" → group(2) is the number
            # Pattern 2: "3rd page" → group(1) is the number
            if match.group(1).isdigit():
                page_number = int(match.group(1))
            else:
                page_number = int(match.group(2))
            return {
                "is_page_specific": True,
                "page_number": page_number
            }

    return {"is_page_specific": False, "page_number": None}


def expand_query_for_fallback(query: str) -> str:
    """Expand query with broader terms for fallback search"""
    query_lower = query.lower()

    expansions = {
        'who': 'person name title role position',
        'what': 'information details data content',
        'when': 'date time schedule timeline',
        'where': 'location place office address',
        'how': 'method process procedure way',
        'why': 'reason cause purpose explanation',
        'manager': 'supervisor lead director head',
        'svp': 'senior vice president executive',
        'table': 'list data information chart',
        'details': 'information specifics data facts'
    }

    expanded_terms = []
    for word, expansion in expansions.items():
        if word in query_lower:
            expanded_terms.append(expansion)

    if expanded_terms:
        return f"{query} {' '.join(expanded_terms)}"

    return query


def filter_high_confidence_results(results: list, threshold: float = HIGH_CONFIDENCE_THRESHOLD) -> list:
    """
    Return only those results whose relevance_score >= threshold.
    We use this for:
      - selecting images to display
      - selecting sources to show in the UI
    """
    return [
        doc for doc in results
        if doc.metadata.get('relevance_score', 0.0) >= threshold
        or doc.metadata.get('high_confidence_match', False)
    ]
