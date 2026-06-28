# config.py - Configuration and Dependencies Management

import os
import warnings
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass
from typing import List, Dict, Any, Optional

# Suppress warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", message=".*deprecated.*")
warnings.filterwarnings("ignore", message=".*pin_memory.*", category=UserWarning)
warnings.filterwarnings("ignore", message=".*Neither CUDA nor MPS are available.*", category=UserWarning)

# Device detection for optimal performance
def get_optimal_device():
    """Detect and return the best available device for PyTorch operations"""
    try:
        import torch
        if torch.backends.mps.is_available() and torch.backends.mps.is_built():
            return 'mps'  # Apple Silicon GPU
        elif torch.cuda.is_available():
            return 'cuda'  # NVIDIA GPU
        else:
            return 'cpu'  # CPU fallback
    except ImportError:
        return 'cpu'

OPTIMAL_DEVICE = get_optimal_device()

# ---------- DEPENDENCY MANAGEMENT ----------

# Core dependencies (required)
try:
    from langchain_huggingface import HuggingFaceEmbeddings
except ImportError:
    try:
        from langchain_community.embeddings import HuggingFaceEmbeddings
    except ImportError:
        HuggingFaceEmbeddings = None

from langchain_community.document_loaders import PyPDFLoader
try:
    from langchain_community.document_loaders import PDFPlumberLoader
except ImportError:
    PDFPlumberLoader = None

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
# RetrievalQA moved around depending on langchain packaging.
# Try classic locations first (since you have langchain-classic installed),
# then fall back to older/newer layouts.
try:
    from langchain.chains import RetrievalQA  # works with classic-style installs
except Exception:
    try:
        from langchain_classic.chains import RetrievalQA  # if provided by langchain-classic
    except Exception:
        RetrievalQA = None

try:
    from langchain_community.llms import Ollama
except ImportError:
    Ollama = None

from langchain_core.prompts import PromptTemplate
from langchain_core.documents import Document
from langchain_core.callbacks import BaseCallbackHandler


# Optional dependencies with graceful fallbacks
try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False
    print("Warning: pandas not available - Excel processing limited")

try:
    import tabula
    HAS_TABULA = True
except ImportError:
    HAS_TABULA = False
    print("Warning: tabula-py not available - advanced table extraction limited")

try:
    import camelot
    HAS_CAMELOT = True
except ImportError:
    HAS_CAMELOT = False
    print("Warning: camelot-py not available - advanced table extraction limited")

try:
    import fitz  # PyMuPDF
    HAS_PYMUPDF = True
except ImportError:
    HAS_PYMUPDF = False
    print("Warning: PyMuPDF not available - image extraction disabled")

try:
    import easyocr
    HAS_OCR = True
except ImportError:
    HAS_OCR = False
    print("Warning: EasyOCR not available - OCR disabled")

try:
    from PIL import Image, ImageEnhance, ImageFilter
    import numpy as np
    HAS_PIL = True
except ImportError:
    HAS_PIL = False
    print("Warning: PIL not available - image processing limited")

try:
    import cv2
    HAS_OPENCV = True
except ImportError:
    HAS_OPENCV = False
    print("Warning: OpenCV not available - advanced image processing disabled")

try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    from matplotlib.patches import Rectangle
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("Warning: matplotlib not available - chart analysis limited")

try:
    import networkx as nx
    HAS_NETWORKX = True
except ImportError:
    HAS_NETWORKX = False

# Advanced NLP dependencies
try:
    from keybert import KeyBERT
    HAS_KEYBERT = True
except ImportError:
    HAS_KEYBERT = False
    print("Warning: KeyBERT not available - advanced keyword extraction disabled")

try:
    import spacy
    HAS_SPACY = True
except ImportError:
    HAS_SPACY = False
    print("Warning: spaCy not available - named entity recognition disabled")

# ---------- CONFIGURATION ----------

class RAGConfig:
    """Centralized configuration for the RAG system"""

    # Paths
    PROJECT_ROOT = Path(__file__).parent.absolute()
    EMBEDDINGS_DIR = PROJECT_ROOT / "embedding"
    DOCS_DIR = PROJECT_ROOT / "docs"
    IMAGES_DIR = PROJECT_ROOT / "images"

    @classmethod
    def get_embeddings_path(cls) -> str:
        """Get embeddings directory as string for compatibility"""
        return str(cls.EMBEDDINGS_DIR)

    # Chunking parameters - Enhanced for better experiment content preservation
    CHUNK_SIZE = 1200  # Increased for better context preservation
    CHUNK_OVERLAP = 200  # Increased overlap for better continuity

    # Model configurations - Enhanced for better experiment detection
    EMBED_MODEL_OPTIONS = [
        "sentence-transformers/all-MiniLM-L6-v2",
        "sentence-transformers/all-mpnet-base-v2",  # Better for technical content
        "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"  # Better multilingual support
    ]
    EMBED_MODEL = EMBED_MODEL_OPTIONS[1]  # Use better model for technical content
    # Override via .env (LLM_MODEL, OLLAMA_NUM_CTX). Default: tinyllama (low RAM).
    LLM_MODEL = os.getenv("LLM_MODEL", "tinyllama")
    OLLAMA_NUM_CTX = int(os.getenv("OLLAMA_NUM_CTX", "1024"))
    
    # Device configuration for optimal performance
    DEVICE = OPTIMAL_DEVICE
    DEVICE_NAME = {
        'mps': 'Apple Silicon GPU (MPS)',
        'cuda': 'NVIDIA GPU (CUDA)', 
        'cpu': 'CPU'
    }.get(OPTIMAL_DEVICE, 'CPU')

    # Company variants for logo detection
    COMPANY_VARIANTS = [
        "intouch", "intouchcx", "intouchx", "intouchc", "in touch",
        "intouch cx", "intouch x", "intouch c", "intouche", 
        "Intouch", "Intouchcx", "Intouchx", "Intouchc", "In touch",
        "Intouch cx", "Intouch x", "Intouch c", "Intouche", "Intouch/"
    ]

    # Processing thresholds
    IMAGE_MIN_SIZE = 75
    IMAGE_MAX_SIZE = 3000
    OCR_CONFIDENCE_THRESHOLD = 0.3
    TABLE_MIN_ROWS = 2

    # Conversation behavior
    # If False: do not include conversation context after the first answer
    USE_CONTEXT_AFTER_FIRST = False

    # Fast query behavior (short/simple questions)
    FAST_QUERY_K = 12  # Increased for better coverage
    FAST_QUERY_MAX_CHARS = 3000  # Increased for more context
    MAX_CONTEXT_CHARS = 10000  # Increased for comprehensive answers
    
    # Performance optimizations
    SEARCH_CACHE_SIZE = 200  # Increased cache size
    PARALLEL_SEARCH_THREADS = 4  # Parallel document search
    MIN_RELEVANCE_THRESHOLD = 0.3  # Minimum relevance score to include results
    
    # Advanced metadata processing settings
    ENABLE_KEYWORD_SEARCH = True
    
    # Multi-format processing settings - Only PDF and PPT supported
    SUPPORTED_AUDIO_FORMATS = []  # Removed - only PDF and PPT supported
    SUPPORTED_DOCUMENT_FORMATS = ['.pdf', '.pptx', '.ppt']  # Only PDF and PPT
    SUPPORTED_DATA_FORMATS = []  # Removed - only PDF and PPT supported
    SUPPORTED_TEXT_FORMATS = []  # Removed - only PDF and PPT supported
    SUPPORTED_ARCHIVE_FORMATS = []  # Removed - only PDF and PPT supported
    
    # Audio processing settings
    WHISPER_MODEL = "base"  # Can be: tiny, base, small, medium, large
    
    # Document processing settings
    MAX_FILE_SIZE_MB = 100  # Maximum file size in MB
    ENABLE_OCR_IN_DOCUMENTS = True
    ENABLE_TABLE_EXTRACTION = True
    
    # Text extraction settings
    DEFAULT_ENCODING = 'utf-8'
    FALLBACK_ENCODINGS = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']
    
    # Enhanced search settings
    ENABLE_ENTITY_SEARCH = True   # Use spaCy entities for better matching
    ENABLE_SEMANTIC_EXPANSION = True  # Expand queries with semantic synonyms
    KEYWORD_BOOST_FACTOR = 0.3   # Boost relevance for keyword matches
    ENTITY_BOOST_FACTOR = 0.4    # Boost relevance for entity matches
    CONTEXT_BOOST_FACTOR = 0.2   # Boost relevance for contextual matches
    
    # Query expansion settings
    MAX_QUERY_EXPANSIONS = 3     # Maximum number of query expansions
    SIMILARITY_THRESHOLD = 0.7   # Threshold for semantic similarity
    EXPANSION_DIVERSITY = 0.8    # Diversity factor for query expansion
    
    # Context tracking settings
    CONTEXT_MEMORY_SIZE = 20     # Number of previous queries to remember
    DOCUMENT_CONTEXT_WEIGHT = 0.7 # Weight for document-based context
    ENTITY_CONTEXT_WEIGHT = 0.3  # Weight for entity-based context

    # System capabilities
    CAPABILITIES = {
        'advanced_tables': HAS_TABULA or HAS_CAMELOT,
        'image_processing': HAS_PYMUPDF and HAS_OCR,
        'chart_analysis': HAS_MATPLOTLIB,
        'enhanced_search': True,
        'conversation_context': True,
        'multi_format': True,
        'keyword_extraction': HAS_KEYBERT,
        'entity_recognition': HAS_SPACY,
        'semantic_expansion': HAS_KEYBERT and HAS_SPACY,
        'advanced_metadata': HAS_KEYBERT and HAS_SPACY
    }

    @classmethod
    def get_capability_status(cls) -> Dict[str, str]:
        """Get formatted capability status"""
        status = {}
        status['device'] = f"✅ ({cls.DEVICE_NAME})"
        status['tables'] = f"{'✅' if cls.CAPABILITIES['advanced_tables'] else '❌'} ({'Tabula' if HAS_TABULA else ''}{' + Camelot' if HAS_CAMELOT else 'Pattern-based only'})"
        status['images'] = f"{'✅' if cls.CAPABILITIES['image_processing'] else '❌'} ({'PyMuPDF + EasyOCR' if HAS_PYMUPDF and HAS_OCR else 'Limited'})"
        status['charts'] = f"{'✅' if cls.CAPABILITIES['chart_analysis'] else '❌'} ({'Advanced' if HAS_MATPLOTLIB else 'Basic OCR only'})"
        status['search'] = "✅ (Intelligent context-aware)"
        status['context'] = "✅ (Entity and topic tracking)"
        status['formats'] = "✅ (PDF and PPT only)"
        status['keywords'] = f"{'✅' if cls.CAPABILITIES['keyword_extraction'] else '❌'} ({'KeyBERT' if HAS_KEYBERT else 'Basic'})"
        status['entities'] = f"{'✅' if cls.CAPABILITIES['entity_recognition'] else '❌'} ({'spaCy NER' if HAS_SPACY else 'Basic'})"
        status['semantic'] = f"{'✅' if cls.CAPABILITIES['semantic_expansion'] else '❌'} ({'Advanced' if HAS_KEYBERT and HAS_SPACY else 'Basic'})"
        return status

# ---------- PROMPT TEMPLATES ----------

class PromptTemplates:
    """Centralized prompt templates with intelligent, context-aware analysis"""

    ADVANCED_CONTEXT_QA_TEMPLATE = """You are an expert document analyst with advanced reasoning capabilities. 
Provide intelligent, context-aware answers using both the document content and conversation context. 
Try keeping the answers a little short, not to be too lengthy, and usually to the point; depending on the question asked.
If asked for document names (do not give entire analysis, page specific analysis or content analysis, just the document name) or roles or anything specific person names, try keeping it very concise and brief.
Adapt your style depending on the type of question: general explanation, data extraction, structured analysis, or contextual reasoning.

CONVERSATION CONTEXT:
{conversation_context}

DOCUMENT CONTENT:
{context}

QUESTION: {question}

ANALYSIS INSTRUCTIONS:
1. REFERENCE RESOLUTION: If the question includes vague references (it, this, that, previous, etc.), resolve them using the conversation context.

2. CONTEXTUAL UNDERSTANDING: Interpret the question carefully:
   - If general, provide a broad yet relevant explanation
   - If specific, extract precise information from the document
   - Adjust depth of analysis to match the question type

3. DATA HANDLING: When the answer involves structured data (tables, lists, charts, records):
   - Extract all relevant details accurately
   - Highlight patterns, groupings, or relationships
   - Present findings clearly and systematically
   - Be precise and avoid assumptions

4. INSIGHT GENERATION: Beyond extraction, explain:
   - Why the information matters
   - What relationships, trends, or hierarchies exist
   - Any broader significance in context

5. VISUAL/CHART ANALYSIS: For charts or images:
   - Identify the type and purpose of the visualization
   - Extract numerical data, categories, and patterns
   - Provide insights and implications connected to the question

6. SOURCE ATTRIBUTION: Always cite sources clearly:
   - Document name/section
   - Page numbers when available

7. RESPONSE FORMAT: Structure the answer for clarity:
   - Use numbered or bulleted points for multiple items
   - Group related information together
   - Keep concise unless detail is explicitly required

8. COMPLETENESS: Cover all relevant aspects of the question without omitting important details.

9. CONSISTENCY: Ensure consistency across multiple queries about the same information.

10. GENDER NEUTRALITY: When referring to people, use gender-neutral language. Use "they" instead of "he/she", "this person" instead of "he/she", and avoid gender-specific pronouns unless explicitly specified in the source material.

OUTPUT STYLE:
- Start with a short direct answer or brief summary
- If needed, add a simple breakdown (e.g., key points, data, insights)
- End with a short source note (document name/section/page if specifically asked for)
- Format your response for clarity. Use bullet points (using '*') for lists and bold text (using '**text**') for emphasis. Avoid other markdown like headers (e.g., #, ##).


ANSWER:"""

    CHART_ANALYSIS_TEMPLATE = """You are a data visualization expert. Provide a complete, intelligent analysis of the chart or graph.

CHART/VISUAL CONTENT:
{chart_content}

CONVERSATION CONTEXT:
{conversation_context}

QUESTION: {question}

CHART ANALYSIS INSTRUCTIONS:
1. IDENTIFY the chart type and its main purpose.
2. EXTRACT visible details:
   - Labels, categories, axes, values, scales
   - Trends, peaks, troughs, anomalies
3. ANALYZE patterns:
   - Upward/downward trends
   - Comparisons across categories
   - Cycles or irregularities
4. GENERATE INSIGHTS:
   - What the data reveals
   - Why it matters in context
   - Implications for the scenario
5. CONNECT findings to the surrounding context or prior conversation.
6. STRUCTURE the response with clarity and systematic explanation.
7. Provide SOURCE attribution at the end.

ANSWER:"""

    TABLE_ANALYSIS_TEMPLATE = """You are a data analysis expert. Provide intelligent, structured analysis of the table in context.

TABLE CONTENT:
{table_content}

CONVERSATION CONTEXT:
{conversation_context}

QUESTION: {question}

TABLE ANALYSIS INSTRUCTIONS:
1. DATA EXTRACTION: Extract all relevant entries, attributes, and values from the table data.
2. ACCURACY: Be precise and accurate - only state what is explicitly shown in the table data.
3. AVOID ASSUMPTIONS: Do not make assumptions or interpretations beyond what is clearly stated in the table.
4. PATTERN & RELATIONSHIP ANALYSIS:
   - Groupings, categories, or hierarchies
   - Notable trends, anomalies, or dependencies
5. CONTEXTUAL SIGNIFICANCE:
   - Purpose of the table in the document
   - Why the data matters for the question
   - Broader meaning in organizational or analytical context
6. STRUCTURE the answer clearly:
   - Start with a summary answer based on table data
   - Follow with detailed breakdown from the table
   - Organize into sections (Data Extraction, Relationships, Insights)
7. SOURCE ATTRIBUTION: Include source details (document name, page numbers if available, type of content).
8. CONSISTENCY: If multiple tables contain similar information, ensure consistency in your analysis.
9. GENDER NEUTRALITY: When referring to people, use gender-neutral language. Use "they" instead of "he/she", "this person" instead of "he/she", and avoid gender-specific pronouns unless explicitly specified in the source material.

ANSWER:"""

    DOCUMENT_SUMMARY_TEMPLATE = """You are an expert document summarizer. Provide a comprehensive, detailed summary of the document content.

CONTEXT:
{context}

CONVERSATION CONTEXT:
{conversation_context}

INSTRUCTIONS:
1. COMPREHENSIVE ANALYSIS: Analyze ALL the provided content to create a complete summary
2. STRUCTURE: Organize the summary with clear sections:
   - Document Overview (purpose, scope, main topic)
   - Key Concepts and Topics
   - Important Details and Findings
   - Technical Information (if applicable)
   - Conclusions or Outcomes
3. DETAILED CONTENT: Include specific details, numbers, names, and important information
4. COMPLETE COVERAGE: Ensure you cover all major aspects of the document
5. CLARITY: Write in clear, professional language
6. SOURCE ATTRIBUTION: Include source document information
7. COMPREHENSIVE SCOPE: This should be a thorough summary, not just key points

QUESTION: {question}

Provide a detailed, comprehensive summary of the document:"""

    PAGE_SPECIFIC_TEMPLATE = """You are an expert document analyst specializing in page-specific content analysis. 
Provide a detailed and accurate answer based ONLY on the content from the specific page(s) provided below.

CONVERSATION CONTEXT:
{conversation_context}

PAGE-SPECIFIC CONTENT:
{context}

QUESTION: {question}

PAGE-SPECIFIC ANALYSIS INSTRUCTIONS:
1.  **Strict Focus:** Confine your answer *exclusively* to the information within the "PAGE-SPECIFIC CONTENT" section. Do not use any external knowledge or information from other parts of the document unless it is provided.
2.  **Describe Everything:** Analyze and describe all content on the page, including text, data, and any descriptions of images or charts.
3.  **Direct Answer:** If the user asks what is on a specific page, describe its contents comprehensively.
4.  **Source Attribution:** State clearly that your answer is based on the content of the requested page (e.g., "Based on the content of slide 47...").
5.  **If Content is Missing:** If the provided context does not contain the requested information, state clearly: "The provided content for the requested page does not contain that information."

ANSWER:"""

    UI_ELEMENT_TEMPLATE = """You are a UI/UX expert analyzing user interface elements from application screenshots. 
Provide detailed, accurate information about UI elements like popups, dialogs, screens, tabs, buttons, and interface components.

CONVERSATION CONTEXT:
{conversation_context}

UI ELEMENT CONTENT:
{context}

QUESTION: {question}

UI ELEMENT ANALYSIS INSTRUCTIONS:
1. UI IDENTIFICATION: Identify all UI elements in the screenshots:
   - Popup/Dialog names and titles
   - Screen/Page names
   - Tab names and navigation elements
   - Button text and interactive elements
   - Labels, fields, and form elements
2. STRUCTURED EXTRACTION: Extract UI element details systematically:
   - Name of the UI element (exact text)
   - Type of element (popup, dialog, screen, tab, button, etc.)
   - Context where it appears
   - Related or surrounding elements
3. CONTEXTUAL UNDERSTANDING: Explain:
   - Purpose of the UI element
   - When/how it appears
   - What actions it enables
   - Relationship to other UI elements
4. PRECISE TERMINOLOGY: Use exact names/text as shown in the screenshots
5. HIERARCHICAL ORGANIZATION: Show UI element relationships:
   - Parent-child relationships (popup contains tabs, screen has buttons)
   - Navigation flow (what leads to what)
6. ANNOTATIONS & CALLOUTS: Include any explanatory text, annotations, or callouts shown
7. SOURCE ATTRIBUTION: Reference specific slides/pages where UI elements appear
8. COMPLETENESS: Cover all requested UI elements comprehensively

RESPONSE FORMAT:
- Start with direct answer to the question
- List UI element names with their types
- Provide context for each element
- Explain relationships if relevant
- Include source references (slide/page numbers)

ANSWER:"""

# ---------- UTILITY FUNCTIONS ----------

def ensure_directory(path: str) -> None:
    """Ensure directory exists"""
    os.makedirs(path, exist_ok=True)

def get_pdf_name(pdf_path: str) -> str:
    """Extract PDF name without extension and replace spaces with underscores"""
    return Path(pdf_path).stem.replace(" ", "_")

def validate_pdf_path(pdf_path: str) -> bool:
    """Validate PDF file exists (legacy function - use validate_document_path for multi-format)"""
    return os.path.isfile(pdf_path) and pdf_path.lower().endswith('.pdf')

def validate_document_path(file_path: str) -> bool:
    """Validate document file exists and is supported format"""
    if not os.path.isfile(file_path):
        return False
    
    # Import here to avoid circular imports
    try:
        from unified_document_processor import FileTypeDetector
        return FileTypeDetector.is_supported(file_path)
    except ImportError:
        # Fallback to PDF only if unified_document_processor not available
        return file_path.lower().endswith('.pdf')

def get_document_name(file_path: str) -> str:
    """Extract document name without extension and replace spaces with underscores (multi-format)"""
    return Path(file_path).stem.replace(" ", "_")

def get_supported_formats() -> List[str]:
    """Get list of all supported file formats - Only PDF and PPT"""
    try:
        from unified_document_processor import FileTypeDetector
        return FileTypeDetector.get_supported_extensions()
    except ImportError:
        return ['.pdf', '.pptx', '.ppt']  # Fallback

# ---------- GLOBAL OBJECTS ----------

config = RAGConfig()
templates = PromptTemplates()

# Expose key configs at module level (for easy import)
EMBED_MODEL_OPTIONS = RAGConfig.EMBED_MODEL_OPTIONS
EMBED_MODEL = RAGConfig.EMBED_MODEL
LLM_MODEL = RAGConfig.LLM_MODEL

# Expose template strings at module level
TABLE_ANALYSIS_TEMPLATE = PromptTemplates.TABLE_ANALYSIS_TEMPLATE
CHART_ANALYSIS_TEMPLATE = PromptTemplates.CHART_ANALYSIS_TEMPLATE
ADVANCED_CONTEXT_QA_TEMPLATE = PromptTemplates.ADVANCED_CONTEXT_QA_TEMPLATE
DOCUMENT_SUMMARY_TEMPLATE = PromptTemplates.DOCUMENT_SUMMARY_TEMPLATE
PAGE_SPECIFIC_TEMPLATE = PromptTemplates.PAGE_SPECIFIC_TEMPLATE
UI_ELEMENT_TEMPLATE = PromptTemplates.UI_ELEMENT_TEMPLATE