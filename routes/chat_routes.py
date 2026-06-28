# # chat_routes.py
# import json
# import time
# import os
# import re
# import traceback
# import logging
# from pathlib import Path
# from flask import Blueprint, jsonify, request, session, Response, stream_with_context, render_template

# from modules.conversation import detect_casual_conversation, generate_casual_response
# from modules.search import (
#     intelligent_document_search,
#     analyze_query_for_page_number,
#     filter_high_confidence_results,
# )
# from modules.response_generator import (
#     generate_prompt_based_on_mode,
#     build_context_from_results,
#     format_sources_for_response,
#     detect_follow_up_query
# )
# from modules.utils import (
#     find_mentioned_document,
#     format_loaded_documents_response,
#     is_query_about_documents,
#     is_referential_query
# )
# from modules.auth import require_login
# from app_state import get_system_state
# from config import PAGE_SPECIFIC_TEMPLATE, config as app_config
# from urllib.parse import urljoin

# chat_bp = Blueprint('chat', __name__, url_prefix='/')
# logger = logging.getLogger(__name__)

# def get_managers_and_llm():
#     state_data = get_system_state()
#     system_initialized, initialization_error, doc_manager, context_manager, _, _, llm = state_data
#     if not system_initialized:
#         return None, None, None, initialization_error or "Unknown initialization error"
#     return doc_manager, context_manager, llm, None

# # =========================
# # Image URL + scoring utils
# # =========================

# def _embed_url(doc_name: str, rel_path: str) -> str:
#     """
#     Build a fully-qualified URL for an embedding asset using the current request host.
#     Example: http://127.0.0.1:9072/embedding/<doc_name>/<rel_path>
#     """
#     # Normalize the relative path
#     rel_path = rel_path.lstrip('/\\')
#     if not rel_path.lower().startswith(('images/', 'tables/')):
#         rel_path = f"images/{rel_path}"

#     # Build a path relative to the API root, then join with request.host_url to add scheme+host+port
#     rel = f"embedding/{doc_name}/{rel_path}"
#     return urljoin(request.host_url, rel)


# def _normalize_tokens(text: str) -> set:
#     text = (text or "").lower()
#     text = re.sub(r'[^a-z0-9\s]', ' ', text)
#     toks = [t for t in text.split() if len(t) > 1]
#     return set(toks)

# def _clean_caption_text(text: str | None) -> str:
#     """
#     Remove markdown image tags and image-ish/kbase URLs from captions.
#     Falls back to trimmed plain text.
#     """
#     if not text:
#         return ""
#     return _strip_external_image_links(text)

# def _text_fields_from_image_meta(m: dict) -> str:
#     fields = []
#     for k in ("caption","title","alt","ocr_text","context_text","text","description",
#               "filename","relative_path","image_relative_path","source_url"):
#         v = m.get(k)
#         if isinstance(v, str) and v.strip():
#             fields.append(v)
#     return " ".join(fields)

# def _image_type_hint_score(query_tokens: set, meta_text: str) -> float:
#     score = 0.0
#     mt = (meta_text or "").lower()
#     if any(x in query_tokens for x in {"table","tabular","excel"}):
#         if "table" in mt: score += 1.0
#     if any(x in query_tokens for x in {"screenshot","login","ui","screen"}):
#         if any(w in mt for w in ["screenshot","screen","ui","login"]): score += 0.8
#     return score

# def _score_image_meta(m: dict, query_tokens: set, candidate_pages: set[int]) -> float:
#     score = 0.0
#     if m.get("referenced_in_md"):
#         score += 1.5
#     page_val = None
#     for k in ("display_page","page","page_number","page_num"):
#         v = m.get(k)
#         if isinstance(v, int):
#             page_val = v; break
#         if isinstance(v, str) and v.isdigit():
#             page_val = int(v); break
#     if page_val is not None and page_val in candidate_pages:
#         score += 4.0
#     meta_text = _text_fields_from_image_meta(m)
#     meta_tokens = _normalize_tokens(meta_text)
#     if meta_tokens:
#         overlap = query_tokens.intersection(meta_tokens)
#         score += min(len(overlap), 6) * 1.0
#     score += _image_type_hint_score(query_tokens, meta_text)
#     return score

# def _derive_rel_from_abs(abs_path: str, doc_name: str) -> str | None:
#     try:
#         base = os.path.normpath(os.path.join(app_config.EMBEDDINGS_DIR, doc_name))
#         ap = os.path.normpath(abs_path)
#         if ap.startswith(base + os.sep):
#             return os.path.relpath(ap, base).replace("\\", "/")
#     except Exception:
#         pass
#     return None

# # ---------- images manifest resolution (handles renamed image paths) ----------
# _IMAGES_MANIFEST_CACHE: dict[str, dict[str, str]] = {}  # {doc_name: {filename -> relative_path}}

# def _load_images_manifest_for_doc(doc_name: str) -> dict[str, str]:
#     """
#     Load and cache the mapping of filename -> relative_path from images_metadata.json
#     so we can find renamed copies like foo.png -> images/foo-1.png.
#     """
#     if doc_name in _IMAGES_MANIFEST_CACHE:
#         return _IMAGES_MANIFEST_CACHE[doc_name]

#     manifest = {}
#     try:
#         manifest_file = Path(app_config.EMBEDDINGS_DIR) / doc_name / "images" / "images_metadata.json"
#         if manifest_file.exists():
#             data = json.loads(manifest_file.read_text(encoding="utf-8"))
#             if isinstance(data, list):
#                 for item in data:
#                     fn = item.get("filename") or os.path.basename(item.get("relative_path", "") or "")
#                     rp = item.get("relative_path")
#                     if fn and rp:
#                         manifest[fn] = rp
#     except Exception as e:
#         logger.warning(f"Failed to load images manifest for {doc_name}: {e}")

#     _IMAGES_MANIFEST_CACHE[doc_name] = manifest
#     return manifest


# def _resolve_rel_via_manifest(doc_name: str, rel_or_filename: str | None) -> str | None:
#     """
#     Given either a relative path or filename, resolve to the *actual* copied
#     path as per images_metadata.json. Falls back to checking existence in /images/.
#     """
#     if not rel_or_filename:
#         return None

#     key = rel_or_filename.lstrip("/\\")
#     if key.lower().startswith("images/"):
#         key = key[len("images/"):]

#     manifest = _load_images_manifest_for_doc(doc_name)

#     # Direct match
#     if rel_or_filename in manifest.values():
#         return rel_or_filename

#     # Filename match
#     if key in manifest:
#         return manifest[key]

#     # Fallback: check physical existence
#     candidate = Path(app_config.EMBEDDINGS_DIR) / doc_name / "images" / key
#     if candidate.exists():
#         return f"images/{key}"

#     return None


# def _collect_image_candidates(results, max_docs: int = 30):
#     cand = []
#     for d in results[:max_docs]:
#         meta = d.metadata or {}
#         if meta.get('type') != 'image':
#             continue

#         doc_name = (
#             meta.get('document') or meta.get('document_name') or
#             meta.get('source_document') or meta.get('doc_name')
#         )

#         # Prefer explicit relative path; else derive from abs path; else filename
#         rel = meta.get('image_relative_path') or meta.get('relative_path')
#         if not rel:
#             abs_img = meta.get('image_path') or meta.get('path')
#             if abs_img and doc_name:
#                 rel = _derive_rel_from_abs(abs_img, doc_name)

#         filename = meta.get('image_filename') or meta.get('filename')
#         # If still nothing, try manifest based resolution by filename
#         rel_fixed = _resolve_rel_via_manifest(doc_name, rel or filename)
#         if not (doc_name and rel_fixed):
#             continue

#         # Build our LOCAL embed url (not kbase)
#         url = _embed_url(doc_name, rel_fixed)

#         cand.append({
#             "doc_name": doc_name,
#             "relative_path": rel_fixed,           # ✅ always local relative path
#             "display_page": meta.get('display_page', 1),
#             "md_index": meta.get('md_index'),
#             "alt": meta.get('alt') or "",
#             "context_text": _clean_caption_text(meta.get('context_text')),  # ✅ scrub kbase links
#             "ocr_text": meta.get('ocr_text') or "",
#             "source_url": meta.get('source_url'),  # kept only as FYI (not used for display)
#             "filename": filename or os.path.basename(rel_fixed),
#             "__meta": meta,
#             "url": url                              # ready URL
#         })
#     return cand



# def _select_ordered_images(user_message: str, results, limit: int = 6, candidate_pages: set[int] | None = None):
#     cand = _collect_image_candidates(results, max_docs=30)
#     if not cand:
#         return []
#     qtokens = _normalize_tokens(user_message)
#     candidate_pages = candidate_pages or set()
#     scored = []
#     for c in cand:
#         score = _score_image_meta(c["__meta"], qtokens, candidate_pages)
#         scored.append((score, c))
#     def _key(item):
#         score, c = item
#         mdi = c["md_index"]
#         mdi_key = 10**9 if mdi is None else mdi
#         return (-score, mdi_key, c["display_page"])
#     scored.sort(key=_key)
#     out = []
#     for _, c in scored[:limit]:
#         url = _embed_url(c["doc_name"], c["relative_path"])
#         out.append({
#             "url": url,
#             "doc_name": c["doc_name"],
#             "relative_path": c["relative_path"],
#             "display_page": c["display_page"],
#             "md_index": c["md_index"],
#             "alt": c["alt"] or "",
#             "context_text": c["context_text"] or "",
#             "filename": c["filename"]
#         })
#     return out

# def _build_blocks(response_text: str, images: list[dict]):
#     paras = [p.strip() for p in re.split(r'\n\s*\n+', response_text or "") if p.strip()]
#     blocks = []
#     img_idx = 0
#     for p in paras:
#         blocks.append({"type": "text", "content": p})
#         if img_idx < len(images):
#             blocks.append({"type": "image", **images[img_idx]})
#             img_idx += 1
#     while img_idx < len(images):
#         blocks.append({"type": "image", **images[img_idx]})
#         img_idx += 1
#     return blocks

# # =======================
# # Streaming sanitization
# # =======================

# # complete markdown image tags: ![alt](url)
# _MD_IMG_RE = re.compile(r'!\[[^\]]*\]\([^)]+\)')

# # links that *behave* like images or are KBase attachments, e.g. [x](https://kbase.../download/attachments/...)
# _MD_LINK_KBASE_RE = re.compile(
#     r'\[[^\]]*\]\((?P<url>https?://[^\s)]+(?:download/attachments|/attachments/)[^\s)]*)\)',
#     re.IGNORECASE
# )

# # bare URLs that look like images or KBase attachments (with or without extension / with ?api=v2)
# _ANY_IMGISH_URL_RE = re.compile(
#     r'(https?://[^\s)]+(?:\.(?:png|jpe?g|gif|webp|bmp|svg)(?:\?[^\s)]*)?|\?api=v2|download/attachments/[^\s)]*))',
#     re.IGNORECASE
# )

# def _strip_external_image_links(text: str) -> str:
#     if not text:
#         return text
#     # Remove explicit markdown image tags
#     text = _MD_IMG_RE.sub('', text)
#     # Remove KBase attachment-style links written as markdown links
#     text = _MD_LINK_KBASE_RE.sub('', text)
#     # Remove bare "image-ish" URLs
#     text = _ANY_IMGISH_URL_RE.sub('', text)
#     # Collapse excess blank lines
#     text = re.sub(r'\n{3,}', '\n\n', text)
#     return text.strip()

# def _incremental_sanitizer():
#     """
#     Returns a callable that, when fed streaming chunks, emits only the *new* sanitized
#     delta. It also suppresses *partial* in-flight patterns like '![](' or '[alt]('
#     that haven't closed with a ')' yet.
#     """
#     raw_so_far = ""
#     clean_so_far = ""

#     IMG_START_RE = re.compile(r'!\[[^\]]*?\]\([^\)]*?$')      # start of ![]( ... not closed
#     LINK_START_RE = re.compile(r'\[[^\]]*?\]\([^\)]*?$')      # start of [ ]( ... not closed

#     def feed(chunk: str) -> str:
#         nonlocal raw_so_far, clean_so_far
#         if not chunk:
#             return ""
#         raw_so_far += str(chunk)

#         # First pass: strip any complete patterns we can already see
#         sanitized_full = _strip_external_image_links(raw_so_far)

#         # Compute the new delta tentatively
#         tentative_delta = sanitized_full[len(clean_so_far):]

#         # Hold back any trailing incomplete image/link tag starts to avoid UI flicker
#         tail = sanitized_full  # work on the entire current sanitized buffer
#         holdback_from = None

#         # Look for a dangling start at the end (no closing ')')
#         m1 = IMG_START_RE.search(raw_so_far)
#         m2 = LINK_START_RE.search(raw_so_far)
#         # If either pattern is currently open (in raw), with no closing paren yet,
#         # and its start lies inside the yet-to-be-emitted region, hold back from there.
#         for m in (m1, m2):
#             if m:
#                 start_idx_in_raw = m.start()
#                 emitted_len_est = len(clean_so_far)
#                 if start_idx_in_raw >= emitted_len_est:
#                     holdback_from = emitted_len_est
#                     break

#         if holdback_from is not None:
#             tentative_delta = ""  # suppress until the tag closes in a later chunk

#         if tentative_delta:
#             clean_so_far += tentative_delta
#             return tentative_delta
#         return ""
#     return feed

# # ==============
# # UI route
# # ==============

# @chat_bp.route('/chatbot')
# @require_login
# def chatbot():
#     try:
#         return render_template('chatbot_new.html')
#     except Exception as e:
#         return f"Template error: {e}", 500

# # ==============
# # Chat API
# # ==============

# @chat_bp.route('/api/chat', methods=['POST'])
# @require_login
# def api_chat():
#     doc_manager, context_manager, llm, error = get_managers_and_llm()
#     if error:
#         return jsonify({'error': 'System not initialized', 'message': error}), 500
#     if llm is None:
#         return jsonify({'error': 'System not initialized', 'message': 'Language model (LLM) is not available.'}), 500

#     try:
#         data = request.get_json()
#         if not data or 'message' not in data:
#             return jsonify({'error': 'No message provided'}), 400

#         user_message = data['message'].strip()
#         if not user_message:
#             return jsonify({'error': 'Empty message'}), 400

#         selected_documents = data.get('loaded_documents', [])
#         search_mode = data.get('search_mode', 'documents_only')
#         lower_msg = user_message.lower()

#         # 1) casual
#         conversation_response = detect_casual_conversation(user_message, selected_documents)
#         if conversation_response:
#             return jsonify({
#                 'response': conversation_response,
#                 'query_type': 'casual_conversation',
#                 'sources': [],
#                 'processing_time': 0.0,
#                 'images': [],
#                 'blocks': [{"type": "text", "content": conversation_response}]
#             })

#         # 2) commands
#         if lower_msg in ['clear', 'reset']:
#             context_manager.clear_history()
#             if hasattr(doc_manager, 'search_cache'):
#                 doc_manager.search_cache.clear()
#             msg = 'Conversation history cleared.'
#             return jsonify({
#                 'response': msg,
#                 'query_type': 'command',
#                 'sources': [],
#                 'processing_time': 0.0,
#                 'images': [],
#                 'blocks': [{"type": "text", "content": msg}]
#             })

#         # 3) memory set
#         if lower_msg.startswith('remember '):
#             content = user_message[len('remember '):].strip()
#             remembered = False
#             if ' is ' in content:
#                 parts = content.split(' is ', 1)
#                 key = parts[0].replace('that ', '').strip(' :.-').strip()
#                 value = parts[1].strip(' .').strip()
#                 if key and value:
#                     context_manager.add_fact(key, value)
#                     remembered = True
#             elif '=' in content:
#                 parts = content.split('=', 1)
#                 key = parts[0].replace('that ', '').strip(' :.-').strip()
#                 value = parts[1].strip(' .').strip()
#                 if key and value:
#                     context_manager.add_fact(key, value)
#                     remembered = True
#             else:
#                 if 'notes' not in context_manager.user_memory:
#                     context_manager.user_memory['notes'] = []
#                 context_manager.user_memory['notes'].append(content)
#                 remembered = True
#             msg = 'Got it, I will remember that.' if remembered else 'Sorry, I could not understand what to remember.'
#             return jsonify({
#                 'response': msg,
#                 'query_type': 'memory_update',
#                 'sources': [],
#                 'processing_time': 0.0,
#                 'images': [],
#                 'blocks': [{"type": "text", "content": msg}]
#             })

#         # 4) quick fact lookup
#         if (lower_msg.startswith('what is ') or lower_msg.startswith('who is ') or lower_msg.startswith('tell me ')) and len(lower_msg.split()) <= 8:
#             key = user_message
#             for prefix in ['what is ', 'who is ', 'tell me about ', 'tell me the ', 'tell me ']:
#                 if lower_msg.startswith(prefix):
#                     key = user_message[len(prefix):].strip(' ?!.')
#                     break
#             fact = context_manager.get_fact(key)
#             if fact:
#                 ans = f"{key.capitalize()}: {fact}"
#                 return jsonify({
#                     'response': ans,
#                     'query_type': 'memory_lookup',
#                     'sources': [],
#                     'processing_time': 0.0,
#                     'images': [],
#                     'blocks': [{"type": "text", "content": ans}]
#                 })

#         # 5) check facts before search
#         stored_fact = context_manager.check_facts_before_search(user_message)
#         if stored_fact:
#             return jsonify({
#                 'response': stored_fact,
#                 'query_type': 'fact_lookup',
#                 'sources': [],
#                 'processing_time': 0.0,
#                 'images': [],
#                 'blocks': [{"type": "text", "content": stored_fact}]
#             })

#         # 6) doc listing/count
#         count_triggers = ['how many docs', 'how many documents', 'number of docs', 'docs count',
#                           'how many files', 'files count', 'number of files']
#         if is_query_about_documents(user_message):
#             if any(t in lower_msg for t in count_triggers):
#                 names = sorted(list(doc_manager.loaded_documents.keys())) if doc_manager and doc_manager.loaded_documents else []
#                 count = len(names)
#                 summary = f"Loaded documents: {count}" + (f" ({', '.join(names)})" if count > 0 else "")
#             else:
#                 summary = format_loaded_documents_response(doc_manager, selected_documents)
#             return jsonify({
#                 'response': summary,
#                 'query_type': 'system_info',
#                 'sources': [],
#                 'processing_time': 0.0,
#                 'images': [],
#                 'blocks': [{"type": "text", "content": summary}]
#             })

#         # 7) require docs
#         if not doc_manager.loaded_documents:
#             return jsonify({'error': 'No documents loaded', 'message': 'Please load at least one document before asking questions.'}), 400

#         # 8) streaming?
#         stream_requested = data.get('stream', False)
#         if isinstance(stream_requested, str):
#             stream_requested = stream_requested.lower() in ['true', '1', 'yes']

#         if stream_requested:
#             def generate_stream():
#                 start_time = time.time()
#                 try:
#                     search_docs_list = list(doc_manager.loaded_documents.keys())
#                     if selected_documents:
#                         search_docs_list = [d for d in search_docs_list if d in selected_documents]

#                     mentioned = find_mentioned_document(user_message, search_docs_list)
#                     if mentioned:
#                         search_docs_list = [mentioned]

#                     if not search_docs_list:
#                         yield f"data: {json.dumps({'type': 'token', 'content': 'Please select at least one document to search.'})}\n\n"
#                         yield f"data: {json.dumps({'type': 'done', 'sources': [], 'processing_time': 0.0, 'images': [], 'blocks': []})}\n\n"
#                         return

#                     yield ": ping\n\n"
#                     yield f"data: {json.dumps({'type': 'status', 'message': 'searching across documents'})}\n\n"

#                     followup_force_context = detect_follow_up_query(user_message)
#                     conversation_context = context_manager.get_enhanced_context_for_query(user_message, force=followup_force_context)

#                     is_fast = len(user_message.split()) <= 6
#                     CONTEXT_K = 15

#                     search_results, _ = intelligent_document_search(
#                         user_message,
#                         doc_manager,
#                         k=CONTEXT_K,
#                         fast=is_fast,
#                         selected_documents=search_docs_list,
#                         context_manager=context_manager
#                     )

#                     if not search_results:
#                         msg = "No relevant information found in the loaded documents."
#                         yield f"data: {json.dumps({'type': 'token', 'content': msg})}\n\n"
#                         yield f"data: {json.dumps({'type': 'done', 'sources': [], 'processing_time': 0.0, 'images': [], 'blocks': []})}\n\n"
#                         return

#                     # 🔸 High-confidence filter (>= 70% match)
#                     high_conf_results = filter_high_confidence_results(search_results, threshold=0.7)
#                     effective_results = high_conf_results or search_results

#                     # Page hint
#                     query_analysis = analyze_query_for_page_number(user_message)
#                     candidate_pages = set()
#                     if query_analysis["is_page_specific"]:
#                         candidate_pages.add(query_analysis["page_number"])

#                     # curated local images only from high-confidence/effective results
#                     images = _select_ordered_images(user_message, effective_results, limit=6, candidate_pages=candidate_pages)

#                     combined_context = build_context_from_results(effective_results, 8, is_fast)
#                     bot_name = context_manager.get_memory('bot_name', 'Catapult')

#                     if query_analysis["is_page_specific"]:
#                         page_num = query_analysis['page_number']
#                         filtered = [d for d in effective_results if d.metadata.get('display_page') == page_num]
#                         if filtered:
#                             effective_results = filtered
#                         combined_context = build_context_from_results(effective_results, 8, is_fast)
#                         prompt = PAGE_SPECIFIC_TEMPLATE.format(
#                             conversation_context=conversation_context,
#                             context=combined_context,
#                             question=user_message
#                         )
#                     else:
#                         prompt = generate_prompt_based_on_mode(
#                             search_mode, bot_name, user_message,
#                             combined_context, conversation_context, followup_force_context
#                         )

#                     # Send the curated local images first (only from high-confidence docs)
#                     yield f"data: {json.dumps({'type': 'images', 'images': images})}\n\n"

#                     # Stream + sanitize (blocks image markdown/URLs, incl. partial starts)
#                     sanitizer = _incremental_sanitizer()
#                     full_response_text = ""
#                     for chunk in llm.stream(prompt):
#                         safe_delta = sanitizer(str(chunk))
#                         if not safe_delta:
#                             continue
#                         full_response_text += safe_delta
#                         yield f"data: {json.dumps({'type': 'token', 'content': safe_delta})}\n\n"

#                     formatted_sources = format_sources_for_response(effective_results, 8)

#                     context_manager.add_exchange(
#                         question=user_message,
#                         answer=full_response_text,
#                         sources=effective_results[:8],
#                         context_used=conversation_context,
#                         query_type="document_query"
#                     )

#                     blocks = _build_blocks(full_response_text, images)
#                     total_time = round(time.time() - start_time, 2)
#                     done_message = {
#                         'type': 'done',
#                         'sources': formatted_sources,
#                         'processing_time': total_time,
#                         'images': images,
#                         'blocks': blocks
#                     }
#                     yield f"data: {json.dumps(done_message)}\n\n"

#                 except Exception as e:
#                     logger.error(f"Streaming error: {e}\n{traceback.format_exc()}")
#                     yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"

#             headers = {
#                 'Cache-Control': 'no-cache, no-transform',
#                 'Connection': 'keep-alive',
#                 'Content-Type': 'text/event-stream',
#                 'X-Accel-Buffering': 'no'
#             }
#             return Response(stream_with_context(generate_stream()), headers=headers)

#         # ---------- Non-streaming ----------
#         start_time = time.time()

#         search_docs_list = list(doc_manager.loaded_documents.keys())
#         if selected_documents:
#             search_docs_list = [d for d in search_docs_list if d in selected_documents]

#         mentioned = find_mentioned_document(user_message, search_docs_list)
#         if mentioned:
#             search_docs_list = [mentioned]

#         if not search_docs_list:
#             msg = "Please select at least one document to search."
#             return jsonify({
#                 'response': msg,
#                 'query_type': 'error',
#                 'sources': [],
#                 'processing_time': 0.0,
#                 'images': [],
#                 'blocks': [{"type": "text", "content": msg}]
#             })

#         followup_force_context = detect_follow_up_query(user_message)
#         conversation_context = context_manager.get_enhanced_context_for_query(user_message, force=followup_force_context)

#         is_fast = len(user_message.split()) <= 6
#         CONTEXT_K_NONSTREAM = 15
#         search_results, _ = intelligent_document_search(
#             user_message,
#             doc_manager,
#             k=CONTEXT_K_NONSTREAM,
#             fast=is_fast,
#             selected_documents=search_docs_list,
#             context_manager=context_manager
#         )

#         if not search_results:
#             msg = "No relevant information found in the loaded documents."
#             return jsonify({
#                 'response': msg,
#                 'query_type': 'general',
#                 'sources': [],
#                 'processing_time': 0.0,
#                 'images': [],
#                 'blocks': [{"type": "text", "content": msg}]
#             })

#         # 🔸 High-confidence filter (>= 70% match)
#         high_conf_results = filter_high_confidence_results(search_results, threshold=0.7)
#         effective_results = high_conf_results or search_results

#         combined_context = build_context_from_results(effective_results, 8, is_fast)
#         bot_name = context_manager.get_memory('bot_name', 'Catapult')
#         prompt = generate_prompt_based_on_mode(
#             search_mode, bot_name, user_message,
#             combined_context, conversation_context, followup_force_context
#         )
#         response = llm.invoke(prompt)
#         full_response_text = response if isinstance(response, str) else str(response)

#         # Final sanitize for non-stream
#         full_response_text = _strip_external_image_links(full_response_text)

#         # pick curated local images ONLY from high-confidence/effective results
#         query_analysis = analyze_query_for_page_number(user_message)
#         candidate_pages = set([query_analysis['page_number']]) if query_analysis["is_page_specific"] else set()
#         images = _select_ordered_images(user_message, effective_results, limit=6, candidate_pages=candidate_pages)

#         sources = effective_results[:8]
#         formatted_sources = format_sources_for_response(sources, 8)
#         processing_time = time.time() - start_time

#         context_manager.add_exchange(
#             question=user_message,
#             answer=full_response_text,
#             sources=sources,
#             context_used=conversation_context,
#             query_type="document_query"
#         )

#         blocks = _build_blocks(full_response_text, images)

#         return jsonify({
#             'response': full_response_text,
#             'query_type': "document_query",
#             'sources': formatted_sources,
#             'processing_time': round(processing_time, 2),
#             'images': images,
#             'blocks': blocks
#         })

#     except Exception as e:
#         logger.error(f"Chat API error: {e}\n{traceback.format_exc()}")
#         return jsonify({'error': 'Internal server error', 'message': str(e)}), 500

import json
import time
import os
import re
import traceback
import logging
from pathlib import Path
from urllib.parse import urljoin  # ✅ needed for _embed_url

from flask import Blueprint, jsonify, request, session, Response, stream_with_context, render_template

from modules.conversation import detect_casual_conversation, generate_casual_response
from modules.search import intelligent_document_search, analyze_query_for_page_number
from modules.response_generator import (
    generate_prompt_based_on_mode,
    build_context_from_results,
    format_sources_for_response,
    detect_follow_up_query
)
from modules.utils import (
    find_mentioned_document,
    format_loaded_documents_response,
    is_query_about_documents,
    is_referential_query
)
from modules.auth import require_login
from app_state import get_system_state
from config import PAGE_SPECIFIC_TEMPLATE, config as app_config

chat_bp = Blueprint('chat', __name__, url_prefix='/')
logger = logging.getLogger(__name__)

# ==================================
# Global threshold for "strong" hits
# ==================================
DOC_RELEVANCE_THRESHOLD = 0.7


def get_managers_and_llm():
    state_data = get_system_state()
    system_initialized, initialization_error, doc_manager, context_manager, _, _, llm = state_data
    if not system_initialized:
        return None, None, None, initialization_error or "Unknown initialization error"
    return doc_manager, context_manager, llm, None


def _friendly_llm_error(exc: Exception) -> str:
    """Turn Ollama/LangChain errors into actionable messages for the UI."""
    msg = str(exc)
    lower = msg.lower()
    if "system memory" in lower or "requires more" in lower and "memory" in lower:
        return (
            "Ollama ran out of RAM while generating a reply. "
            "Close other apps, restart Ollama (`ollama stop` then `ollama serve`), "
            "or use a smaller model (e.g. set environment variable LLM_MODEL=phi3:mini "
            "and run `ollama pull phi3:mini`)."
        )
    if "connection" in lower or "refused" in lower:
        return (
            "Cannot reach Ollama. Start it with `ollama serve` and ensure "
            f"the model `{app_config.LLM_MODEL}` is pulled (`ollama pull {app_config.LLM_MODEL}`)."
        )
    return msg


def _split_by_relevance(results, threshold: float = DOC_RELEVANCE_THRESHOLD):
    """
    Split results into:
    - high_relevance: relevance_score >= threshold
    - low_relevance:  relevance_score <  threshold
    """
    high, low = [], []
    for d in results:
        score = d.metadata.get("relevance_score", 0.0)
        if score >= threshold:
            high.append(d)
        else:
            low.append(d)
    return high, low


# =========================
# Image URL + scoring utils
# =========================

def _embed_url(doc_name: str, rel_path: str) -> str:
    """
    Build a fully-qualified URL for an embedding asset using the current request host.
    Example: http://127.0.0.1:9072/embedding/<doc_name>/<rel_path>
    """
    # Normalize the relative path
    rel_path = rel_path.lstrip('/\\')
    if not rel_path.lower().startswith(('images/', 'tables/')):
        rel_path = f"images/{rel_path}"

    # Build a path relative to the API root, then join with request.host_url to add scheme+host+port
    rel = f"embedding/{doc_name}/{rel_path}"
    return urljoin(request.host_url, rel)


def _normalize_tokens(text: str) -> set:
    text = (text or "").lower()
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    toks = [t for t in text.split() if len(t) > 1]
    return set(toks)


def _clean_caption_text(text: str | None) -> str:
    """
    Remove markdown image tags and image-ish/kbase URLs from captions.
    Falls back to trimmed plain text.
    """
    if not text:
        return ""
    return _strip_external_image_links(text)


def _text_fields_from_image_meta(m: dict) -> str:
    fields = []
    for k in (
        "caption", "title", "alt", "ocr_text", "context_text", "text", "description",
        "filename", "relative_path", "image_relative_path", "source_url"
    ):
        v = m.get(k)
        if isinstance(v, str) and v.strip():
            fields.append(v)
    return " ".join(fields)


def _image_type_hint_score(query_tokens: set, meta_text: str) -> float:
    score = 0.0
    mt = (meta_text or "").lower()
    if any(x in query_tokens for x in {"table", "tabular", "excel"}):
        if "table" in mt:
            score += 1.0
    if any(x in query_tokens for x in {"screenshot", "login", "ui", "screen"}):
        if any(w in mt for w in ["screenshot", "screen", "ui", "login"]):
            score += 0.8
    return score


def _score_image_meta(m: dict, query_tokens: set) -> float:
    """
    Relative scoring *within* the already context-filtered images.
    Docs and pages are filtered earlier, so here we just rank by semantic match.
    """
    score = 0.0
    meta_text = _text_fields_from_image_meta(m)
    meta_tokens = _normalize_tokens(meta_text)
    if meta_tokens:
        overlap = query_tokens.intersection(meta_tokens)
        score += min(len(overlap), 6) * 1.0
    score += _image_type_hint_score(query_tokens, meta_text)
    return score


def _derive_rel_from_abs(abs_path: str, doc_name: str) -> str | None:
    try:
        base = os.path.normpath(os.path.join(app_config.EMBEDDINGS_DIR, doc_name))
        ap = os.path.normpath(abs_path)
        if ap.startswith(base + os.sep):
            return os.path.relpath(ap, base).replace("\\", "/")
    except Exception:
        pass
    return None


# ---------- images manifest resolution (handles renamed image paths) ----------
_IMAGES_MANIFEST_CACHE: dict[str, dict[str, str]] = {}  # {doc_name: {filename -> relative_path}}

#  NEW: full metadata cache from images_metadata.json
_IMAGES_FULLMETA_CACHE: dict[str, list[dict]] = {}  # {doc_name: [full meta dicts]}


def _load_images_manifest_for_doc(doc_name: str) -> dict[str, str]:
    """
    Load and cache the mapping of filename -> relative_path from images_metadata.json
    so we can find renamed copies like foo.png -> images/foo-1.png.
    """
    if doc_name in _IMAGES_MANIFEST_CACHE:
        return _IMAGES_MANIFEST_CACHE[doc_name]

    manifest: dict[str, str] = {}
    try:
        manifest_file = Path(app_config.EMBEDDINGS_DIR) / doc_name / "images" / "images_metadata.json"
        if manifest_file.exists():
            data = json.loads(manifest_file.read_text(encoding="utf-8"))
            if isinstance(data, list):
                for item in data:
                    fn = item.get("filename") or os.path.basename(item.get("relative_path", "") or "")
                    rp = item.get("relative_path")
                    if fn and rp:
                        manifest[fn] = rp
    except Exception as e:
        logger.warning(f"Failed to load images manifest for {doc_name}: {e}")

    _IMAGES_MANIFEST_CACHE[doc_name] = manifest
    return manifest


def _resolve_rel_via_manifest(doc_name: str, rel_or_filename: str | None) -> str | None:
    """
    Given either a relative path or filename, resolve to the *actual* copied
    path as per images_metadata.json. Falls back to checking existence in /images/.
    """
    if not rel_or_filename:
        return None

    key = rel_or_filename.lstrip("/\\")
    if key.lower().startswith("images/"):
        key = key[len("images/"):]

    manifest = _load_images_manifest_for_doc(doc_name)

    # Direct match
    if rel_or_filename in manifest.values():
        return rel_or_filename

    # Filename match
    if key in manifest:
        return manifest[key]

    # Fallback: check physical existence
    candidate = Path(app_config.EMBEDDINGS_DIR) / doc_name / "images" / key
    if candidate.exists():
        return f"images/{key}"

    return None


def _collect_image_candidates(
    results,
    max_docs: int = 30,
    allowed_docs: set[str] | None = None,
    allowed_pages: set[int] | None = None
):
    """
    Collect image chunks from `results`, but optionally restrict:
      - to specific documents (allowed_docs)
      - to specific pages (allowed_pages)

    This is the main place we enforce: "only images from the same docs & pages as the context".
    """
    cand = []
    for d in results[:max_docs]:
        meta = d.metadata or {}
        if meta.get('type') != 'image':
            continue

        doc_name = (
            meta.get('source_document')
            or meta.get('document')
            or meta.get('document_name')
            or meta.get('doc_name')
        )

        if not doc_name:
            continue

        if allowed_docs and doc_name not in allowed_docs:
            continue

        # Detect page number
        page_val = None
        for k in ("display_page", "page", "page_number", "page_num"):
            v = meta.get(k)
            if isinstance(v, int):
                page_val = v
                break
            if isinstance(v, str) and v.isdigit():
                page_val = int(v)
                break

        if allowed_pages and (page_val is None or page_val not in allowed_pages):
            # Skip images from pages not part of the current text context
            continue

        # Prefer explicit relative path; else derive from abs path; else filename
        rel = meta.get('image_relative_path') or meta.get('relative_path')
        if not rel:
            abs_img = meta.get('image_path') or meta.get('path')
            if abs_img and doc_name:
                rel = _derive_rel_from_abs(abs_img, doc_name)

        filename = meta.get('image_filename') or meta.get('filename')

        # Resolve to actual copied path
        rel_fixed = _resolve_rel_via_manifest(doc_name, rel or filename)
        if not (doc_name and rel_fixed):
            continue

        url = _embed_url(doc_name, rel_fixed)

        cand.append({
            "doc_name": doc_name,
            "relative_path": rel_fixed,
            "display_page": page_val if page_val is not None else meta.get('display_page', 1),
            "md_index": meta.get('md_index'),
            "alt": meta.get('alt') or "",
            "context_text": _clean_caption_text(meta.get('context_text')),
            "ocr_text": meta.get('ocr_text') or "",
            "source_url": meta.get('source_url'),
            "filename": filename or os.path.basename(rel_fixed),
            "__meta": meta,
            "url": url
        })
    return cand


# def _select_ordered_images(
#     user_message: str,
#     all_results,
#     limit: int = 6,
#     context_results=None
# ):
#     """
#     Select images that belong to the SAME docs & pages as the final context,
#     with a sensible fallback:

#       1. Prefer images from the same docs AND pages as the text context.
#       2. If that yields < 2 images, fall back to same docs (any page).
#       3. Rank all candidate images against the query text and return up to `limit`.
#     """
#     if not context_results:
#         return []

#     # 1) Determine which documents & pages are in the context
#     context_docs: set[str] = set()
#     context_pages: set[int] = set()

#     for d in context_results:
#         meta = d.metadata or {}
#         doc_name = (
#             meta.get('source_document')
#             or meta.get('document')
#             or meta.get('document_name')
#             or meta.get('doc_name')
#         )
#         if doc_name:
#             context_docs.add(doc_name)

#         page_val = None
#         for k in ("display_page", "page", "page_number", "page_num", "display_page_num"):
#             v = meta.get(k)
#             if isinstance(v, int):
#                 page_val = v
#                 break
#             if isinstance(v, str) and v.isdigit():
#                 page_val = int(v)
#                 break
#         if page_val is not None:
#             context_pages.add(page_val)

#     if not context_docs:
#         return []

#     # 2) First pass: images from SAME docs & SAME pages as context
#     cand = _collect_image_candidates(
#         all_results,
#         max_docs=200,  # look deeper into results so we don't miss images
#         allowed_docs=context_docs,
#         allowed_pages=context_pages if context_pages else None
#     )

#     # 3) Fallback: if we got < 2 images, relax page restriction
#     if len(cand) < 2:
#         cand = _collect_image_candidates(
#             all_results,
#             max_docs=200,
#             allowed_docs=context_docs,
#             allowed_pages=None  # any page in the same docs
#         )

#     if not cand:
#         return []

#     # 4) Rank them semantically against the query (within already-filtered set)
#     qtokens = _normalize_tokens(user_message)
#     scored = []
#     for c in cand:
#         score = _score_image_meta(c["__meta"], qtokens)
#         scored.append((score, c))

#     def _key(item):
#         score, c = item
#         mdi = c["md_index"]
#         mdi_key = 10**9 if mdi is None else mdi
#         return (-score, c["doc_name"], c["display_page"], mdi_key)

#     scored.sort(key=_key)

#     # 5) Output up to `limit`, stripping internal __meta from payload
#     out = []
#     for _, c in scored[:limit]:
#         out.append({
#             "url": c["url"],
#             "doc_name": c["doc_name"],
#             "relative_path": c["relative_path"],
#             "display_page": c["display_page"],
#             "md_index": c["md_index"],
#             "alt": c["alt"] or "",
#             "context_text": c["context_text"] or "",
#             "filename": c["filename"]
#         })
#     return out

def _load_images_fullmeta_for_doc(doc_name: str) -> list[dict]:
    """
    Load and cache the full list of image metadata dicts for a document
    from images_metadata.json.

    Each item typically has:
      - filename
      - relative_path
      - display_page / page
      - caption / context_text / ocr_text
      - etc.
    """
    if doc_name in _IMAGES_FULLMETA_CACHE:
        return _IMAGES_FULLMETA_CACHE[doc_name]

    meta_list: list[dict] = []
    try:
        manifest_file = Path(app_config.EMBEDDINGS_DIR) / doc_name / "images" / "images_metadata.json"
        if manifest_file.exists():
            data = json.loads(manifest_file.read_text(encoding="utf-8"))
            if isinstance(data, list):
                meta_list = [m for m in data if isinstance(m, dict)]
    except Exception as e:
        logger.warning(f"Failed to load full image metadata for {doc_name}: {e}")

    _IMAGES_FULLMETA_CACHE[doc_name] = meta_list
    return meta_list

def _select_ordered_images(
    user_message: str,
    all_results,
    limit: int = 6,
    context_results=None
):
    """
    Select images based on the FINAL text context:

    1. Determine which docs + pages are in `context_results`.
    2. Load ALL images for those docs from images_metadata.json.
    3. Prefer images on the same pages as the context.
    4. If none found for those pages, fall back to any page in those docs.
    5. Rank by semantic match against the user query, then return up to `limit`.

    NOTE: We deliberately do NOT rely only on image chunks appearing
    in `all_results` because that would cap us to a single image even
    when a page actually has multiple images.
    """
    if not context_results:
        return []

    # 1) Determine which documents & pages belong to the final context
    context_docs: set[str] = set()
    context_pages: set[int] = set()

    for d in context_results:
        meta = d.metadata or {}
        doc_name = (
            meta.get('source_document')
            or meta.get('document')
            or meta.get('document_name')
            or meta.get('doc_name')
        )
        if doc_name:
            context_docs.add(doc_name)

        page_val = None
        for k in ("display_page", "page", "page_number", "page_num", "display_page_num"):
            v = meta.get(k)
            if isinstance(v, int):
                page_val = v
                break
            if isinstance(v, str) and v.isdigit():
                page_val = int(v)
                break
        if page_val is not None:
            context_pages.add(page_val)

    if not context_docs:
        return []

    # 2) Collect candidate images from images_metadata.json
    def _collect_from_docs(allowed_pages: set[int] | None):
        cand_local = []
        for doc_name in context_docs:
            fullmeta = _load_images_fullmeta_for_doc(doc_name)
            if not fullmeta:
                continue

            for m in fullmeta:
                # figure out which page this image belongs to
                page_val = None
                for k in ("display_page", "page", "page_number", "page_num", "display_page_num"):
                    v = m.get(k)
                    if isinstance(v, int):
                        page_val = v
                        break
                    if isinstance(v, str) and v.isdigit():
                        page_val = int(v)
                        break

                # page filter if provided
                if allowed_pages and page_val is not None and page_val not in allowed_pages:
                    continue

                rel = m.get("relative_path") or m.get("image_relative_path")
                filename = m.get("filename")
                if not rel and filename:
                    # best-effort fallback
                    rel = f"images/{filename}"
                if not rel:
                    continue

                url = _embed_url(doc_name, rel)

                cand_local.append({
                    "doc_name": doc_name,
                    "relative_path": rel,
                    "display_page": page_val if page_val is not None else 1,
                    "md_index": m.get("md_index"),
                    "alt": m.get("alt") or "",
                    "context_text": _clean_caption_text(
                        m.get("context_text") or m.get("caption") or m.get("description")
                    ),
                    "ocr_text": m.get("ocr_text") or "",
                    "source_url": m.get("source_url"),
                    "filename": filename or os.path.basename(rel),
                    "__meta": m,
                    "url": url,
                })
        return cand_local

    # First pass: same docs + same pages as context (if we know the pages)
    candidates = []
    if context_pages:
        candidates = _collect_from_docs(context_pages)

    # Fallback: if that yields nothing, relax page restriction to any page in those docs
    if not candidates:
        candidates = _collect_from_docs(None)

    if not candidates:
        return []

    # 3) Rank by semantic similarity against the query
    qtokens = _normalize_tokens(user_message)
    scored = []
    for c in candidates:
        score = _score_image_meta(c["__meta"], qtokens)
        scored.append((score, c))

    def _key(item):
        score, c = item
        mdi = c["md_index"]
        mdi_key = 10**9 if mdi is None else mdi
        return (-score, c["doc_name"], c["display_page"], mdi_key)

    scored.sort(key=_key)

    # 4) Trim to `limit` and strip internal __meta
    out = []
    for _, c in scored[:limit]:
        out.append({
            "url": c["url"],
            "doc_name": c["doc_name"],
            "relative_path": c["relative_path"],
            "display_page": c["display_page"],
            "md_index": c["md_index"],
            "alt": c["alt"] or "",
            "context_text": c["context_text"] or "",
            "filename": c["filename"],
        })
    return out




def _build_blocks(response_text: str, images: list[dict]):
    paras = [p.strip() for p in re.split(r'\n\s*\n+', response_text or "") if p.strip()]
    blocks = []
    img_idx = 0
    for p in paras:
        blocks.append({"type": "text", "content": p})
        if img_idx < len(images):
            blocks.append({"type": "image", **images[img_idx]})
            img_idx += 1
    while img_idx < len(images):
        blocks.append({"type": "image", **images[img_idx]})
        img_idx += 1
    return blocks


# =======================
# Streaming sanitization
# =======================

_MD_IMG_RE = re.compile(r'!\[[^\]]*\]\([^)]+\)')
_MD_LINK_KBASE_RE = re.compile(
    r'\[[^\]]*\]\((?P<url>https?://[^\s)]+(?:download/attachments|/attachments/)[^\s)]*)\)',
    re.IGNORECASE
)
_ANY_IMGISH_URL_RE = re.compile(
    r'(https?://[^\s)]+(?:\.(?:png|jpe?g|gif|webp|bmp|svg)(?:\?[^\s)]*)?|\?api=v2|download/attachments/[^\s)]*))',
    re.IGNORECASE
)


def _strip_external_image_links(text: str) -> str:
    if not text:
        return text
    text = _MD_IMG_RE.sub('', text)
    text = _MD_LINK_KBASE_RE.sub('', text)
    text = _ANY_IMGISH_URL_RE.sub('', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()


def _incremental_sanitizer():
    raw_so_far = ""
    clean_so_far = ""

    IMG_START_RE = re.compile(r'!\[[^\]]*?\]\([^\)]*?$')
    LINK_START_RE = re.compile(r'\[[^\]]*?\]\([^\)]*?$')

    def feed(chunk: str) -> str:
        nonlocal raw_so_far, clean_so_far
        if not chunk:
            return ""
        raw_so_far += str(chunk)

        sanitized_full = _strip_external_image_links(raw_so_far)
        tentative_delta = sanitized_full[len(clean_so_far):]

        tail = sanitized_full
        holdback_from = None

        m1 = IMG_START_RE.search(raw_so_far)
        m2 = LINK_START_RE.search(raw_so_far)
        for m in (m1, m2):
            if m:
                start_idx_in_raw = m.start()
                emitted_len_est = len(clean_so_far)
                if start_idx_in_raw >= emitted_len_est:
                    holdback_from = emitted_len_est
                    break

        if holdback_from is not None:
            tentative_delta = ""

        if tentative_delta:
            clean_so_far += tentative_delta
            return tentative_delta
        return ""

    return feed


# ==============
# Chat API
# ==============

@chat_bp.route('/api/chat', methods=['POST'])
@require_login
def api_chat():
    doc_manager, context_manager, llm, error = get_managers_and_llm()
    if error:
        return jsonify({'error': 'System not initialized', 'message': error}), 500
    # Allow requests without LLM - can still do document search
    # if llm is None:
    #     return jsonify({'error': 'System not initialized', 'message': 'Language model (LLM) is not available.'}), 500

    try:
        data = request.get_json()
        if not data or 'message' not in data:
            return jsonify({'error': 'No message provided'}), 400

        user_message = data['message'].strip()
        if not user_message:
            return jsonify({'error': 'Empty message'}), 400

        selected_documents = data.get('loaded_documents', [])
        search_mode = data.get('search_mode', 'documents_only')
        lower_msg = user_message.lower()

        # 1) casual
        conversation_response = detect_casual_conversation(user_message, selected_documents)
        if conversation_response:
            return jsonify({
                'response': conversation_response,
                'query_type': 'casual_conversation',
                'sources': [],
                'processing_time': 0.0,
                'images': [],
                'blocks': [{"type": "text", "content": conversation_response}]
            })

        # 2) commands
        if lower_msg in ['clear', 'reset']:
            context_manager.clear_history()
            if hasattr(doc_manager, 'search_cache'):
                doc_manager.search_cache.clear()
            msg = 'Conversation history cleared.'
            return jsonify({
                'response': msg,
                'query_type': 'command',
                'sources': [],
                'processing_time': 0.0,
                'images': [],
                'blocks': [{"type": "text", "content": msg}]
            })

        # 3) memory set
        if lower_msg.startswith('remember '):
            content = user_message[len('remember '):].strip()
            remembered = False
            if ' is ' in content:
                parts = content.split(' is ', 1)
                key = parts[0].replace('that ', '').strip(' :.-').strip()
                value = parts[1].strip(' .').strip()
                if key and value:
                    context_manager.add_fact(key, value)
                    remembered = True
            elif '=' in content:
                parts = content.split('=', 1)
                key = parts[0].replace('that ', '').strip(' :.-').strip()
                value = parts[1].strip(' .').strip()
                if key and value:
                    context_manager.add_fact(key, value)
                    remembered = True
            else:
                if 'notes' not in context_manager.user_memory:
                    context_manager.user_memory['notes'] = []
                context_manager.user_memory['notes'].append(content)
                remembered = True
            msg = 'Got it, I will remember that.' if remembered else 'Sorry, I could not understand what to remember.'
            return jsonify({
                'response': msg,
                'query_type': 'memory_update',
                'sources': [],
                'processing_time': 0.0,
                'images': [],
                'blocks': [{"type": "text", "content": msg}]
            })

        # 4) quick fact lookup
        if (lower_msg.startswith('what is ') or lower_msg.startswith('who is ') or lower_msg.startswith('tell me ')) and len(lower_msg.split()) <= 8:
            key = user_message
            for prefix in ['what is ', 'who is ', 'tell me about ', 'tell me the ', 'tell me ']:
                if lower_msg.startswith(prefix):
                    key = user_message[len(prefix):].strip(' ?!.')
                    break
            fact = context_manager.get_fact(key)
            if fact:
                ans = f"{key.capitalize()}: {fact}"
                return jsonify({
                    'response': ans,
                    'query_type': 'memory_lookup',
                    'sources': [],
                    'processing_time': 0.0,
                    'images': [],
                    'blocks': [{"type": "text", "content": ans}]
                })

        # 5) check facts before search
        stored_fact = context_manager.check_facts_before_search(user_message)
        if stored_fact:
            return jsonify({
                'response': stored_fact,
                'query_type': 'fact_lookup',
                'sources': [],
                'processing_time': 0.0,
                'images': [],
                'blocks': [{"type": "text", "content": stored_fact}]
            })

        # 6) doc listing/count
        count_triggers = ['how many docs', 'how many documents', 'number of docs', 'docs count',
                          'how many files', 'files count', 'number of files']
        if is_query_about_documents(user_message):
            if any(t in lower_msg for t in count_triggers):
                names = sorted(list(doc_manager.loaded_documents.keys())) if doc_manager and doc_manager.loaded_documents else []
                count = len(names)
                summary = f"Loaded documents: {count}" + (f" ({', '.join(names)})" if count > 0 else "")
            else:
                summary = format_loaded_documents_response(doc_manager, selected_documents)
            return jsonify({
                'response': summary,
                'query_type': 'system_info',
                'sources': [],
                'processing_time': 0.0,
                'images': [],
                'blocks': [{"type": "text", "content": summary}]
            })

        # 7) require docs
        if not doc_manager.loaded_documents:
            return jsonify({'error': 'No documents loaded', 'message': 'Please load at least one document before asking questions.'}), 400

        # 8) streaming?
        stream_requested = data.get('stream', False)
        if isinstance(stream_requested, str):
            stream_requested = stream_requested.lower() in ['true', '1', 'yes']

        if stream_requested:
            def generate_stream():
                start_time = time.time()
                try:
                    search_docs_list = list(doc_manager.loaded_documents.keys())
                    if selected_documents:
                        search_docs_list = [d for d in search_docs_list if d in selected_documents]

                    mentioned = find_mentioned_document(user_message, search_docs_list)
                    if mentioned:
                        search_docs_list = [mentioned]

                    if not search_docs_list:
                        yield f"data: {json.dumps({'type': 'token', 'content': 'Please select at least one document to search.'})}\n\n"
                        yield f"data: {json.dumps({'type': 'done', 'sources': [], 'processing_time': 0.0, 'images': [], 'blocks': []})}\n\n"
                        return

                    yield ": ping\n\n"
                    yield f"data: {json.dumps({'type': 'status', 'message': 'searching across documents'})}\n\n"

                    followup_force_context = detect_follow_up_query(user_message)
                    conversation_context = context_manager.get_enhanced_context_for_query(
                        user_message,
                        force=followup_force_context
                    )

                    is_fast = len(user_message.split()) <= 6
                    CONTEXT_K = 15

                    search_results, _ = intelligent_document_search(
                        user_message,
                        doc_manager,
                        k=CONTEXT_K,
                        fast=is_fast,
                        selected_documents=search_docs_list,
                        context_manager=context_manager
                    )

                    if not search_results:
                        msg = "No relevant information found in the loaded documents."
                        yield f"data: {json.dumps({'type': 'token', 'content': msg})}\n\n"
                        yield f"data: {json.dumps({'type': 'done', 'sources': [], 'processing_time': 0.0, 'images': [], 'blocks': []})}\n\n"
                        return

                    # 🔹 Use only high-relevance results as the "primary" context
                    primary_results, secondary_results = _split_by_relevance(search_results)
                    if not primary_results:
                        primary_results = search_results

                    # Page hint
                    query_analysis = analyze_query_for_page_number(user_message)
                    if query_analysis["is_page_specific"]:
                        page_num = query_analysis['page_number']
                        filtered = [
                            d for d in primary_results
                            if d.metadata.get('display_page') == page_num
                        ]
                        if filtered:
                            primary_results = filtered

                    # Build context from primary results only
                    combined_context = build_context_from_results(primary_results, 8, is_fast)
                    bot_name = context_manager.get_memory('bot_name', 'Catapult')

                    if query_analysis["is_page_specific"]:
                        prompt = PAGE_SPECIFIC_TEMPLATE.format(
                            conversation_context=conversation_context,
                            context=combined_context,
                            question=user_message
                        )
                    else:
                        prompt = generate_prompt_based_on_mode(
                            search_mode,
                            bot_name,
                            user_message,
                            combined_context,
                            conversation_context,
                            followup_force_context
                        )

                    # 🔹 Images: only from docs & pages present in primary_results
                    images = _select_ordered_images(
                        user_message=user_message,
                        all_results=search_results,
                        limit=6,
                        context_results=primary_results
                    )

                    # Send images meta first
                    yield f"data: {json.dumps({'type': 'images', 'images': images})}\n\n"

                    # Stream + sanitize
                    sanitizer = _incremental_sanitizer()
                    full_response_text = ""
                    for chunk in llm.stream(prompt):
                        safe_delta = sanitizer(str(chunk))
                        if not safe_delta:
                            continue
                        full_response_text += safe_delta
                        yield f"data: {json.dumps({'type': 'token', 'content': safe_delta})}\n\n"

                    formatted_sources = format_sources_for_response(primary_results, 8)

                    context_manager.add_exchange(
                        question=user_message,
                        answer=full_response_text,
                        sources=primary_results[:8],
                        context_used=conversation_context,
                        query_type="document_query"
                    )

                    blocks = _build_blocks(full_response_text, images)
                    total_time = round(time.time() - start_time, 2)
                    done_message = {
                        'type': 'done',
                        'sources': formatted_sources,
                        'processing_time': total_time,
                        'images': images,
                        'blocks': blocks
                    }
                    yield f"data: {json.dumps(done_message)}\n\n"

                except Exception as e:
                    logger.error(f"Streaming error: {e}\n{traceback.format_exc()}")
                    yield f"data: {json.dumps({'type': 'error', 'message': _friendly_llm_error(e)})}\n\n"
                    yield f"data: {json.dumps({'type': 'done', 'sources': [], 'processing_time': 0.0, 'images': [], 'blocks': []})}\n\n"

            headers = {
                'Cache-Control': 'no-cache, no-transform',
                'Connection': 'keep-alive',
                'Content-Type': 'text/event-stream',
                'X-Accel-Buffering': 'no'
            }
            return Response(stream_with_context(generate_stream()), headers=headers)

        # ---------- Non-streaming ----------
        start_time = time.time()

        search_docs_list = list(doc_manager.loaded_documents.keys())
        if selected_documents:
            search_docs_list = [d for d in search_docs_list if d in selected_documents]

        mentioned = find_mentioned_document(user_message, search_docs_list)
        if mentioned:
            search_docs_list = [mentioned]

        if not search_docs_list:
            msg = "Please select at least one document to search."
            return jsonify({
                'response': msg,
                'query_type': 'error',
                'sources': [],
                'processing_time': 0.0,
                'images': [],
                'blocks': [{"type": "text", "content": msg}]
            })

        followup_force_context = detect_follow_up_query(user_message)
        conversation_context = context_manager.get_enhanced_context_for_query(
            user_message,
            force=followup_force_context
        )

        is_fast = len(user_message.split()) <= 6
        CONTEXT_K_NONSTREAM = 15
        search_results, _ = intelligent_document_search(
            user_message,
            doc_manager,
            k=CONTEXT_K_NONSTREAM,
            fast=is_fast,
            selected_documents=search_docs_list,
            context_manager=context_manager
        )

        if not search_results:
            msg = "No relevant information found in the loaded documents."
            return jsonify({
                'response': msg,
                'query_type': 'general',
                'sources': [],
                'processing_time': 0.0,
                'images': [],
                'blocks': [{"type": "text", "content": msg}]
            })

        # 🔹 High-relevance subset for non-stream too
        primary_results, secondary_results = _split_by_relevance(search_results)
        if not primary_results:
            primary_results = search_results

        # Page-specific filtering
        query_analysis = analyze_query_for_page_number(user_message)
        if query_analysis["is_page_specific"]:
            page_num = query_analysis['page_number']
            filtered = [
                d for d in primary_results
                if d.metadata.get('display_page') == page_num
            ]
            if filtered:
                primary_results = filtered

        combined_context = build_context_from_results(primary_results, 8, is_fast)
        bot_name = context_manager.get_memory('bot_name', 'Catapult')
        prompt = generate_prompt_based_on_mode(
            search_mode,
            bot_name,
            user_message,
            combined_context,
            conversation_context,
            followup_force_context
        )
        response = llm.invoke(prompt)
        full_response_text = response if isinstance(response, str) else str(response)

        # Final sanitize for non-stream
        full_response_text = _strip_external_image_links(full_response_text)

        # 🔹 Context-aware images
        images = _select_ordered_images(
            user_message=user_message,
            all_results=search_results,
            limit=6,
            context_results=primary_results
        )

        sources = primary_results[:8]
        formatted_sources = format_sources_for_response(sources, 8)
        processing_time = time.time() - start_time

        context_manager.add_exchange(
            question=user_message,
            answer=full_response_text,
            sources=sources,
            context_used=conversation_context,
            query_type="document_query"
        )

        blocks = _build_blocks(full_response_text, images)

        return jsonify({
            'response': full_response_text,
            'query_type': "document_query",
            'sources': formatted_sources,
            'processing_time': round(processing_time, 2),
            'images': images,
            'blocks': blocks
        })

    except Exception as e:
        logger.error(f"Chat API error: {e}\n{traceback.format_exc()}")
        return jsonify({'error': 'Internal server error', 'message': str(e)}), 500
