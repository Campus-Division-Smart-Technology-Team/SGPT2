# def perform_federated_search(query: str, top_k: int) -> Tuple[List[Dict[str, Any]], str, str, bool]:
#     """
#     Perform federated search with building-aware prioritization.

#     Returns: (results, answer, publication_date_info, score_too_low)
#     """
#     # Extract building name from query if present
#     target_building = extract_building_from_query(query)

#     if target_building:
#         logging.info("🏢 Detected building: %s", target_building)
#         st.info(
#             f"🏢 Detected building: **{target_building}** - searching for all related documents")

#     all_hits: List[Dict[str, Any]] = []

#     # Search across all target indexes
#     for idx_name in TARGET_INDEXES:
#         # If we have a target building, enhance the query
#         if target_building:
#             enhanced_query = f"{query} {target_building}"
#             building_hits = search_one_index(
#                 idx_name, enhanced_query, top_k * 2, embed_model=None)
#             all_hits.extend(building_hits)

#         # Also do a standard search
#         general_hits = search_one_index(
#             idx_name, query, top_k, embed_model=None)
#         all_hits.extend(general_hits)

#     # Remove duplicates based on ID
#     seen_ids = set()
#     unique_hits = []
#     for hit in all_hits:
#         hit_id = hit.get('id')
#         if hit_id and hit_id not in seen_ids:
#             seen_ids.add(hit_id)
#             unique_hits.append(hit)

#     logging.info("Total unique hits before filtering: %d", len(unique_hits))

#     # DEBUG: Log document types found
#     doc_type_counts = {}
#     for hit in unique_hits:
#         doc_type = hit.get('document_type', 'unknown')
#         doc_type_counts[doc_type] = doc_type_counts.get(doc_type, 0) + 1
#     logging.info("Document types found: %s", doc_type_counts)

#     # DEBUG: Log top 5 results before filtering
#     logging.info("Top 5 results before filtering:")
#     for i, hit in enumerate(unique_hits[:5]):
#         logging.info(
#             "  %d. building_name='%s', doc_type='%s', score=%.3f, key='%s'",
#             i+1,
#             hit.get('building_name'),
#             hit.get('document_type'),
#             hit.get('score', 0),
#             hit.get('key', '')[:50]
#         )
#     # If we have a target building, filter and prioritise
#     if target_building:
#         building_specific_hits = filter_results_by_building(
#             unique_hits, target_building)

#         logging.info("Hits matching '%s': %d", target_building,
#                      len(building_specific_hits))

#         # DEBUG: Log document types in filtered results
#         filtered_doc_type_counts = {}
#         for hit in building_specific_hits:
#             doc_type = hit.get('document_type', 'unknown')
#             filtered_doc_type_counts[doc_type] = filtered_doc_type_counts.get(
#                 doc_type, 0) + 1
#         logging.info("Filtered document types: %s", filtered_doc_type_counts)

#         # Log what we found
#         logging.info("Top results after filtering:")
#         for i, hit in enumerate(building_specific_hits[:5]):
#             logging.info(
#                 "  %d. building_name='%s', doc_type='%s', score=%.3f",
#                 i+1,
#                 hit.get('building_name'),
#                 hit.get('document_type'),
#                 hit.get('score', 0))
#         # If we found any matches, use them; otherwise, fall back to prioritised full list
#         if building_specific_hits:
#             unique_hits = building_specific_hits
#         else:
#             logging.warning(
#                 "No exact matches for '%s', using all results", target_building)
#             unique_hits = prioritise_building_results(
#                 unique_hits, target_building)

#     # Get top K results by score
#     top_hits = nlargest(min(top_k, len(unique_hits)),
#                         unique_hits, key=lambda m: (m.get("score") or 0))

#     # DEBUG: Log final top results
#     logging.info("Final top %d results:", len(top_hits))
#     for i, hit in enumerate(top_hits):
#         logging.info(
#             "  %d. building_name='%s', doc_type='%s', score=%.3f, key='%s'",
#             i+1,
#             hit.get('building_name'),
#             hit.get('document_type'),
#             hit.get('score', 0),
#             hit.get('key', '')[:50])

#     answer = ""
#     publication_date_info = ""
#     score_too_low = False

#     # Check if top score is below threshold
#     if top_hits:
#         top_score = top_hits[0].get("score", 0)
#         if top_score < MIN_SCORE_THRESHOLD:
#             score_too_low = True
#             answer = f"I found some results, but they don't seem relevant enough to your question. The best matching score was {top_score:.3f}, which is below the allowable threshold of {MIN_SCORE_THRESHOLD}. Regan has told me to say I don't know. Please try rephrasing your question or asking about something else."
#             return top_hits, answer, publication_date_info, score_too_low

#     # Group results by building for better context
#     building_groups = group_results_by_building(top_hits)
#     # building_summary = get_building_context_summary(building_groups)

#     logging.info("Building groups: %s", list(building_groups.keys()))

#     if top_hits and st.session_state.get("generate_llm_answer", True):
#         # If query is building-focused, use specialised answer generation
#         if target_building and target_building in building_groups:
#             logging.info(
#                 "Generating building-focused answer for %s", target_building)
#             answer, publication_date_info = generate_building_focused_answer(
#                 query, top_hits[0], top_hits, target_building, building_groups
#             )
#         else:
#             # Use standard answer generation
#             logging.info("Generating standard answer")
#             answer, publication_date_info = enhanced_answer_with_source_date(
#                 query, top_hits[0], top_hits
#             )

#         # Add building summary if multiple buildings found and not targeting specific building
#         if len(building_groups) > 1 and not target_building:
#             answer += "\n\n**Note:** Results found across multiple buildings:\n"
#             for building, results in list(building_groups.items())[:3]:
#                 answer += f"\n- **{building}**: {len(results)} result(s)"

#     elif top_hits:
#         # If LLM answer generation is disabled, still try to get date info
#         # Find the highest-scoring operational doc
#         operational_docs = [r for r in top_hits if r.get(
#             'document_type') == 'operational_doc']

#         if operational_docs:
#             top_operational = operational_docs[0]  # Already sorted by score
#             key_value = top_operational.get("key", "")

#             index_name = top_operational.get("index") or ""
#             if key_value and index_name:
#                 idx = open_index(index_name)
#                 latest_date, _ = search_source_for_latest_date(
#                     idx, key_value, top_operational.get(
#                         "namespace", DEFAULT_NAMESPACE)
#                 )
#                 if latest_date:
#                     parsed = parse_date_string(latest_date)
#                     display_date = format_display_date(parsed)
#                     publication_date_info = f"📅 Top operational document last updated: **{display_date}**"
#                 else:
#                     publication_date_info = "📅 **Publication date unknown** for top operational document"

#     return top_hits, answer, publication_date_info, score_too_low
