#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Building name extraction and lookup utilities for query processing.
"""

import re
from typing import Optional, List, Dict, Any
from difflib import get_close_matches
import logging


def extract_building_from_query(query: str, known_buildings: List[str] = None) -> Optional[str]:
    """
    Extract building name/code from user query using multiple strategies.

    Args:
        query: User's search query
        known_buildings: Optional list of known building names for fuzzy matching

    Returns:
        Extracted building name or None
    """
    query_lower = query.lower()

    # Strategy 1: Explicit building references with patterns
    building_patterns = [
        r'(?:building|property|site)\s+([A-Z0-9][A-Z0-9\s-]+?)(?:\s+(?:building|bms|controls|system)|\?|$)',
        r'(?:at|in|for)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*?)(?:\s+(?:building|bms|controls)|\?|$)',
        r'\b(Senate House|Berkeley Square|Retort House|Whiteladies|Dentistry|Indoor Sports Hall)\b',
    ]

    for pattern in building_patterns:
        match = re.search(pattern, query, re.IGNORECASE)
        if match:
            building_candidate = match.group(1).strip()
            # If we have known buildings, try to match
            if known_buildings:
                matched = find_closest_building_name(
                    building_candidate, known_buildings)
                if matched:
                    logging.info(
                        f"Extracted building from query: '{building_candidate}' -> '{matched}'")
                    return matched
            return building_candidate

    # Strategy 2: Look for known building names directly in query
    if known_buildings:
        for building in known_buildings:
            if building.lower() in query_lower:
                logging.info(f"Found building name in query: '{building}'")
                return building

    # Strategy 3: Fuzzy match against known buildings
    if known_buildings:
        # Extract potential building names (capitalized phrases)
        words = query.split()
        for i in range(len(words)):
            for j in range(i+1, min(i+4, len(words)+1)):  # Check up to 3-word phrases
                phrase = ' '.join(words[i:j])
                if phrase[0].isupper():  # Only check capitalized phrases
                    matched = find_closest_building_name(
                        phrase, known_buildings)
                    if matched and matched != phrase:
                        logging.info(
                            f"Fuzzy matched building: '{phrase}' -> '{matched}'")
                        return matched

    return None


def find_closest_building_name(extracted_name: str, known_buildings: List[str]) -> Optional[str]:
    """
    Find the closest matching building name using fuzzy matching.

    Args:
        extracted_name: Building name extracted from query or filename
        known_buildings: List of known building names

    Returns:
        Matched building name or None
    """
    if not extracted_name or not known_buildings:
        return None

    # Strategy 1: Exact match (case-insensitive)
    for building in known_buildings:
        if building.lower() == extracted_name.lower():
            return building

    # Strategy 2: Substring match (extracted name in building name)
    for building in known_buildings:
        if extracted_name.lower() in building.lower():
            return building

    # Strategy 3: Reverse substring match (building name in extracted name)
    for building in known_buildings:
        if building.lower() in extracted_name.lower():
            return building

    # Strategy 4: Fuzzy match using difflib (65% similarity)
    matches = get_close_matches(
        extracted_name, known_buildings, n=1, cutoff=0.65)
    if matches:
        return matches[0]

    return None


def group_results_by_building(results: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    """
    Group search results by building name.

    Args:
        results: List of search results with metadata

    Returns:
        Dictionary mapping building names to their results
    """
    grouped = {}

    for result in results:
        building_name = result.get('building_name', 'Unknown')
        if building_name not in grouped:
            grouped[building_name] = []
        grouped[building_name].append(result)

    return grouped


def get_building_context_summary(building_results: Dict[str, List[Dict[str, Any]]]) -> str:
    """
    Create a summary of buildings found in search results.

    Args:
        building_results: Grouped results by building

    Returns:
        Summary string
    """
    if not building_results:
        return ""

    summary_parts = []
    for building, results in building_results.items():
        doc_types = set()
        for r in results:
            doc_type = r.get('document_type', 'unknown')
            doc_types.add(doc_type)

        type_str = ', '.join(doc_types)
        summary_parts.append(
            f"- {building}: {len(results)} results ({type_str})")

    return "Buildings found:\n" + "\n".join(summary_parts)


def prioritize_building_results(results: List[Dict[str, Any]],
                                target_building: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    Reorder results to prioritize a specific building.

    Args:
        results: List of search results
        target_building: Building name to prioritize

    Returns:
        Reordered list of results
    """
    if not target_building:
        return results

    target_building_lower = target_building.lower()

    # Separate results: target building first, then others
    priority_results = []
    other_results = []

    for result in results:
        building_name = result.get('building_name', '').lower()
        if target_building_lower in building_name or building_name in target_building_lower:
            priority_results.append(result)
        else:
            other_results.append(result)

    # Return combined list with priority results first
    return priority_results + other_results


def extract_building_names_from_index(idx) -> List[str]:
    """
    Extract unique building names from a Pinecone index (if possible).
    This is a fallback if we don't have the CSV loaded.

    Note: This requires iterating through the index which may be slow.
    In practice, it's better to cache building names from the CSV.
    """
    # This is a placeholder - Pinecone doesn't provide an easy way to get unique metadata values
    # In practice, you should load building names from your source CSV or maintain a separate cache
    logging.warning(
        "extract_building_names_from_index called - this is not implemented efficiently")
    return []
