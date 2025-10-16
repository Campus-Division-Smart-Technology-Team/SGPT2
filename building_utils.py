#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Building name extraction and lookup utilities for query processing.
"""

import re
from typing import Optional, List, Dict, Any
from difflib import get_close_matches
import logging


def extract_building_from_query(query: str, known_buildings: Optional[List[str]] = None) -> Optional[str]:
    """
    Extract building name/code from user query using multiple strategies.
    """
    query_lower = query.lower()

    # Strategy 1: Common building names (hardcoded for reliability)
    common_buildings = [
        'Senate House',
        'Berkeley Square',
        'Retort House',
        'Whiteladies',
        'Dentistry',
        'Indoor Sports Hall',
    ]

    for building in common_buildings:
        if building.lower() in query_lower:
            logging.info("Found common building in query: '%s'", building)
            return building

    # Strategy 2: Explicit building references with patterns
    building_patterns = [
        r'(?:building|property|site)\s+([A-Z][A-Za-z\s]+?)(?:\s+(?:building|bms|controls|system)|\?|$)',
        r'(?:at|in|for|about)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*?)(?:\s+(?:building|bms|controls)|\?|$|\.)',
    ]

    for pattern in building_patterns:
        match = re.search(pattern, query, re.IGNORECASE)
        if match:
            building_candidate = match.group(1).strip()
            # Clean up the candidate
            building_candidate = re.sub(
                r'\s+(the|a|an)$', '', building_candidate, flags=re.IGNORECASE)

            if len(building_candidate.split()) <= 4:  # Reasonable building name length
                logging.info(
                    f"Extracted building from pattern: '{building_candidate}'")

                # Try to match against common buildings first
                for common in common_buildings:
                    if common.lower() in building_candidate.lower() or building_candidate.lower() in common.lower():
                        return common

                return building_candidate

    # Strategy 3: Look for known building names if provided
    if known_buildings:
        for building in known_buildings:
            if len(building) > 3 and building.lower() in query_lower:
                logging.info(f"Found known building in query: '{building}'")
                return building

    return None


def find_closest_building_name(extracted_name: str, known_buildings: List[str]) -> Optional[str]:
    """Find the closest matching building name using fuzzy matching."""
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

    # Strategy 4: Fuzzy match using difflib (80% similarity)
    matches = get_close_matches(
        extracted_name, known_buildings, n=1, cutoff=0.80)
    if matches:
        return matches[0]

    return None


def normalise_building_name(building_name: str) -> str:
    """
    Normalise a building name by removing common suffixes.
    E.g., "Senate House BMS Controls" -> "Senate House"
    """
    if not building_name:
        return building_name

    # Remove common suffixes
    patterns = [
        r'\s+BMS.*$',
        r'\s+Controls.*$',
        r'\s+Project.*$',
        r'\s+Manual.*$',
        r'\s+-\s+.*$',  # Remove anything after a dash
    ]

    normalised = building_name
    for pattern in patterns:
        normalised = re.sub(pattern, '', normalised, flags=re.IGNORECASE)

    return normalised.strip()


def group_results_by_building(results: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    """
    Group search results by normalised building name.
    This ensures "Senate House" and "Senate House BMS Controls" are grouped together.
    """
    grouped = {}

    for result in results:
        building_name = result.get('building_name', 'Unknown')

        # Normalise the building name for grouping
        normalised_name = normalise_building_name(building_name)

        # If normalization made it empty, use original
        if not normalised_name:
            normalised_name = building_name

        if normalised_name not in grouped:
            grouped[normalised_name] = []

        # Store with normalised name for consistent grouping
        result_copy = result.copy()
        result_copy['_normalised_building'] = normalised_name
        grouped[normalised_name].append(result_copy)

    return grouped


def get_building_context_summary(building_results: Dict[str, List[Dict[str, Any]]]) -> str:
    """Create a summary of buildings found in search results."""
    if not building_results:
        return ""

    summary_parts = []
    for building, results in building_results.items():
        doc_types = {}
        for r in results:
            doc_type = r.get('document_type', 'unknown')
            doc_types[doc_type] = doc_types.get(doc_type, 0) + 1

        type_str = ', '.join(
            [f"{count} {dtype}" for dtype, count in doc_types.items()])
        summary_parts.append(
            f"- {building}: {len(results)} results ({type_str})")

    return "Buildings found:\n" + "\n".join(summary_parts)


def prioritise_building_results(results: List[Dict[str, Any]],
                                target_building: str) -> List[Dict[str, Any]]:
    """
    Reorder results to prioritise a specific building.
    Uses flexible matching to catch name variations.
    """
    if not target_building:
        return results

    target_normalised = normalise_building_name(target_building).lower()

    # Separate results: target building first, then others
    priority_results = []
    other_results = []

    for result in results:
        building_name = result.get('building_name', '')
        normalised = normalise_building_name(building_name).lower()

        # Check if this result matches the target building
        if (target_normalised in normalised or
            normalised in target_normalised or
                target_building.lower() in building_name.lower()):
            priority_results.append(result)
        else:
            other_results.append(result)

    logging.info(
        f"Prioritised {len(priority_results)} results for '{target_building}', {len(other_results)} other results")

    # Return combined list with priority results first
    return priority_results + other_results
