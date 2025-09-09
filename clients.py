#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Client initialisation for Pinecone and OpenAI.
"""

import os
from pinecone import Pinecone
from openai import OpenAI


def get_oai() -> OpenAI:
    """ Initialise and return OpenAI client."""
    return OpenAI()


def get_pc() -> Pinecone:
    """ Initialise and return Pinecone client."""
    return Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))


# Initialise clients
pc = get_pc()
oai = get_oai()
