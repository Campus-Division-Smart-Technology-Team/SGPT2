import os
import io
import json
import re
import hashlib
import time
import math
import random
from typing import Iterable, List, Dict, Tuple
from datetime import datetime

import boto3
from botocore.exceptions import ClientError
from pypdf import PdfReader
from docx import Document as DocxDocument
import tiktoken
from dotenv import load_dotenv

from pinecone import Pinecone, ServerlessSpec
from openai import OpenAI
from openai import APIError, RateLimitError

# ---------------- Env & constants ----------------
load_dotenv()

print("AWS key:", os.getenv("AWS_ACCESS_KEY_ID"))
print("Pinecone key:", os.getenv("PINECONE_API_KEY")[:20] + "...")
