from transformers import BertForMaskedLM, BertTokenizer, logging
from gensim.models import KeyedVectors
from nltk.stem import PorterStemmer

import nltk
import torch
import math
from . import config

from .config import *

# Word embeddings.
embeddings = None

# Surpress warnings.
logging.set_verbosity(logging.ERROR)

# Load BERT model and tokenizer.
print("\nLoading BERT model...")
tokenizer = BertTokenizer.from_pretrained("bert-base-multilingual-uncased")
model = BertForMaskedLM.from_pretrained("bert-base-multilingual-uncased")
print("\nBERT model loaded!")

# Create stemmer object.
stemmer = PorterStemmer()

def load_id_embeddings(language, user_id):
    """Load word embeddings for selected language."""
    try:
        print(f"\nAttempting to load {data_map[user_id].lang} embeddings for id {user_id}...")
        wv_path = "simplifier/embeddings/" + data_map[user_id].lang + ".kv"
        wv_model = KeyedVectors.load(wv_path, mmap='r')
        print(f"\nLoaded {data_map[user_id].lang} embeddings for id {user_id}!")
    except FileNotFoundError:
        print(f"\nNo embeddings for language {data_map[user_id].lang} and id {user_id} found. Using without...")
        wv_model = None
    return wv_model
    
def load_embeddings(language):
    """Load word embeddings for selected language."""
    try:
        print(f"\nAttempting to load {config.lang} embeddings...")
        wv_path = "simplifier/embeddings/" + config.lang + ".kv"
        wv_model = KeyedVectors.load(wv_path, mmap='r')
        print(f"\nLoaded {config.lang} embeddings!")
    except FileNotFoundError:
        print(f"\nNo embeddings for language {config.lang} found. Using without...")
        wv_model = None
    return wv_model
