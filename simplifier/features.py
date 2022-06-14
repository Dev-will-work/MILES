from . import apsynp
from . import models
from .config import *

def cosine_id_sim(original, c, user_id):
    """Get cosine similarity using vector model."""
    try:
        return data_map[user_id].embeddings.similarity(original, c)
    except KeyError:
        return 0

def apsyn_id_sim(original, c, user_id):
    """Get ApsynP result."""
    try:
        return apsynp.detection(original, c, data_map[user_id].embeddings)[1]
    except KeyError:
        return 0

def cosine_sim(original, c):
    """Get cosine similarity using vector model."""
    try:
        return models.embeddings.similarity(original, c)
    except KeyError:
        return 0

def apsyn_sim(original, c):
    """Get ApsynP result."""
    try:
        return apsynp.detection(original, c, models.embeddings)[1]
    except KeyError:
        return 0
