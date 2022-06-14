import torch

# Current language for simplification.

# If word embeddings have been loaded.
loaded_embeddings = False

# Threshold for finding complex words from frequency.
min_complexity = 4.5

# Number of candidates to generate.
candidate_num = 10

# Use cuda if GPU is detected, otherwise use CPU.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# List of supported languages.
supported_langs = ["ar","bg","ca","cs","da",
                   "nl","en","fi","fr","de",
                   "id","it","nb","pl","pt","ro",
                   "ru","es","sv","tr","uk","hu"]
                   
lang = "en"

class UserData:

    def __init__(self, lang="en", disable_embeddings=False, embeddings=None):
        self.lang = lang
        self.disable_embeddings = disable_embeddings
        self.embeddings = embeddings
        
    def set_lang(self, lang):
        self.lang = lang
    
    def set_disable(self, disable_embeddings):
        self.disable_embeddings = disable_embeddings
        
    def set_embeddings(self, embeddings):
        self.embeddings = embeddings
        
    def __str__(self):
        return f"[ {self.lang}, {self.disable_embeddings}, {self.embeddings}]"
    

data_map = dict()
