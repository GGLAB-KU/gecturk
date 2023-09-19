import requests
from functools import lru_cache
from abc import ABC


class MorpAnalyzer(ABC):
    def __init__(self):
        pass
    
    def analyze(self, text):
        """Accepts a string and returns a list of morphological analyses for each word in the string."""
        pass

class Morse(MorpAnalyzer):
    def __init__(self):
        pass
    
    @lru_cache(maxsize=128)
    def analyze(self, text):
        data = data = "{\"text\": \"%s\"}" % text
        data = data.encode()
        result = requests.post("http://localhost:8080/analyze/", data, headers={"Auth-Token": "sometoken"})
        
        return result.json()["analysis"]