import re

class RegexTokenizer:
    def __init__(self, pattern):
        self._pattern = pattern
        self._regex = re.compile(pattern)

    def tokenize(self, text):
        return self._regex.findall(text)
