import pandas as pd
import re
import unicodedata
import inflect
import contractions
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer

import spacy
nlp = spacy.load('en_core_web_sm')
STOPWORD_EXCEPTIONS = ["whatever", "whenever", "about", "nothing", \
    "empty", "none", "more", "somewhere", "most", "not", "never"]
nlp.Defaults.stop_words -= set(STOPWORD_EXCEPTIONS)


class TextCleaner(BaseEstimator, TransformerMixin):
    def __init__(self, variables=None):
        if not isinstance(variables, list):
            self.variables = [variables]
        else:
            self.variables = variables
            
    def _remove_https_links(self, text):
        return re.sub(r'https?://\S+', '', text, flags=re.MULTILINE)

    def _replace_non_alphanumeric(self, text):
        return re.sub(r'[^\w\'\$ ]', ' ', text, flags=re.MULTILINE)

    def _denoise_text(self, text):
        text = self._remove_https_links(text)
        text = self._replace_non_alphanumeric(text)
        return text

    def _replace_contractions(self, text):
        return contractions.fix(text)

    def _normalize(self, text):
        text = self._denoise_text(text)
        text = self._replace_contractions(text)
        return text.lower()
                    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X = X.copy()
        for features in self.variables:
            X[features] = X[features].apply(self._normalize)
        return X


class TextTokenizer(BaseEstimator, TransformerMixin):
    def __init__(self, variables=None):
        if not isinstance(variables, list):
            self.variables = [variables]
        else:
            self.variables = variables
            
    def _lemmatize_and_remove_stop_words(self, text):
        return [t.lemma_ for t in nlp(text) if not t.is_stop and len(t.lemma_) > 1]
    
    def _remove_non_ascii(self, words):
        """Remove non-ASCII characters from list 
        of tokenized words."""
        new_words = []
        for word in words:
            new_word = unicodedata.normalize('NFKD', word).encode('ascii', 'ignore').decode('utf-8', 'ignore')
            new_words.append(new_word)
        return new_words
    
    def _replace_numbers(self, words):
        """Replace all integer occurrences in list of 
        tokenized words with textual representation"""
        p = inflect.engine()
        new_words = []
        for word in words:
            if word.isdigit():
                new_word = p.number_to_words(word)
                new_words.append(new_word)
            else:
                new_words.append(word)
        return new_words
    
    def _normalize(self, text):
        words = self._lemmatize_and_remove_stop_words(text)
        words = self._remove_non_ascii(words)
        words = self._replace_numbers(words)
        return ' '.join(words)
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X = X.copy()
        for variable in self.variables:
            X[variable] = X[variable].apply(self._normalize)
        return X


class Vectorizer(TfidfVectorizer):
    def __init__(self, variable=None, **kwargs):
        self.variable = variable
        super().__init__(**kwargs)

    def fit(self, X, y=None):
        return super().fit(X[self.variable], y=None)

    def fit_transform(self, raw_documents, y=None):
        return super().fit_transform(raw_documents[self.variable], y=y)

    def transform(self, raw_documents, copy='deprecated'):
        return super().transform(raw_documents[self.variable], copy=copy)
