import spacy
from sklearn.base import BaseEstimator, TransformerMixin
nlp = spacy.load('en_core_web_sm')

class TextTokenizer(BaseEstimator, TransformerMixin):
    def __init__(self, variable=None, stopword_exceptions=None):
        self.variable = variable
        if stopword_exceptions:
            nlp.Defaults.stop_words -= set(list(stopword_exceptions))
            
    def _lemmatize_and_remove_stop_words(self, text):
        return [t.lemma_ for t in nlp(text) if not t.is_stop and len(t.lemma_) > 1]
    
    def _normalize(self, text):
        words = self._lemmatize_and_remove_stop_words(text)
        return ' '.join(words)

    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X = X.copy()
        X = X[self.variable].apply(self._normalize)
        return X
