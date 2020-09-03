import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class DatasetCreator(BaseEstimator, TransformerMixin):
    VALID_COLS = ['title', 'score', 'num_comments', 'created_at', 'text']
    TARGET = 'label'
    
    def __init__(self, cols_to_drop_na=None, train=True, labels=None):
        self.train = train
        self.LABELS = labels

        if cols_to_drop_na is not None:
            if not isinstance(cols_to_drop_na, list):
                self.cols_to_drop_na = [cols_to_drop_na]
            else:
                self.cols_to_drop_na = cols_to_drop_na
        else:
            self.cols_to_drop_na = None
        
        if self.train:
            self.VALID_COLS = self.VALID_COLS + [self.TARGET]
        
    def _drop_na(self, X):
        if self.cols_to_drop_na is not None:
            return X.dropna(subset=self.cols_to_drop_na)
        else: 
            return X   
    
    def _get_valid_labels_only(self, X):
        if self.train:
            return X[X[self.TARGET].str.lower().isin(self.LABELS)]
        else:
            return X
    
    def _concatenate_title_body(self, X):
        X['text'] = X['title'].fillna('') + ' ' + X['body'].fillna('')
        return X
    
    def _convert_to_datetime(self, X):
        X['created_at'] = pd.to_datetime(X['created_at'])
        return X.sort_values(by='created_at').reset_index()
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X = X.copy()
        X = self._drop_na(X)
        X = self._get_valid_labels_only(X)
        X = self._concatenate_title_body(X)
        X = self._convert_to_datetime(X)
        X = X[self.VALID_COLS]
        return X
    
    
class DropUnnecessaryFeatures(BaseEstimator, TransformerMixin):
    def __init__(self, variables_to_drop=None):
        self.variables = variables_to_drop
        
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X = X.copy()
        X = X.drop(self.variables, axis=1)
        return X