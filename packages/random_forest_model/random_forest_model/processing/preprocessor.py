from sklearn.base import BaseEstimator, TransformerMixin


class ValidTextCreator(BaseEstimator, TransformerMixin):
    LABELS = ['screeners', 'bad test', 'ratings', 'recorder', 'live convo', 'no test', 'mobile', 'bug', 'payment']
    VALID_COLS = ['title', 'score', 'num_comments', 'created_at', 'text', 'label']
    
    def __init__(self, cols_to_drop_na=None, predict=False):
        self.predict = predict

        if cols_to_drop_na is not None:
            if not isinstance(cols_to_drop_na, list):
                self.cols_to_drop_na = [cols_to_drop_na]
            else:
                self.cols_to_drop_na = cols_to_drop_na
        else:
            self.cols_to_drop_na = None
        
        if self.predict:
            self.VALID_COLS = ['title', 'score', 'num_comments', 'created_at', 'text']
        
    def _drop_na(self, X):
        if self.cols_to_drop_na is not None:
            return X.dropna(subset=self.cols_to_drop_na)
        else: 
            return X   
    
    def _get_valid_labels_only(self, X):
        if self.predict:
            return X
        else:
            return X[X['label'].str.lower().isin(self.LABELS)]
    
    def _concatenate_title_body(self, X):
        X['text'] = X['title'].fillna('') + ' ' + X['body'].fillna('')
        return X
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X = X.copy()
        X = self._drop_na(X)
        X = self._get_valid_labels_only(X)
        X = self._concatenate_title_body(X)
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