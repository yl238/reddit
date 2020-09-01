from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier

from rf_model.processing import features
from rf_model.processing import preprocessor as pp

VARIABLES_TO_DROP = ['title', 'score', 'num_comments', 'created_at']


vectorizer_pipe = Pipeline(
    [   
        ('drop_features',
        pp.DropUnnecessaryFeatures(variables_to_drop=VARIABLES_TO_DROP)
        ),
        ('text_cleaner',
        features.TextCleaner(variables='text'),
        ),
        ('text_tokenizer',
        features.TextTokenizer(variables='text'),
        ),
        ('vectorizer',
        features.Vectorizer(variable='text', ngram_range=(1, 2), analyzer='word', max_features=1500)),
    ]
)

train_pipe = Pipeline(
    [
        ('random_forest_classifier',
        RandomForestClassifier(n_estimators=100, random_state=42)
        ),
    ]
)