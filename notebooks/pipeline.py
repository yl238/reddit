from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier

import features
import helpers as hp

VARIABLES_TO_DROP = ['title', 'score', 'num_comments', 'created_at']


vectorizer_pipe = Pipeline(
    [   
        ('drop_features',
        hp.DropUnnecessaryFeatures(variables_to_drop=VARIABLES_TO_DROP)
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