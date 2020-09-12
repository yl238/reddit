from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import TfidfVectorizer

from reddit.config.base import config
from reddit.processing import preprocessor as pp
from reddit.processing import features

svc_pipeline = Pipeline(
    [
        ('create_text',
            pp.InputTextCreator(features=config.model_config.features,
                                text_field=config.model_config.text_field),
        ),
        ('text_cleaner',
            pp.TextCleaner(variable=config.model_config.text_field
            ),
        ),
        ('text_tokenizer',
            features.TextTokenizer(variable=config.model_config.text_field,
                stopword_exceptions=config.model_config.stopword_exceptions),
        ),
        ('vectorizer',
            TfidfVectorizer(**config.model_config.vectorizer_params),
        ),
        ('svc',
            LinearSVC(**config.model_config.model_params),
        ),
    ]
)
