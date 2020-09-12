# Sample Test passing with nose and pytest
import pytest
import pandas as pd 
from io import StringIO
from reddit.processing.preprocessor import (
    InputTextCreator,
    MajorityClassDownsampler,
    TextCleaner,
)
from reddit.config.base import config


def test_downsampling_produces_correct_size(raw_training_data):
    majority_class = config.model_config.downsample_params['majority_class']
    target = config.model_config.target
    df_majority = raw_training_data[raw_training_data[target] == majority_class]
    frac = 0.3

    sampler = MajorityClassDownsampler(sample_fraction=frac, 
                                      majority_class=majority_class,
                                      target=target)
    downsampled = sampler.transform(raw_training_data)
    assert len(downsampled[downsampled[target] == majority_class]) == int(frac*len(df_majority))


def test_text_creator_handles_nans(mock_dataset):
    creator = InputTextCreator(features=['title', 'body'],
                               text_field='text')
    created_df = creator.transform(mock_dataset)
    text = created_df['text'].values
    assert text[0] == 'right_missing '
    assert text[1] == 'a1234 abcd'
    assert text[2] == 'home 34 b 4'
    assert text[3] == ' left_missing'


def test_text_clean_removes_html(uncleaned_dataset):
    cleaner = TextCleaner(variable='text')
    cleaned = cleaner.transform(uncleaned_dataset)
    values = cleaned['text'].values
    assert values[2] == ' to be'


def test_text_clean_expand_contractions(uncleaned_dataset):
    cleaner = TextCleaner(variable='text')
    cleaned = cleaner.transform(uncleaned_dataset)
    values = cleaned['text'].values
    assert values[3] == 'i have box'


def test_text_clean_replaces_special_chars(uncleaned_dataset):
    cleaner = TextCleaner(variable='text')
    cleaned = cleaner.transform(uncleaned_dataset)
    assert cleaned['text'].values[0] == 'you are $3do2a'
    assert cleaned['text'].values[1] == 'abced '


