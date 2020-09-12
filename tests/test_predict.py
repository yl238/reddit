import math

from reddit.predict import make_prediction
from reddit.processing.data_management import load_dataset


def test_make_single_prediction(predict_dataset):
    single_test_input = predict_dataset[1:2]
    subject = make_prediction(input_data=single_test_input)

    assert subject is not None
    assert isinstance(subject.get('predictions')[0], str)
    assert subject.get('predictions')[0] == 'live convo'    


def test_make_multiple_predictions(predict_dataset):
    original_data_length = len(predict_dataset)
    multiple_test_input = predict_dataset
    subject = make_prediction(input_data=multiple_test_input)

    assert subject is not None
    assert len(subject.get('predictions')) == 3

    # We expect some rows to be filtered out
    assert len(subject.get('predictions')) != original_data_length

