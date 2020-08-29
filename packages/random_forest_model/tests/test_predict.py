import math

from random_forest_model.predict import make_prediction
from random_forest_model.processing.data_management import load_dataset


def test_make_single_prediction():
    test_data = load_dataset(file_name='test.csv')
    single_test_json = test_data[0:1]

    subject = make_prediction(input_data=single_test_json)

    assert subject is not None
    assert isinstance(subject.get('predictions')[0], float)
    assert math.ceil(subject.get('predictions')[0]) == 117590


def test_make_multiple_predictions():
    test_data = load_dataset(file_name='test.csv')
    original_data_length = len(test_data)
    multiple_test_json = test_data

    subject = make_prediction(input_data=multiple_test_json)
    
    # Then
    assert subject is not None
    assert len(subject.get('predictions')) == 1451

    # We expect some rows to be filtered out
    assert len(subject.get('predictions')) != original_data_length
