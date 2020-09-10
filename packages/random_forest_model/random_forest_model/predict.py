import pathlib
import numpy as np 
import pandas as pd 

import rf_model
from rf_model.processing.preprocessor import ValidTextCreator
from rf_model.processing.data_management import load_pipeline, load_dataset, save_dataset

PACKAGE_ROOT = pathlib.Path(rf_model.__file__).resolve().parent


vectorizer_pipeline_name = 'vectorizer_pipe.pkl'
vectorize_pipe = load_pipeline(file_name=vectorizer_pipeline_name)

model_pipe_name = 'rf_pipe.pkl'
model_pipe = load_pipeline(file_name=model_pipe_name)

def make_prediction(*, input_data) -> dict:
    """Make prediction using the saved model pipelines."""
    X_pred = vectorize_pipe.transform(input_data)
    y_pred = model_pipe.predict(X_pred)
    results = {'predictions': y_pred}

    return results


if __name__ == '__main__':
    predict_file = 'labels_700-998.csv'
    predict_data = load_dataset(file_name=predict_file)

    creator = ValidTextCreator(predict=True)
    data = creator.transform(predict_data)

    results = make_prediction(input_data=data)
    data['predicted'] = results['predictions']

    save_dataset(data=data, file_name='predicted_labels.csv')
    
