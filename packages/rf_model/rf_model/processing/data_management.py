import pathlib
import pandas as pd 
import joblib
from sklearn.pipeline import Pipeline

import rf_model

PACKAGE_ROOT = pathlib.Path(rf_model.__file__).resolve().parent
TRAINED_PIPELINE_DIR = PACKAGE_ROOT / "trained_pipelines"
DATASET_DIR = PACKAGE_ROOT / "datasets"

def load_dataset(*, file_name: str) -> pd.DataFrame:
    data_path = DATASET_DIR / file_name
    _data = pd.read_csv(data_path)
    return _data

def save_dataset(*, data: pd.DataFrame, file_name: str) -> None:
    data_path = DATASET_DIR / file_name
    data.to_csv(data_path, index=False)


def save_pipeline(*, pipeline_to_persist: Pipeline, file_name: str) -> None:
    save_file_path = TRAINED_PIPELINE_DIR / file_name
    joblib.dump(pipeline_to_persist, save_file_path)


def load_pipeline(*, file_name: str) -> Pipeline:
    file_path = TRAINED_PIPELINE_DIR / file_name
    trained_pipeline = joblib.load(file_path)
    return trained_pipeline

