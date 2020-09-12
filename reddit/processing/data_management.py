import joblib
import pandas as pd
import typing as t
from sklearn.pipeline import Pipeline

import logging

from reddit.config.base import config, DATASET_DIR, TRAINED_MODEL_DIR
from reddit import __version__ as _version

_logger = logging.getLogger(__name__)

def load_dataset(*, file_name: str) -> pd.DataFrame:
    data = pd.read_csv(f"{DATASET_DIR}/{file_name}")
    return data


def save_pipeline(*, pipeline_to_persist) -> None:
    """Persist the pipeline. Save the versioned model.
    """
    save_file_name = f"{config.app_config.pipeline_save_file}{_version}.pkl"
    save_path = TRAINED_MODEL_DIR / save_file_name

    remove_old_pipelines(files_to_keep=[save_file_name])

    joblib.dump(pipeline_to_persist, save_path)
    _logger.info(f"saved pipeline: {save_file_name}")


def load_pipeline(*, file_name: str) -> Pipeline:
    """Load a persisted pipeline
    """
    file_path = TRAINED_MODEL_DIR / file_name
    trained_model = joblib.load(filename=file_path)
    return trained_model


def remove_old_pipelines(*, files_to_keep: t.List[str]) -> None:
    """Remove old model pipelines. This is to ensure there is a 
    simple one-to-one mapping between the package version and the
    model version to be imported and used by other applications.
    However, we do also include the immediate previous
    pipeline version for differential testing purposes.

    Parameters
    ----------
    files_to_keep : t.List[str]
        files to not delete
    """
    do_not_delete = files_to_keep + ['__init__.py']
    for model_file in TRAINED_MODEL_DIR.iterdir():
        if model_file.name not in do_not_delete:
            model_file.unlink()