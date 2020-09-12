import numpy as np
import pandas as pd

from reddit.processing.data_management import load_pipeline
from reddit.config.base import config
from reddit.processing.validation import validate_inputs
from reddit import __version__ as _version

import logging
import typing as t

_logger = logging.getLogger(__name__)

pipeline_file_name = f"{config.app_config.pipeline_save_file}{_version}.pkl"
_svc_pipe = load_pipeline(file_name=pipeline_file_name)

def make_prediction(*, input_data: t.Union[pd.DataFrame, dict]) -> dict:
    data = pd.DataFrame(input_data)
    validated_data = validate_inputs(input_data=data)

    prediction = _svc_pipe.predict(validated_data[config.model_config.features])

    results = {'predictions': prediction, 'version': _version}

    _logger.info(
        f"Making predictions with model version: {_version} "
        f"Inputs: {validated_data} "
        f"Predictions: {results}"
    )

    return results