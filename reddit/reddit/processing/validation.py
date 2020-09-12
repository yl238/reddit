from reddit.config.base import config

import pandas as pd

def drop_na_inputs(*, input_data: pd.DataFrame) -> pd.DataFrame:
    """check prediction inputs for na values and filter."""
    validated_data = input_data.copy()
    if input_data[config.model_config.features].isnull().any().all():
        validated_data = validated_data.dropna(
            axis=0, subset=config.model_config.features
        )
    return validated_data


def drop_invalid_labels(*, input_data: pd.DataFrame) -> pd.DataFrame:
    """Drop targets not in allowed list"""
    validated_data = input_data.copy()
    validated_data = validated_data[validated_data[config.model_config.target].isin(
        config.model_config.valid_targets
    )]
    return validated_data


def validate_training_data(*, input_data: pd.DataFrame) -> pd.DataFrame:
    """Validate training data by removing NaNs and empty strings"""
    validated_data = input_data.copy()
    validated_data = drop_invalid_labels(input_data=validated_data)
    validated_data = drop_na_inputs(input_data=validated_data)
    return validated_data


def validate_inputs(*, input_data: pd.DataFrame) -> pd.DataFrame:
    """Check model inputs to match required fields
    """
    validated_data = input_data.copy()
    columns = validated_data.columns.values
    for f in config.model_config.input_columns:
        if f not in columns:
            raise ValueError(f'Input data does not contain the field {f}, abort.')
    
    validated_data = drop_na_inputs(input_data=validated_data)
    
    return validated_data