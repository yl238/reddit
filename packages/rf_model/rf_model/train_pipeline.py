from pathlib import Path
import pandas as pd

from sklearn.metrics import classification_report

import rf_model
from rf_model.processing.preprocessor import ValidTextCreator
from rf_model.pipeline import vectorizer_pipe, train_pipe
from rf_model.processing.data_management import save_pipeline

TARGET = 'label'
if __name__ == '__main__':
    path = Path(rf_model.__file__).resolve().parent

    file = path / "datasets/reddit_raw_with_labels.csv"
    df = pd.read_csv(file)

    creator = ValidTextCreator(cols_to_drop_na=[TARGET])
    data = creator.transform(df)

    # Split by time
    n_test = int(len(data)*0.2)
    train_df, test_df = data.iloc[n_test:, :], data.iloc[:n_test,:]
    
    y_train = train_df[TARGET]
    X_train = train_df.drop(columns=TARGET, axis=1)
    
    vectorizer_pipe.fit(X_train, y_train)
    X_train = vectorizer_pipe.transform(X_train)
    train_pipe.fit(X_train, y_train)

    # Save pipelines
    save_pipeline(pipeline_to_persist=vectorizer_pipe, file_name='vectorizer_pipe.pkl')
    save_pipeline(pipeline_to_persist=train_pipe, file_name='rf_pipe.pkl')

    # Validate
    y_test = test_df[TARGET]
    X_test = test_df.drop(columns=TARGET, axis=1)

    X_test = vectorizer_pipe.transform(X_test)
    y_pred = train_pipe.predict(X_test)
    
    print(classification_report(y_test, y_pred))
