import pandas as pd
import joblib
import pandas as pd
from sklearn.base import BaseEstimator


def load_model(path: str):
    try:
        return joblib.load(path)
    except Exception as e:
        print(f'Error loading model from {path}: {e}')
        return None


def predict(model: BaseEstimator, df: pd.DataFrame) -> pd.DataFrame:
    return model.predict(df)