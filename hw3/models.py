# ----------------- ИМПОРТЫ ----------------

import os
import time
import yaml
import numpy as np
import pandas as pd

from sqlalchemy import create_engine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression

import warnings
warnings.filterwarnings("ignore")

# ---------- ПЕРЕМЕННЫЕ/КОНСТАНТЫ ----------

RUNTIME_DOCKER = os.environ.get('RUNTIME_DOCKER', False)

if RUNTIME_DOCKER:
    POSTGRES_HOST = os.environ['POSTGRES_HOST']
    POSTGRES_DB = os.environ['POSTGRES_DB']
    POSTGRES_USER = os.environ['POSTGRES_USER']
    POSTGRES_PASSWORD = os.environ['POSTGRES_PASSWORD']
else:
    with open("config.yaml") as f:
        config = yaml.safe_load(f)
    POSTGRES_HOST = config["POSTGRES_HOST"]
    POSTGRES_DB = config["POSTGRES_DB"]
    POSTGRES_USER = config["POSTGRES_USER"]
    POSTGRES_PASSWORD = config["POSTGRES_PASSWORD"]

POSTGRES_CONN_STRING = f"postgresql+psycopg2://{POSTGRES_USER}:{POSTGRES_PASSWORD}@{POSTGRES_HOST}:5432/{POSTGRES_DB}"

# ----------------- ФУНКЦИИ ----------------


class ModelInternalError(Exception):
    def __init__(self, message="Something went wrong"):
        self.message = message
        super().__init__(self.message)


class ModelDataError(Exception):
    def __init__(self, message="Something went wrong with your dataset"):
        self.message = message
        super().__init__(self.message)


def get_dataset_from_db():
    engine_postgres = create_engine(POSTGRES_CONN_STRING)
    __data = pd.read_sql_query(
        """
        SELECT
            "fixed acidity",
            "volatile acidity",
            "citric acid",
            "residual sugar",
            "chlorides",
            "free sulfur dioxide",
            "total sulfur dioxide",
            "density",
            "pH",
            "sulphates",
            "alcohol",
            "quality"
        FROM public.dataset;
        """,
        engine_postgres
    )
    engine_postgres.dispose()
    return __data


def check_data_quality(df):
    EXPECTED_COLUMNS = [
        "fixed acidity", "volatile acidity", "citric acid",
        "residual sugar", "chlorides", "free sulfur dioxide",
        "total sulfur dioxide", "density", "pH",
        "sulphates", "alcohol", "quality"
    ]

    if not isinstance(df, pd.DataFrame):
        raise ModelDataError(message="Model recieved something weird instead of a pd.DataFrame!")

    if not set(df.columns.tolist()) == set(EXPECTED_COLUMNS):
        raise ModelDataError(message="Model recieved a dataframe with weird columns!")

    if not df.shape[0] > 0:
        raise ModelDataError(message="Model recieved an empty dataframe!")

    if not df.isnull().sum().sum() == 0:
        raise ModelDataError(message="Model recieved a dataframe with nulls!")


# ------------------- КОД ------------------

class Model():
    def __init__(self, model_type, model_args={}, random_seed=42):
        np.random.seed(random_seed)

        self.modelType = None
        self.isTrained = None

        self.__replaceTrain = {3: 0, 4: 1, 5: 2, 6: 3, 7: 4, 8: 5, 9: 6}
        self.__replaceUser = {0: 3, 1: 4, 2: 5, 3: 6, 4: 7, 5: 8, 6: 9}

        try:
            __data = get_dataset_from_db()
        except Exception as e:
            raise ModelInternalError(message=f"Could not load data from database! Error:\n{getattr(e, 'message', repr(e))}")

        check_data_quality(__data)

        __data.drop(columns=["total sulfur dioxide"], inplace=True)
        self.__xtrain, self.__xtest, self.__ytrain, self.__ytest = train_test_split(
            __data.drop(columns=["quality"]),
            __data.quality.replace(self.__replaceTrain),
            test_size=0.2)

        self.__scaler = MinMaxScaler()
        self.__xtrain = self.__scaler.fit_transform(self.__xtrain)
        self.__xtest = self.__scaler.transform(self.__xtest)

        if model_type == "xgboost":
            try:
                self.model = XGBClassifier(**model_args)
                self.modelType = "xgboost"
                self.isTrained = False
            except Exception as e:
                raise ModelInternalError(
                    message=
                    f"Problem while initializing model! Error:\n{getattr(e, 'message', repr(e))}"
                )
        elif model_type == "logreg":
            try:
                self.model = LogisticRegression(**model_args)
                self.modelType = "logreg"
                self.isTrained = False
            except Exception as e:
                raise ModelInternalError(
                    message=
                    f"Problem while initializing model! Error:\n{getattr(e, 'message', repr(e))}"
                )
        else:
            raise ModelInternalError(
                message=
                f"Model type must be 'xgboost' or 'logreg'! {model_type} got.")

    def train(self):
        __start = time.time()
        self.model.fit(self.__xtrain, self.__ytrain)
        self.isTrained = True
        return f"Your model was successfully trained in {round(time.time() - __start, 3)} seconds!"

    def predict(self, input_dict):
        if self.isTrained:
            try:
                return self.__replaceUser[self.model.predict(
                    self.__scaler.transform(pd.json_normalize(input_dict)))[0]]
            except Exception as e:
                raise ModelInternalError(
                    message=
                    f"Problem while predicting! Error:\n{getattr(e, 'message', repr(e))}"
                )
        else:
            raise ModelInternalError(message="Model is not trained!")

    def score(self):
        if self.isTrained:
            return {
                "train_accuracy": self.model.score(self.__xtrain, self.__ytrain),
                "validation_accuracy": self.model.score(self.__xtest, self.__ytest)
            }
        else:
            raise ModelInternalError(message="Model is not trained!")
