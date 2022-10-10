import time
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression

import warnings

warnings.filterwarnings("ignore")


class ModelInternalError(Exception):
    def __init__(self, message="Something went wrong"):
        self.message = message
        super().__init__(self.message)


class Model():
    def __init__(self, path, model_type, model_args={}, random_seed=42):
        np.random.seed(random_seed)

        self.modelType = None
        self.isTrained = None

        self.__replaceTrain = {3: 0, 4: 1, 5: 2, 6: 3, 7: 4, 8: 5, 9: 6}
        self.__replaceUser = {0: 3, 1: 4, 2: 5, 3: 6, 4: 7, 5: 8, 6: 9}

        try:
            self.data = pd.read_csv(path, sep=";")
        except Exception as e:
            raise ModelInternalError(message="Could not open data file!")

        self.data.drop(columns=["total sulfur dioxide"], inplace=True)
        self.__xtrain, self.__xtest, self.__ytrain, self.__ytest = train_test_split(
            self.data.drop(columns=['quality']),
            self.data.quality.replace(self.__replaceTrain),
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
        return f'Your model was successfully trained in {round(time.time() - __start, 3)} seconds!'

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
        return {
            'train_accuracy': self.model.score(self.__xtrain, self.__ytrain),
            'validation_accuracy': self.model.score(self.__xtest, self.__ytest)
        }
