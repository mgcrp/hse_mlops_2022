# ----------------- ИМПОРТЫ ----------------

import sys
sys.path.insert(0, sys.path[0][:sys.path[0].rfind("/")])

import re
import pytest
import numpy as np
import pandas as pd

import models
from models import Model, ModelDataError, ModelInternalError

# ---------- ПЕРЕМЕННЫЕ/КОНСТАНТЫ ----------

TEST_DATA_PATH = "tests/test_modelClass_data.csv"
TEST_PREDICT = {
    "fixed_acidity": 0,
    "volatile_acidity": 0,
    "citric_acid": 0,
    "residual_sugar": 0,
    "chlorides": 0,
    "free_sulfur_dioxide": 0,
    "density": 0,
    "pH": 0,
    "sulphates": 0,
    "alcohol": 0
}

# ----------------- ФУНКЦИИ ----------------

@pytest.fixture()
def fixture_testData():
    __df = pd.read_csv(TEST_DATA_PATH, sep=";")
    yield __df

# ------------------- КОД ------------------

# 0 - Простой импорт-тест
def test_import(mocker, fixture_testData):
    def get_copy():
        return fixture_testData.copy()

    mocker.patch.object(models, "get_dataset_from_db", get_copy)

    __model = Model("logreg")
    __model = Model("xgboost")

# 1 - Тестируем инит c кривыми данными
def test_data(mocker, fixture_testData):
    def get_copy():
        return fixture_testData.copy()

    mocker.patch.object(models, "get_dataset_from_db", return_value="lol")
    with pytest.raises(ModelDataError, match="Model recieved something weird instead of a pd.DataFrame!"):
        __model = Model("logreg")
        __model = Model("xgboost")

    mocker.patch.object(models, "get_dataset_from_db", return_value=pd.DataFrame(columns=get_copy().columns).drop(columns=get_copy().columns[0]))
    with pytest.raises(ModelDataError, match="Model recieved a dataframe with weird columns!"):
        __model = Model("logreg")
        __model = Model("xgboost")

    mocker.patch.object(models, "get_dataset_from_db", return_value=pd.DataFrame(columns=get_copy().columns))
    with pytest.raises(ModelDataError, match="Model recieved an empty dataframe!"):
        __model = Model("logreg")
        __model = Model("xgboost")

    tmp = get_copy()
    tmp.loc[tmp.shape[0] - 1, tmp.columns[0]] = np.nan
    mocker.patch.object(models, "get_dataset_from_db", return_value=tmp)
    with pytest.raises(ModelDataError, match="Model recieved a dataframe with nulls!"):
        __model = Model("logreg")
        __model = Model("xgboost")

    mocker.patch.object(models, "get_dataset_from_db", get_copy)
    __model = Model("logreg")
    __model = Model("xgboost")

# 2 - Тестируем действия с необученной моделью
def test_untrained(mocker, fixture_testData):
    def get_copy():
        return fixture_testData.copy()

    mocker.patch.object(models, "get_dataset_from_db", get_copy)
    __model = Model("logreg")
    __model = Model("xgboost")

    with pytest.raises(ModelInternalError, match="Model is not trained!"):
        __model = Model("logreg")
        __model.score()
        __model.predict(TEST_PREDICT)

        __model = Model("xgboost")
        __model.score()
        __model.predict(TEST_PREDICT)

# 2 - Тестируем что обученная модель адекватит
def test_trained(mocker, fixture_testData):
    def get_copy():
        return fixture_testData.copy()

    mocker.patch.object(models, "get_dataset_from_db", get_copy)

    __model = Model("logreg")
    assert re.fullmatch(r"Your model was successfully trained in (\d*\.)?\d+ seconds!", __model.train()), "Wrong log after model trained"
    assert isinstance(__model.predict(TEST_PREDICT), int), "Wrong PREDICT output type"
    assert isinstance(__model.score(), dict), "Wrong SCORE output type"
    assert set(__model.score().keys()) == set(["train_accuracy", "validation_accuracy"]), "Wrong SCORE output structure"
    assert isinstance(__model.score()["train_accuracy"], float), "Wrong SCORE.train_accuracy data type"
    assert isinstance(__model.score()["validation_accuracy"], float), "Wrong SCORE.validation_accuracy data type"


    __model = Model("xgboost")
    assert re.fullmatch(r"Your model was successfully trained in (\d*\.)?\d+ seconds!", __model.train()), "Wrong log after model trained"
    assert isinstance(__model.predict(TEST_PREDICT), int), "Wrong PREDICT output type"
    assert isinstance(__model.score(), dict), "Wrong SCORE output type"
    assert set(__model.score().keys()) == set(["train_accuracy", "validation_accuracy"]), "Wrong SCORE output structure"
    assert isinstance(__model.score()["train_accuracy"], float), "Wrong SCORE.train_accuracy data type"
    assert isinstance(__model.score()["validation_accuracy"], float), "Wrong SCORE.validation_accuracy data type"
