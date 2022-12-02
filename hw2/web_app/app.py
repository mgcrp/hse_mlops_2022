# ----------------- ИМПОРТЫ ----------------

import os
import yaml
import pickle
import psycopg2
import pandas as pd

from io import BytesIO
from flask import Flask
from flask_restx import Resource, Api, reqparse, fields
from models import Model
from sqlalchemy import create_engine

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


def get_existing_models():
    """
    Забирает из БД список существующих моделей

    :return models: list(str)
    """
    engine_postgres = create_engine(POSTGRES_CONN_STRING)
    __modelsList = pd.read_sql_query(
        """
        SELECT DISTINCT "modelName"
        FROM public.models;
        """,
        engine_postgres
    ).modelName.tolist()
    engine_postgres.dispose()

    return __modelsList


# ------------------- КОД ------------------

app = Flask(__name__)
app.config["BUNDLE_ERRORS"] = True
api = Api(app)

model_add = api.model(
    "Model.add.input", {
        "name":
        fields.String(
            required=True,
            title="Model name",
            description="Used as a key in local models storage; Must be unique;"
        ),
        "type":
        fields.String(required=True,
                      title="Model type",
                      description="Must be 'logreg' or 'xgboost';"),
        "params":
        fields.String(
            required=True,
            title="Model params",
            description="Params to use in model.fit(); Must be valid dict;",
            default="{}")
    })

model_predict = api.model(
    "Model.predict.input", {
        "name":
        fields.String(required=True,
                      title="Model name",
                      description="Name of your existing trained model;"),
        "fixed_acidity":
        fields.Float(required=True,
                     title="fixed_acidity",
                     description="A fixed_acidity feature value;",
                     default=0),
        "volatile_acidity":
        fields.Float(required=True,
                     title="volatile_acidity",
                     description="A volatile_acidity feature value;",
                     default=0),
        "citric_acid":
        fields.Float(required=True,
                     title="citric_acid",
                     description="A citric_acid feature value;",
                     default=0),
        "residual_sugar":
        fields.Float(required=True,
                     title="residual_sugar",
                     description="A residual_sugar feature value;",
                     default=0),
        "chlorides":
        fields.Float(required=True,
                     title="chlorides",
                     description="A chlorides feature value;",
                     default=0),
        "free_sulfur_dioxide":
        fields.Float(required=True,
                     title="free_sulfur_dioxide",
                     description="A free_sulfur_dioxide feature value;",
                     default=0),
        "density":
        fields.Float(required=True,
                     title="density",
                     description="A density feature value;",
                     default=0),
        "pH":
        fields.Float(required=True,
                     title="pH",
                     description="A pH feature value;",
                     default=0),
        "sulphates":
        fields.Float(required=True,
                     title="sulphates",
                     description="A sulphates feature value;",
                     default=0),
        "alcohol":
        fields.Float(required=True,
                     title="alcohol",
                     description="A alcohol feature value;",
                     default=0)
    })

parserRemove = reqparse.RequestParser(bundle_errors=True)
parserRemove.add_argument("name",
                          type=str,
                          required=True,
                          help="Name of a model you want to remove",
                          location="args")

parserTrain = reqparse.RequestParser(bundle_errors=True)
parserTrain.add_argument("name",
                         type=str,
                         required=True,
                         help="Name of a model you want to train",
                         location="args")


@api.route("/models/list")
class ModelList(Resource):
    @api.doc(responses={201: "Success"})
    def get(self):
        engine_postgres = create_engine(POSTGRES_CONN_STRING)
        __models = pd.read_sql_query(
            """
            SELECT
                "modelName" as "models", "modelType", "modelParams",
                "isTrained", "trainAccuracy", "testAccuracy",
                "modifyDate"
            FROM public.models;
            """,
            engine_postgres
        )
        engine_postgres.dispose()
        __models.modifyDate = __models.modifyDate.astype(str)
        __models.reset_index(drop=True, inplace=True)
        __models.set_index("models", inplace=True)

        return __models.to_dict(orient="index"), 201


@api.route("/models/add")
class ModelAdd(Resource):
    @api.expect(model_add)
    @api.doc(
        responses={
            201: "Success",
            401: "'params' error; Params must be a valid json or dict",
            402:
            "Error while initializing model; See description for more info",
            403: "Model with a given name already exists",
            408: "Failed to reach DB"
        })
    def post(self):
        __name = api.payload["name"]
        __type = api.payload["type"]
        __rawParams = api.payload["params"]

        try:
            __params = eval(__rawParams)
        except Exception as e:
            return {
                "status": "Failed",
                "message":
                "'params' error; Params must be a valid json or dict"
            }, 401

        try:
            __modelsList = get_existing_models()
        except Exception as e:
            return {
                "status": "Failed",
                "message": getattr(e, "message", repr(e))
            }, 408

        if __name not in __modelsList:
            try:
                __model = Model(model_type=__type, model_args=__params)
                __weights = BytesIO()
                pickle.dump(__model, __weights)
                __weights.seek(0)

                engine_postgres = create_engine(POSTGRES_CONN_STRING)
                engine_postgres.execution_options(autocommit=True).execute(
                    f"""
                    INSERT INTO public.models ("modelName", "modelType", "modelParams", "weights")
                    VALUES (%s,%s,%s,%s);
                    """,
                    (__name, __type, __rawParams, psycopg2.Binary(__weights.read()))
                )
                engine_postgres.dispose()

                return {"status": "OK", "message": "Model created!"}, 201
            except Exception as e:
                raise
                return {
                    "status": "Failed",
                    "message": getattr(e, "message", repr(e))
                }, 402
        else:
            return {
                "status": "Failed",
                "message": "Model with a given name already exists"
            }, 403


@api.route("/models/remove")
class ModelRemove(Resource):
    @api.expect(parserRemove)
    @api.doc(responses={
        201: "Success",
        404: "Model with a given name does not exist",
        408: "Failed to reach DB"
    })
    def delete(self):
        __name = parserRemove.parse_args()["name"]

        try:
            __modelsList = get_existing_models()
        except Exception as e:
            return {
                "status": "Failed",
                "message": getattr(e, "message", repr(e))
            }, 408

        if __name not in __modelsList:
            return {
                "status": "Failed",
                "message": "Model with a given name does not exist"
            }, 404
        else:
            engine_postgres = create_engine(POSTGRES_CONN_STRING)
            engine_postgres.execution_options(autocommit=True).execute(
                f"""
                DELETE
                FROM public.models
                WHERE "modelName" = '{__name}';
                """
            )
            engine_postgres.dispose()
            return {"status": "OK", "message": "Model removed!"}, 201


@api.route("/models/train")
class ModelTrain(Resource):
    @api.expect(parserTrain)
    @api.doc(
        responses={
            201: "Success",
            404: "Model with a given name does not exist",
            406: "Error while training model; See description for more info",
            408: "Failed to reach DB"
        })
    def post(self):
        __name = parserTrain.parse_args()["name"]

        try:
            __modelsList = get_existing_models()
        except Exception as e:
            return {
                "status": "Failed",
                "message": getattr(e, "message", repr(e))
            }, 408

        if __name not in __modelsList:
            return {
                "status": "Failed",
                "message": "Model with a given name does not exist!"
            }, 404
        else:
            try:
                engine_postgres = create_engine(POSTGRES_CONN_STRING)
                __modelRaw = engine_postgres.execution_options(autocommit=True).execute(
                    f"""
                    SELECT weights
                    FROM public.models
                    WHERE "modelName" = '{__name}';
                    """
                ).fetchone()
                engine_postgres.dispose()
                __model = pickle.loads(__modelRaw[0])

                __msg = __model.train()

                __weights = BytesIO()
                pickle.dump(__model, __weights)
                __weights.seek(0)

                engine_postgres = create_engine(POSTGRES_CONN_STRING)
                engine_postgres.execution_options(autocommit=True).execute(
                    f"""
                    UPDATE public.models
                    SET
                        "isTrained" = True,
                        "trainAccuracy" = {round(__model.score()["train_accuracy"], 20)},
                        "testAccuracy" = {round(__model.score()["validation_accuracy"], 20)},
                        "weights" = %s,
                        "modifyDate" = now()
                    WHERE "modelName" = '{__name}';
                    """,
                    (psycopg2.Binary(__weights.read()))
                )
                engine_postgres.dispose()
                return {"status": "OK", "message": __msg}, 201
            except Exception as e:
                raise
                return {
                    "status": "Failed",
                    "message": getattr(e, "message", repr(e))
                }, 406


@api.route("/models/predict")
class ModelPredict(Resource):
    @api.expect(model_predict)
    @api.doc(
        responses={
            201: "Success",
            404: "Model with a given name does not exist",
            407: "Error while predicting result; See description for more info",
            408: "Failed to reach DB"
        })
    def post(self):
        __name = api.payload["name"]
        __params = api.payload
        __params.pop("name")

        try:
            __modelsList = get_existing_models()
        except Exception as e:
            return {
                "status": "Failed",
                "message": getattr(e, "message", repr(e))
            }, 408

        if __name not in __modelsList:
            return {
                "status": "Failed",
                "message": "Model with a given name does not exist!"
            }, 404
        else:
            try:
                engine_postgres = create_engine(POSTGRES_CONN_STRING)
                __modelRaw = engine_postgres.execution_options(autocommit=True).execute(
                    f"""
                    SELECT weights
                    FROM public.models
                    WHERE "modelName" = '{__name}';
                    """
                ).fetchone()
                engine_postgres.dispose()

                __model = pickle.loads(__modelRaw[0])
                return {"result": __model.predict(__params)}, 201
            except Exception as e:
                return {
                    "status": "Failed",
                    "message": getattr(e, "message", repr(e))
                }, 407


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
