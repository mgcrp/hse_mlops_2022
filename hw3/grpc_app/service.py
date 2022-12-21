# ----------------- ИМПОРТЫ ----------------

import os
import yaml
import pickle
import psycopg2
import pandas as pd
from io import BytesIO
from models import Model
from sqlalchemy import create_engine

import grpc
import grpc_pb2
import grpc_pb2_grpc

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


class ModelHandler(grpc_pb2_grpc.ModelServicer):
    def ModelList(self, request, context):
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
        __modelsDict = __models.to_dict(orient="index")

        __result = grpc_pb2.ModelListResponse(models=[
            grpc_pb2.ModelDescription(
                name=i,
                type=__modelsDict[i]['modelType'],
                isTrained=__modelsDict[i]['isTrained'],
                train_accuracy=None if pd.isnull(__modelsDict[i]['isTrained']) else
                __modelsDict[i]['trainAccuracy'],
                validation_accuracy=None if pd.isnull(__modelsDict[i]['isTrained']) else
                __modelsDict[i]['testAccuracy'],
            ) for i in __modelsDict.keys()
        ])

        return __result

    def ModelAdd(self, request, context):
        __name = request.name
        __type = grpc_pb2.ModelType.Name(request.type)
        __rawParams = request.params

        try:
            __params = eval(__rawParams)
        except Exception as e:
            return grpc_pb2.ModelSimpleResponse(
                status=grpc_pb2.ModelMessageType.Value("Failed"),
                message="'params' error; Params must be a valid json or dict")

        try:
            __modelsList = get_existing_models()
        except Exception as e:
            return grpc_pb2.ModelSimpleResponse(
                status=grpc_pb2.ModelMessageType.Value("Failed"),
                message=getattr(e, "message", repr(e)))

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

                return grpc_pb2.ModelSimpleResponse(
                    status=grpc_pb2.ModelMessageType.Value("OK"),
                    message="Model created!")
            except Exception as e:
                return grpc_pb2.ModelSimpleResponse(
                    status=grpc_pb2.ModelMessageType.Value("Failed"),
                    message=getattr(e, "message", repr(e)))
        else:
            return grpc_pb2.ModelSimpleResponse(
                status=grpc_pb2.ModelMessageType.Value("Failed"),
                message="Model with a given name already exists")

    def ModelRemove(self, request, context):
        __name = request.name

        try:
            __modelsList = get_existing_models()
        except Exception as e:
            return grpc_pb2.ModelSimpleResponse(
                status=grpc_pb2.ModelMessageType.Value("Failed"),
                message=getattr(e, "message", repr(e)))

        if __name not in __modelsList:
            return grpc_pb2.ModelSimpleResponse(
                status=grpc_pb2.ModelMessageType.Value("Failed"),
                message="Model with a given name does not exist")
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
            return grpc_pb2.ModelSimpleResponse(
                status=grpc_pb2.ModelMessageType.Value("OK"),
                message="Model removed!")

    def ModelTrain(self, request, context):
        __name = request.name

        try:
            __modelsList = get_existing_models()
        except Exception as e:
            return grpc_pb2.ModelSimpleResponse(
                status=grpc_pb2.ModelMessageType.Value("Failed"),
                message=getattr(e, "message", repr(e)))

        if __name not in __modelsList:
            return grpc_pb2.ModelSimpleResponse(
                status=grpc_pb2.ModelMessageType.Value("Failed"),
                message="Model with a given name does not exist!")
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

                return grpc_pb2.ModelSimpleResponse(
                    status=grpc_pb2.ModelMessageType.Value("OK"),
                    message=__msg)
            except Exception as e:
                return grpc_pb2.ModelSimpleResponse(
                    status=grpc_pb2.ModelMessageType.Value("Failed"),
                    message=getattr(e, "message", repr(e)))

    def ModelPredict(self, request, context):
        __name = request.name
        __params = {
            "fixed_acidity": request.fixed_acidity,
            "volatile_acidity": request.volatile_acidity,
            "citric_acid": request.citric_acid,
            "residual_sugar": request.residual_sugar,
            "chlorides": request.chlorides,
            "free_sulfur_dioxide": request.free_sulfur_dioxide,
            "density": request.density,
            "pH": request.pH,
            "sulphates": request.sulphates,
            "alcohol": request.alcohol,
        }

        try:
            __modelsList = get_existing_models()
        except Exception as e:
            return grpc_pb2.ModelSimpleResponse(
                status=grpc_pb2.ModelMessageType.Value("Failed"),
                message=getattr(e, "message", repr(e)))

        if __name not in __modelsList:
            return grpc_pb2.ModelSimpleResponse(
                status=grpc_pb2.ModelMessageType.Value("Failed"),
                message="Model with a given name does not exist!")
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

                return grpc_pb2.ModelPredictResponse(
                    result=__model.predict(__params))
            except Exception as e:
                return grpc_pb2.ModelSimpleResponse(
                    status=grpc_pb2.ModelMessageType.Value("Failed"),
                    message=getattr(e, "message", repr(e)))
