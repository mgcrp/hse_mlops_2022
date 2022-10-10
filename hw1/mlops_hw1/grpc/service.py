import grpc
import grpc_pb2
import grpc_pb2_grpc
from models import Model

MAX_MODEL_NUM = 10
DATA_PATH = "data/winequality-white.csv"
MODELS_DICT = dict()


class ModelHandler(grpc_pb2_grpc.ModelServicer):
    def ModelList(self, request, context):
        __result = grpc_pb2.ModelListResponse(models=[
            grpc_pb2.ModelDescription(
                name=i,
                type=MODELS_DICT[i].modelType,
                isTrained=MODELS_DICT[i].isTrained,
                train_accuracy=None if not MODELS_DICT[i].isTrained else
                MODELS_DICT[i].score()["train_accuracy"],
                validation_accuracy=None if not MODELS_DICT[i].isTrained else
                MODELS_DICT[i].score()["validation_accuracy"],
            ) for i in MODELS_DICT.keys()
        ])

        return __result

    def ModelAdd(self, request, context):
        __name = request.name
        __type = grpc_pb2.ModelType.Name(request.type)
        __params = request.params

        try:
            __params = eval(__params)
        except Exception as e:
            return grpc_pb2.ModelSimpleResponse(
                status=grpc_pb2.ModelMessageType.Value("Failed"),
                message="'params' error; Params must be a valid json or dict")

        if len(MODELS_DICT) >= MAX_MODEL_NUM:
            return grpc_pb2.ModelSimpleResponse(
                status=grpc_pb2.ModelMessageType.Value("Failed"),
                message=
                "The max number of models has been reached; You must delete one before creating another"
            )

        if __name not in MODELS_DICT.keys():
            try:
                MODELS_DICT[__name] = Model(path=DATA_PATH,
                                            model_type=__type,
                                            model_args=__params)
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
        if __name not in MODELS_DICT.keys():
            return grpc_pb2.ModelSimpleResponse(
                status=grpc_pb2.ModelMessageType.Value("Failed"),
                message="Model with a given name does not exist")
        else:
            MODELS_DICT.pop(__name)
            return grpc_pb2.ModelSimpleResponse(
                status=grpc_pb2.ModelMessageType.Value("OK"),
                message="Model removed!")

    def ModelTrain(self, request, context):
        __name = request.name

        if __name not in MODELS_DICT.keys():
            return grpc_pb2.ModelSimpleResponse(
                status=grpc_pb2.ModelMessageType.Value("Failed"),
                message="Model with a given name does not exist!")
        else:
            try:
                __msg = MODELS_DICT[__name].train()
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
        if __name not in MODELS_DICT.keys():
            return grpc_pb2.ModelSimpleResponse(
                status=grpc_pb2.ModelMessageType.Value("Failed"),
                message="Model with a given name does not exist!")
        else:
            try:
                return grpc_pb2.ModelPredictResponse(
                    result=MODELS_DICT[__name].predict(__params))
            except Exception as e:
                return grpc_pb2.ModelSimpleResponse(
                    status=grpc_pb2.ModelMessageType.Value("Failed"),
                    message=getattr(e, "message", repr(e)))
