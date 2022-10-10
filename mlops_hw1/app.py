from flask import Flask
from flask_restx import Resource, Api, reqparse, fields
from models import Model

MAX_MODEL_NUM = 10
DATA_PATH = "data/winequality-white.csv"
MODELS_DICT = dict()

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
        return {
            "models": {
                i: {
                    "type":
                    MODELS_DICT[i].modelType,
                    "isTrained":
                    MODELS_DICT[i].isTrained,
                    "train_accuracy":
                    None if not MODELS_DICT[i].isTrained else
                    MODELS_DICT[i].score()["train_accuracy"],
                    "validation_accuracy":
                    None if not MODELS_DICT[i].isTrained else
                    MODELS_DICT[i].score()["validation_accuracy"],
                }
                for i in MODELS_DICT.keys()
            }
        }, 201


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
            408: "The max number of models has been reached"
        })
    def post(self):
        __name = api.payload["name"]
        __type = api.payload["type"]
        __params = api.payload["params"]

        try:
            __params = eval(__params)
        except Exception as e:
            return {
                "status": "Failed",
                "message":
                "'params' error; Params must be a valid json or dict"
            }, 401

        if len(MODELS_DICT) >= MAX_MODEL_NUM:
            return {
                "status":
                "Failed",
                "message":
                "The max number of models has been reached; You must delete one before creating another"
            }, 408

        if __name not in MODELS_DICT.keys():
            try:
                MODELS_DICT[__name] = Model(path=DATA_PATH,
                                            model_type=__type,
                                            model_args=__params)
                return {"status": "OK", "message": "Model created!"}, 201
            except Exception as e:
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
        404: "Model with a given name does not exist"
    })
    def delete(self):
        __name = parserRemove.parse_args()["name"]
        if __name not in MODELS_DICT.keys():
            return {
                "status": "Failed",
                "message": "Model with a given name does not exist"
            }, 404
        else:
            MODELS_DICT.pop(__name)
            return {"status": "OK", "message": "Model removed!"}, 201


@api.route("/models/train")
class ModelTrain(Resource):
    @api.expect(parserTrain)
    @api.doc(
        responses={
            201: "Success",
            404: "Model with a given name does not exist",
            406: "Error while training model; See description for more info"
        })
    def get(self):
        __name = parserTrain.parse_args()["name"]

        if __name not in MODELS_DICT.keys():
            return {
                "status": "Failed",
                "message": "Model with a given name does not exist!"
            }, 404
        else:
            try:
                __msg = MODELS_DICT[__name].train()
                return {"status": "OK", "message": __msg}, 201
            except Exception as e:
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
            407: "Error while predicting result; See description for more info"
        })
    def post(self):
        __name = api.payload["name"]
        __params = api.payload
        __params.pop("name")
        if __name not in MODELS_DICT.keys():
            return {
                "status": "Failed",
                "message": "Model with a given name does not exist!"
            }, 404
        else:
            try:
                return {"result": MODELS_DICT[__name].predict(__params)}, 201
            except Exception as e:
                return {
                    "status": "Failed",
                    "message": getattr(e, "message", repr(e))
                }, 407


if __name__ == "__main__":
    app.run(debug=True)
