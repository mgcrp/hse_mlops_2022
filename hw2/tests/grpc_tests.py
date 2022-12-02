import time
import grpc
import grpc_pb2
import grpc_pb2_grpc


def run():
    print("Scenario 1: List models")
    with grpc.insecure_channel("localhost:50051") as channel:
        stub = grpc_pb2_grpc.ModelStub(channel)
        response = stub.ModelList(grpc_pb2.Empty())
    print("Result: " + str(response))
    print("---")

    time.sleep(2)

    print("Scenario 2: Add model")
    with grpc.insecure_channel("localhost:50051") as channel:
        stub = grpc_pb2_grpc.ModelStub(channel)
        response = stub.ModelAdd(
            grpc_pb2.ModelAddRequest(name="model1",
                                     type="xgboost",
                                     params="{}"))
    print("Result: " + str(response))

    time.sleep(2)

    with grpc.insecure_channel("localhost:50051") as channel:
        stub = grpc_pb2_grpc.ModelStub(channel)
        response = stub.ModelAdd(
            grpc_pb2.ModelAddRequest(name="model2", type="logreg",
                                     params="{}"))
    print("Result: " + str(response))

    time.sleep(2)

    with grpc.insecure_channel("localhost:50051") as channel:
        stub = grpc_pb2_grpc.ModelStub(channel)
        response = stub.ModelAdd(
            grpc_pb2.ModelAddRequest(name="model3", type="logreg",
                                     params="{}"))
    print("Result: " + str(response))
    print("---")

    time.sleep(2)

    print("Scenario 3: List models")
    with grpc.insecure_channel("localhost:50051") as channel:
        stub = grpc_pb2_grpc.ModelStub(channel)
        response = stub.ModelList(grpc_pb2.Empty())
    print("Result: " + str(response))
    print("---")

    time.sleep(2)

    print("Scenario 4: Train")
    with grpc.insecure_channel("localhost:50051") as channel:
        stub = grpc_pb2_grpc.ModelStub(channel)
        response = stub.ModelTrain(grpc_pb2.ModelSimpleRequest(name="model1"))
    print("Result: " + str(response))
    print("---")

    time.sleep(2)

    print("Scenario 5: List models")
    with grpc.insecure_channel("localhost:50051") as channel:
        stub = grpc_pb2_grpc.ModelStub(channel)
        response = stub.ModelList(grpc_pb2.Empty())
    print("Result: " + str(response))
    print("---")

    print("Scenario 6: Remove model")
    with grpc.insecure_channel("localhost:50051") as channel:
        stub = grpc_pb2_grpc.ModelStub(channel)
        response = stub.ModelRemove(grpc_pb2.ModelSimpleRequest(name="model3"))
    print("Result: " + str(response))
    print("---")

    print("Scenario 7: Remove same model")
    with grpc.insecure_channel("localhost:50051") as channel:
        stub = grpc_pb2_grpc.ModelStub(channel)
        response = stub.ModelRemove(grpc_pb2.ModelSimpleRequest(name="model3"))
    print("Result: " + str(response))
    print("---")

    print("Scenario 8: List models")
    with grpc.insecure_channel("localhost:50051") as channel:
        stub = grpc_pb2_grpc.ModelStub(channel)
        response = stub.ModelList(grpc_pb2.Empty())
    print("Result: " + str(response))
    print("---")

    print("Scenario 9: Predict")
    with grpc.insecure_channel("localhost:50051") as channel:
        stub = grpc_pb2_grpc.ModelStub(channel)
        response = stub.ModelPredict(
            grpc_pb2.ModelPredictRequest(
                name='model1',
                fixed_acidity=0,
                volatile_acidity=0,
                citric_acid=0,
                residual_sugar=0,
                chlorides=0,
                free_sulfur_dioxide=0,
                density=0,
                pH=0,
                sulphates=0,
                alcohol=0,
            ))
    print("Result: " + str(response))
    print("---")


if __name__ == "__main__":
    run()
