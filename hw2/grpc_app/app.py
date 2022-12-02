from service import ModelHandler
import grpc_pb2_grpc
from concurrent import futures
import grpc


def serve():
    port = "50051"
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=1))
    grpc_pb2_grpc.add_ModelServicer_to_server(ModelHandler(), server)
    server.add_insecure_port("[::]:" + port)
    server.start()
    print("Server started, listening on " + port)
    server.wait_for_termination()


if __name__ == "__main__":
    serve()
