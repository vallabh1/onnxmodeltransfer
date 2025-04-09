# server.py
import xmlrpc.server
import onnxruntime as ort
import numpy as np
import cv2

class ModelServer:
    def __init__(self):
        self.model = None
        self.ort_session = None

    def load_model(self, binary_data):
        with open("received_model.onnx", "wb") as f:
            f.write(binary_data.data)
        self.ort_session = ort.InferenceSession("received_model.onnx", providers=["CPUExecutionProvider"])
        return "Model received and loaded."

    def run_inference(self):
        target_size = (480, 640)
        rgb = cv2.imread("../ESANET/colors.png")
        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
        rgb = cv2.resize(rgb, (target_size[1], target_size[0]))

        depth = cv2.imread("../ESANET/depths.png", cv2.IMREAD_UNCHANGED)
        if depth.ndim == 3:
            depth = cv2.cvtColor(depth, cv2.COLOR_BGR2GRAY)
        depth = cv2.resize(depth, (target_size[1], target_size[0]))
        depth = depth.astype(np.float32)


        ort_inputs = {
            "rgb": rgb,
            "depth": depth
        }
        output = self.ort_session.run(None, ort_inputs)
        
        pred = output[0]
        print((pred).shape)
        # pred = np.argmax(logits, axis=1) 
        pred = (pred)
        return pred.tolist()


server = xmlrpc.server.SimpleXMLRPCServer(("localhost", 5001))
model_server = ModelServer()
server.register_function(model_server.load_model, "load_model")
server.register_function(model_server.run_inference, "run_inference")
print("Server is ready.")
server.serve_forever()
