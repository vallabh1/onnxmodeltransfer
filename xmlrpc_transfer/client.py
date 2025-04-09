# client.py
import xmlrpc.client
import cv2
import numpy as np
import matplotlib.pyplot as plt
import time
server = xmlrpc.client.ServerProxy("http://localhost:5001")

with open("../ESANET/esanet_with_preproc.onnx", "rb") as f:
    model_data = f.read()
time1 = time.time()
response = server.load_model(xmlrpc.client.Binary(model_data))
time2 = time.time()
print(time2-time1)
print("Model load response:", response)


segmentation = server.run_inference()
seg_map = np.array(segmentation, dtype=np.uint16)
print("seg_map shape:", seg_map.shape)

plt.imshow(seg_map, cmap='tab20')

# plt.figure(figsize=(12, 4))
# plt.subplot(1, 2, 1)
# plt.title("RGB")
# plt.imshow(rgb)
# plt.axis("off")

# plt.subplot(1, 3, 2)
# plt.title("Depth")
# plt.imshow(depth, cmap='gray')
# plt.axis("off")

# plt.subplot(1, 3, 3)
# plt.title("Segmentation")
# plt.imshow(seg_map, cmap='tab20')
# plt.axis("off")

# plt.tight_layout()
plt.show()

