import onnx
import tensorflow
import torch
import torchvision
from onnx_tf.backend import prepare

from cc import cc

# Load the PyTorch model
model = torchvision.models.detection.fasterrcnn_mobilenet_v3_large_320_fpn(pretrained=True)
model.eval()

# Create dummy input
dummy_input = torch.randn(1, 3, 512, 512)

print(cc("YELLOW", "Exporting the model to ONNX from PyTorch..."))

# Export the model to ONNX
torch.onnx.export(model, dummy_input, "fasterrcnn_mobilenet_v3_large_320_fpn.onnx", opset_version=11)

print(cc("GREEN", "Model exported to ONNX!"))
print(cc("YELLOW", "Exporting the model to TensorFlow from ONNX..."))

# Load the ONNX model
onnx_model = onnx.load("fasterrcnn_mobilenet_v3_large_320_fpn.onnx")

# Convert to TensorFlow
tf_rep = prepare(onnx_model)
tf_rep.export_graph("fasterrcnn_mobilenet_v3_large_320_fpn.pb")

print(cc("GREEN", "Model exported to TensorFlow!"))
print(cc("YELLOW", "Exporting the model to TensorFlow Lite from TensorFlow..."))

# Convert the model
converter = tensorflow.lite.TFLiteConverter.from_saved_model("fasterrcnn_mobilenet_v3_large_320_fpn.pb")
tflite_model = converter.convert()

# Save the model
with open("fasterrcnn_mobilenet_v3_large_320_fpn.tflite", "wb") as f:
    f.write(tflite_model)

print(cc("GREEN", "Model exported to TensorFlow Lite!"))
