import torch
import onnx
import tensorflow as tf
import subprocess
import os

# Define paths
input_model_path = './input/inference_graph.pth'
onnx_model_path = './output/model.onnx'
saved_model_dir = './output/tf_model'
tflite_model_path = './output/model.tflite'

# Load your PyTorch model
model = torch.load(input_model_path)
model.eval()

# Dummy input matching the model's input dimensions
dummy_input = torch.randn(1, 3, 512, 512)

# Export the model to ONNX format
torch.onnx.export(model, dummy_input, onnx_model_path, opset_version=11)
print(f"ONNX model saved at {onnx_model_path}")

# Convert ONNX model to TensorFlow SavedModel using onnx2tf
subprocess.run(['onnx2tf', '-i', onnx_model_path, '-o', saved_model_dir])
print(f"TensorFlow SavedModel saved at {saved_model_dir}")

# Convert TensorFlow SavedModel to TFLite
converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
tflite_model = converter.convert()

# Save the TFLite model
with open(tflite_model_path, 'wb') as f:
    f.write(tflite_model)
print(f"TFLite model saved at {tflite_model_path}")
