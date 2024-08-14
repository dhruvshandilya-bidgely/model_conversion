from tensorflow.keras.models import load_model
import torch
import tf2onnx
from onnx2pytorch import ConvertModel

ev_l1_loaded_model = load_model("ev_l1_cnn.h5")
ev_l2_loaded_model = load_model("ev_l2_cnn.h5")

ev_l1_onnx_model, _ = tf2onnx.convert.from_keras(ev_l1_loaded_model)
ev_l2_onnx_model, _ = tf2onnx.convert.from_keras(ev_l2_loaded_model)

ev_l1_pytorch_model = ConvertModel(ev_l1_onnx_model)
ev_l2_pytorch_model = ConvertModel(ev_l2_onnx_model)

torch.save(ev_l1_pytorch_model,'ev_l1_cnn_pytorch.pt')
torch.save(ev_l2_pytorch_model,'ev_l2_cnn_pytorch.pt')