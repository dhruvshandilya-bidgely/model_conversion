import torch
from onnx2pytorch import ConvertModel
import onnx

ev_l1_onnx_model = onnx.load('ev_l1_loaded_model.onnx')
ev_l2_onnx_model = onnx.load('ev_l2_loaded_model.onnx')

ev_l1_pytorch_model = ConvertModel(ev_l1_onnx_model)
ev_l2_pytorch_model = ConvertModel(ev_l2_onnx_model)

torch.save(ev_l1_pytorch_model,'ev_l1_cnn_pytorch.pt')
torch.save(ev_l2_pytorch_model,'ev_l2_cnn_pytorch.pt')