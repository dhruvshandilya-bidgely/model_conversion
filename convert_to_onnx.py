from tensorflow.keras.models import load_model
import tf2onnx
import onnx

ev_l1_loaded_model = load_model("ev_l1_cnn.h5")
ev_l2_loaded_model = load_model("ev_l2_cnn.h5")

ev_l1_onnx_model, _ = tf2onnx.convert.from_keras(ev_l1_loaded_model)
ev_l2_onnx_model, _ = tf2onnx.convert.from_keras(ev_l2_loaded_model)

onnx.save(ev_l1_onnx_model, "ev_l1_onnx_model.onnx")
onnx.save(ev_l2_onnx_model, "ev_l2_onnx_model.onnx")