from tensorflow.keras.models import load_model
import tf2onnx
from onnx_tf.backend import prepare
import onnx

ev_l1_onnx_model = onnx.load('ev_l1_loaded_model.onnx')
ev_l2_onnx_model = onnx.load('ev_l2_loaded_model.onnx')

ev_l1_tf_model = prepare(ev_l1_onnx_model)
ev_l2_tf_model = prepare(ev_l2_onnx_model)

ev_l1_tf_model.save('ev_l1_tf_model.keras')
ev_l2_tf_model.save('ev_l2_tf_model.keras')