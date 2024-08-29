from tensorflow.keras.models import load_model
from tensorflow import summary
import torch
import torchsummary

def compare_models( pt_model_name: str):
    """
    Prints model architecture of both models

    Parameters:
        tf_model_name       (str)       : tensorflow model path to be compared
        pt_model_name       (str)       : pytorch model path to be compared
    
    Returns:
        (None) : prints both model layers to be inspected.
    """


    pt_loaded_model = torch.load(pt_model_name)
    print(torchsummary.summary(pt_loaded_model))

compare_models('ev_l1_cnn_pytorch.pt')
