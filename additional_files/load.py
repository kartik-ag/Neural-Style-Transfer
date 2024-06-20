import argparse
import os
import sys
import time
import re

import numpy as np
import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms
import torch.onnx

import additional_files.vgg_matrix as vgg_matrix
from additional_files.transform import TransformerNet
from additional_files.vgg16 import Vgg16
import streamlit as st

# we will use the conecpt of caching here that is once a user has used a particular model instead of loading
# it again and again everytime they use it we will cache the model.

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



@st.cache_resource
def load_model(model_path):

    with torch.no_grad():
        style_model = TransformerNet()  # transformer_net.py contain the style model
        state_dict = torch.load(model_path)
        # remove saved deprecated running_* keys in InstanceNorm from the checkpoint
        for k in list(state_dict.keys()):
            if re.search(r'in\d+\.running_(mean|var)$', k):
                del state_dict[k]
        style_model.load_state_dict(state_dict)
        style_model.to(device)
        style_model.eval()
        return style_model
    


if __name__ == "__main__":
    main()
