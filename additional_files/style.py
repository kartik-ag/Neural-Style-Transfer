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
import additional_files.load_image as load_img
import additional_files.save_image as save_img
import additional_files.vgg_matrix as mat
from additional_files.transform import TransformerNet
from additional_files.vgg16 import Vgg16
import streamlit as st


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")




# @st.cache_resource
def stylize(style_model, content_image, output_image):

    # if the content image is a path then
    if type(content_image) == "str":
        content_image = load_img.load_image(
            content_image)
    content_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.mul(255))
    ])
    content_image = content_transform(content_image)

    # to treat a single image like a batch
    content_image = content_image.unsqueeze(0).to(device)

    with torch.no_grad():
        output = style_model(content_image).cpu()

    # output image here is the path to the output image
    img = save_img.save_image(output_image, output[0])
    return img


if __name__ == "__main__":
    main()
