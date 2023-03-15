import streamlit as st
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
# from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from model import UNET
import cv2
from utils import get_loaders,get_prediction
from dataset import LeprosyDataset
from torch.utils.data import DataLoader
from PIL import Image

from io import BytesIO
import base64
import os
import numpy as np
import matplotlib.pyplot as plt
from torchvision.models.segmentation import fcn_resnet50


display_width = 480
display_height= 320
def app():
    st.write("# Image Segmentation Model")

    # @st.cache_resource
    def load_model():
        # st.write("STARTED LOADING")
        model = UNET(in_channels=3, out_channels=3)
        checkpoint = torch.load("model/model-segment-classify.tar", map_location=torch.device('cpu'))
        model.load_state_dict(checkpoint["state_dict"])
        return model

    model = load_model()
    def get_prediction(img_file):
        IMAGE_HEIGHT = 160  # 5792 originally
        IMAGE_WIDTH = 240
        test_transforms = A.Compose(
        [
            A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
            A.Normalize(
                mean=[0.0, 0.0, 0.0],
                std=[1.0, 1.0, 1.0],
                max_pixel_value=255.0,
            ),
            ToTensorV2(),
        ],
        )

        image = np.array(img_file.convert("RGB"))

        transformed_image = test_transforms(image=image)["image"]
        transformed_image = torch.tensor(transformed_image).unsqueeze(0)
        transformed_image_np = transformed_image.squeeze().permute(1, 2, 0).numpy()
        model.eval()
        with torch.no_grad():
            preds = torch.argmax(torch.sigmoid(model(transformed_image)), dim=1)

        preds_np = preds.squeeze().numpy()
        preds_np=preds_np.astype(np.uint8)

        print("Array data type:", preds_np.dtype)
        preds_np[preds_np==1] = 255
        preds_np[preds_np==2] = 100
        preds_np = preds_np.astype(np.uint8)
        segmented_image = Image.fromarray(preds_np)

        # Convert the segmented image to RGBA mode
        segmented_image = segmented_image.convert("RGBA")
        orig = Image.fromarray(image)
        orig = orig.resize((240, 160))
        orig = orig.convert("RGBA")

        # Blend the two images
        blended_image = Image.blend(orig, segmented_image, alpha=0)

        m_non_lep = np.where(preds_np == 100, 1, 0)
        m_lep = np.where(preds_np == 255, 1, 0)

        c = np.array(blended_image)
        c = c[..., :3]
        c[m_non_lep==1] = 0
        c[m_lep==1] = 255
        c =  Image.fromarray(c)
        c = c.resize((display_width, display_height))
        return c


    legend_style = '''
    <style>
        .legend-box {
            background-color: gray;
            border: 1px solid gray;
            padding: 10px;
            border-radius: 5px;
            width: 200px;
            margin: 0 auto;
            text-align: center;
        }
        .legend-title {
            font-weight: bold;
            margin-bottom: 10px;
        }
        .circle {
            width: 20px;
            height: 20px;
            border-radius: 50%;
            display: inline-block;
            margin-right: 10px;
            border: 1px solid black;
            vertical-align: middle;
            }
        .leprosy {
            background-color: white;
            text-align: left;
        }
        .non_leprosy {
            background-color: black;
            color: white;
            text-align: left;
        }
    </style>
    '''
    col1, col2 = st.columns(2)
    my_upload = st.sidebar.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])

    def segement_image(file_upload):

        new_image = get_prediction(img_file = img_orig)
        col2.write("Segmented Image")
        col2.image(new_image, channels="L")
        st.sidebar.markdown("\n")
        _, col3 = st.columns(2)
       # Display the legend
        col3.markdown(legend_style, unsafe_allow_html=True)
        col3.write('<div class="legend-box"><div class="legend-title">Legend</div><span class="circle leprosy"></span><span class="leprosy">Leprosy</span><br><span class="circle non_leprosy"></span><span class="non_leprosy">Other Dermatoses</span></div>', unsafe_allow_html=True)

    if my_upload is not None:

        img_orig = Image.open(my_upload)
        display_image = img_orig.resize((display_width, display_height))
        col1.write("Original Image :camera:")
        col1.image(display_image)
        if st.button("Segment Image"):
            segement_image(file_upload=my_upload)
    else:
        st.write("Please upload image to interact with the model")
