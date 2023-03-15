import streamlit as st
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from model import UNET
from utils import get_loaders,get_prediction
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

#constant
display_width = 480
display_height= 320

def app():
    st.write("# Image Classification Model")

    # @st.cache_resource
    def load_model():
        model = UNET(in_channels=3, out_channels=3)
        checkpoint = torch.load("model/model-segment-classify.tar", map_location=torch.device('cpu'))
        model.load_state_dict(checkpoint["state_dict"])
        return model

    model = load_model()
    def get_prediction(img_file):
        IMAGE_HEIGHT = 160
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
        model.eval()
        with torch.no_grad():
            preds = torch.argmax(torch.sigmoid(model(transformed_image)), dim=1)
        lep_num = preds[preds==1].cpu().numpy().size
        non_lep_num = preds[preds==2].cpu().numpy().size
        if lep_num/(lep_num+non_lep_num) > 0.95: return "Leprosy"
        else: return "Non-leprosy"


    col1, col2 = st.columns(2)
    my_upload = st.sidebar.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])

    def classify_image(file_upload):
        pred_class = get_prediction(img_file = img_orig)
        st.sidebar.markdown("\n")
        st.write("#### <em>Class:   </em>"  + pred_class, unsafe_allow_html=True)

    if my_upload is not None:
        img_orig = Image.open(my_upload)
        display_image = img_orig.resize((display_width, display_height))
        col1.write("Original Image :camera:")
        col1.image(display_image)
        if st.button("Get Classification"):
            classify_image(my_upload)
    else:
        st.write("Please upload image to interact with the model")

