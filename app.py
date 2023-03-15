
import streamlit as st
from multiapp import MultiApp
from pathlib import Path
from apps import image_classify, segment_image, classify_operation

# st.set_page_config(page_title="Rubenstein Library Card Catalog", page_icon= "data/img/DUL_logo_blue.jpg",
#                    layout='wide', initial_sidebar_state='auto')

st.set_page_config(layout="wide", page_title="Image Background Remover")

app = MultiApp()

app.add_app("Image Classification", image_classify.app)

app.add_app("Image Segmentation", segment_image.app)
app.add_app("Operational Classification", classify_operation.app)
# The main app
app.run()
