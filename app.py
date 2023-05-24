import os
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import tensorflow as tf
import efficientnet.tfkeras
from PIL import Image
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model, load_model

from GradCamUtility import GradCamUtils

MalNetActivations = GradCamUtils()


@st.cache_resource
def get_model(choice):
    if choice == "EfficientNetB0":
        predict_model = load_model("Model/EfficientNetB0_TL_Model.h5")
    elif choice == "Custom Model":
        predict_model = load_model("Model/Custom_Model_with_GAP_Layer.h5")
    else:
        predict_model = load_model("Model/DenseNet121_TL_Model.h5")

    gradcam_model = load_model("Model/Custom_Model_with_GAP_Layer.h5")
    return predict_model, gradcam_model


@st.cache_data
def load_image(image_file):
    img = Image.open(image_file)
    return img


def predict(img, predict_model):
    threshold = 0.50
    img = img.resize((135, 135))
    img = image.img_to_array(img)
    img_3d = img / 255
    img_4d = np.expand_dims(img_3d, axis=0)
    predicted_probabilities = predict_model.predict(img_4d)
    prediction_binaries = int(predicted_probabilities > threshold)

    return predicted_probabilities, prediction_binaries, img_4d, img_3d


@st.cache_data
def get_history():
    history_files = os.listdir("Training History")
    history_files.sort()
    return history_files


@st.cache_data
def performance(files):
    name = [
        "Custom_Model_AVG",
        "Custom_Model_GAP",
        "DenseNet121",
        "EfficientNetB4",
        "EfficientNetB0",
        "InceptionResNetV2",
        "InceptionV3",
        "ResNet50V2",
        "Xception",
    ]

    data_val_acc = []
    data_val_loss = []

    for i, file in enumerate(files):
        history = pd.read_csv("Training History/" + file)
        history.rename(columns={"Unnamed: 0": "epoch"}, inplace=True)

        val_acc_line = go.Scatter(
            x=history["epoch"],
            y=history["val_accuracy"],
            name=name[i],  # Use the history file name as the legend label
        )
        data_val_acc.append(val_acc_line)

        val_loss_line = go.Scatter(
            x=history["epoch"],
            y=history["val_loss"],
            name=name[i],  # Use the history file name as the legend label
        )

        data_val_loss.append(val_loss_line)

    layout = go.Layout(
        title=dict(text="Validation Accuracy", x=0.4),
        titlefont=dict(size=20),
        xaxis=dict(title="Epoch"),
        yaxis=dict(title="Validation Accuracy"),
        height=500,
        width=1000,
    )

    fig = go.Figure(data=data_val_acc, layout=layout)

    st.plotly_chart(fig)

    layout = go.Layout(
        title=dict(text="Validation Loss", x=0.4),
        titlefont=dict(size=20),
        xaxis=dict(title="Epoch"),
        yaxis=dict(title="Validation Loss"),
        height=500,
        width=1000,
    )

    fig = go.Figure(data=data_val_loss, layout=layout)

    st.plotly_chart(fig)


def main():
    # title
    st.set_page_config(page_title="Malaria Parasite Detection", layout="wide")
    st.title("Malaria Parasite Detection")

    # sidebar
    activities = ["Load", "Predict", "Performance"]
    choice = st.sidebar.selectbox("Select Activity", activities)

    # main page
    if choice == "Load":
        st.subheader("Load Image")
        image_files = st.file_uploader(
            "Upload Image", type=["jpg", "png", "jpeg"], accept_multiple_files=True
        )
        col1, col2 = st.columns(2)

        for i, image_file in enumerate(image_files):
            if image_file is not None:
                if i % 2 == 0:
                    img = load_image(image_file)
                    with col1:
                        st.image(img, width=300, caption="Uploaded Image")
                else:
                    img = load_image(image_file)
                    with col2:
                        st.image(img, width=300, caption="Uploaded Image")

    elif choice == "Predict":
        model = ["EfficientNetB0", "Custom Model", "DenseNet121"]
        model_choice = st.sidebar.selectbox("Select Model", model)

        predict_model, gradcam_model = get_model(model_choice)

        # Get the last layer of the convolutional block
        last_conv_layer_name = "batch_normalization_2"

        # Get the final classification layers
        classifier_layer_names = ["global_average_pooling2d", "dense"]

        st.subheader("Predict Result")
        image_files = st.file_uploader(
            "Upload Image", type=["jpg", "png", "jpeg"], accept_multiple_files=True
        )

        for image_file in image_files:
            if image_file is not None:
                image = load_image(image_file)
                # st.image(img, width=300, caption='Uploaded Image')
                probability, binary, img_4d, img_3d = predict(image, predict_model)

                heatmap = MalNetActivations.ComputeGradCAMHeatmap(
                    img_4d, gradcam_model, last_conv_layer_name, classifier_layer_names
                )

                if np.isnan(heatmap).any():
                    continue

                prob_scr = round(probability[0][0], 2) * 100

                if prob_scr == 0:
                    infection_severity = "Uninfected"
                    pass
                elif prob_scr > 0 and prob_scr <= 5:
                    infection_severity = "Very Low"
                    pass
                elif prob_scr > 5 and prob_scr <= 25:
                    infection_severity = "Low"
                    pass
                elif prob_scr > 25 and prob_scr <= 50:
                    infection_severity = "Moderate"
                    pass
                else:
                    infection_severity = "High"

                st.write(
                    """**Infection Probability: {:.2%}**\n\n **Infection Severity: {}**""".format(
                        float(probability[0][0]), infection_severity
                    )
                )

                super_imposed_image = MalNetActivations.GetSuperImposedCAMImage(
                    heatmap, img_3d
                )

                MalNetActivations.DisplaySuperImposedImages(
                    img_3d, heatmap, super_imposed_image
                )

    elif choice == "Performance":
        st.subheader("Model Performance")
        history_files = get_history()

        performance(history_files)


if __name__ == "__main__":
    main()
