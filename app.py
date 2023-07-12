import os
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
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
    elif choice == "DenseNet121":
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
    # print(history_files)
    return history_files


@st.cache_data
def performance(val_acc, val_loss, lr):
    layout = go.Layout(
        title=dict(text="Validation Accuracy", x=0.4),
        titlefont=dict(size=20),
        xaxis=dict(title="Epoch"),
        yaxis=dict(title="Validation Accuracy"),
        height=500,
        width=1000,
    )
    fig = go.Figure(data=val_acc, layout=layout)
    st.plotly_chart(fig)

    layout = go.Layout(
        title=dict(text="Validation Loss", x=0.4),
        titlefont=dict(size=20),
        xaxis=dict(title="Epoch"),
        yaxis=dict(title="Validation Loss"),
        height=500,
        width=1000,
    )

    fig = go.Figure(data=val_loss, layout=layout)
    st.plotly_chart(fig)

    layout = go.Layout(
        title=dict(text="Learning Rate", x=0.4),
        titlefont=dict(size=20),
        xaxis=dict(title="Epoch"),
        yaxis=dict(title="Learning Rate"),
        height=500,
        width=1000,
    )

    fig = go.Figure(data=lr, layout=layout)
    st.plotly_chart(fig)


def main():
    # title
    st.set_page_config(page_title="Malaria Parasite Detection", layout="wide")
    st.title("Malaria Parasite Detection")

    # sidebar
    activities = ["Predict", "Performance"]
    choice = st.sidebar.selectbox("Select Activity", activities)

    # main page
    if choice == "Predict":
        positive, negative = 0, 0
        name_positive, name_negative = [], []
        model = ["Custom Model", "EfficientNetB0", "DenseNet121"]
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

                if binary == 0:
                    negative += 1
                    name_negative.append(image_file.name)
                else:
                    positive += 1
                    name_positive.append(image_file.name)

                super_imposed_image = MalNetActivations.GetSuperImposedCAMImage(
                    heatmap, img_3d
                )

                MalNetActivations.DisplaySuperImposedImages(
                    img_3d, heatmap, super_imposed_image
                )

    elif choice == "Performance":
        st.subheader("Model Performance")
        history_files = get_history()

        choice = st.sidebar.selectbox("Select Model", history_files)

        val_acc = []
        train_acc = []
        val_loss = []
        lr = []

        # for i, history in enumerate(history_files):
        df = pd.read_csv("Training History/" + choice)
        df.rename(columns={"Unnamed: 0": "epoch"}, inplace=True)

        fig = px.line(
            df,
            x="epoch",
            y=["accuracy", "val_accuracy"],
            color_discrete_sequence=["#28FFED", "#004DEE"],
            markers=True,
            height=400,
            width=800,
        )
        fig.update_layout(xaxis_title="Epoch", yaxis_title="Accuracy")

        st.plotly_chart(fig)

        fig = px.line(
            df,
            x="epoch",
            y=["loss", "val_loss"],
            color_discrete_sequence=["#FF0000", "#FFAE0C"],
            markers=True,
            height=400,
            width=800,
        )
        fig.update_layout(xaxis_title="Epoch", yaxis_title="Loss")

        st.plotly_chart(fig)

        #     val_acc_line = go.Scatter(
        #         x=df["epoch"],
        #         y=df["val_accuracy"],
        #         name=history.split(".")[0],
        #     )
        #     val_acc.append(val_acc_line)

        #     val_loss_line = go.Scatter(
        #         x=df["epoch"],
        #         y=df["val_loss"],
        #         name=history.split(".")[0],
        #     )
        #     val_loss.append(val_loss_line)

        #     learning_rate_line = go.Scatter(
        #         x=df["epoch"],
        #         y=df["lr"],
        #         name=history.split(".")[0],
        #     )
        #     lr.append(learning_rate_line)

        # performance(val_acc, val_loss, lr)


if __name__ == "__main__":
    main()
