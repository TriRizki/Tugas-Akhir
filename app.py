# import libraries
import streamlit as st
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model, load_model

from GradCamUtility import GradCamUtils
MalNetActivations = GradCamUtils()


def get_model():
    model = load_model('Model/EfficientNetB0_TL_Model.h5', compile=False)
    return model


model = get_model()
# Get the last layer of the convolutional block
last_conv_layer_name = "batch_normalization_2"

# Get the final classification layers
classifier_layer_names = [
    "global_average_pooling2d",
    "dense"
]


def load_image(image_file):
    img = Image.open(image_file)
    return img


def predict(img):
    threshold = 0.50
    img = img.resize((135, 135))
    img = image.img_to_array(img)
    img_3d = img/255
    img_4d = np.expand_dims(img_3d, axis=0)

    predicted_probabilities = model.predict(img_4d)
    prediction_binaries = int(predicted_probabilities > threshold)

    return predicted_probabilities, prediction_binaries, img_4d, img_3d


def main():
    # title
    st.title('Malaria Parasite Detection')

    # sidebar
    activities = ['Load Image', 'Predict']
    choice = st.sidebar.selectbox('Select Activity', activities)

    # main page
    if choice == 'Load Image':
        i = 0
        st.subheader('Load Image')
        image_files = st.file_uploader(
            'Upload Image', type=['jpg', 'png', 'jpeg'], accept_multiple_files=True)
        col1, col2 = st.columns(2)
        for image_file in image_files:
            if image_file is not None:
                if i % 2 == 0:
                    img = load_image(image_file)
                    with col1:
                        st.image(img, width=300, caption='Uploaded Image')
                else:
                    img = load_image(image_file)
                    with col2:
                        st.image(img, width=300, caption='Uploaded Image')
                i += 1

    elif choice == 'Predict':
        st.subheader('Predict Result')
        image_files = st.file_uploader(
            'Upload Image', type=['jpg', 'png', 'jpeg'], accept_multiple_files=True)
        for image_file in image_files:
            if image_file is not None:
                image = load_image(image_file)
                # st.image(img, width=300, caption='Uploaded Image')
                probability, binary, img_4d, img_3d = predict(image)
                if probability[0][0] == 1.0 or probability[0][0] == 0.0:
                    continue
                if binary == 0:
                    st.write(
                        'This is Uninfected with score: {}'.format(probability[0][0]))
                else:
                    st.write(
                        'This is Parasitized with score: {}'.format(probability[0][0]))

                heatmap = MalNetActivations.ComputeGradCAMHeatmap(
                    img_4d, model, last_conv_layer_name, classifier_layer_names)

                super_imposed_image = MalNetActivations.GetSuperImposedCAMImage(
                    heatmap, img_3d)

                fig, ax = plt.subplots(1, 3, figsize=(15, 15))
                ax[0].imshow(img_3d)
                ax[0].set_title('Original Image')
                ax[1].imshow(heatmap)
                ax[1].set_title('GradCAM Heatmap')
                ax[2].imshow(super_imposed_image)
                ax[2].set_title('GradCAM Superimposed Image')
                st.pyplot(fig)


if __name__ == '__main__':
    main()
