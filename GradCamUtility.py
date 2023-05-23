# Import Libraries
import tensorflow as tf
from tensorflow import keras
import cv2
import numpy as np
import streamlit as st

# Display
from IPython.display import Image
import matplotlib.pyplot as plt
import matplotlib.cm as cm


class GradCamUtils:
    def __init__(self):
        pass

    def GetImageArrayInBatch(self, img_path, size):
        img = keras.preprocessing.image.load_img(img_path, target_size=size)

        # Get the image array
        img_array = keras.preprocessing.image.img_to_array(img)

        # We add a dimension to transform our array into a "batch"
        # of size (1, size_x, size_y, channel)
        img_array = np.expand_dims(array, axis=0)
        return img_array

    @st.cache_data
    def ComputeGradCAMHeatmap(
        _self, img_array, _model, last_conv_layer_name, classifier_layer_names
    ):
        # First, we create a model that maps the input image to the activations
        # of the last conv layer
        model = _model
        last_conv_layer = model.get_layer(last_conv_layer_name)
        last_conv_layer_model = keras.Model(model.inputs, last_conv_layer.output)

        # Second, we create a model that maps the activations of the last conv
        # layer to the final class predictions
        classifier_input = keras.Input(shape=last_conv_layer.output.shape[1:])
        x = classifier_input
        for layer_name in classifier_layer_names:
            x = model.get_layer(layer_name)(x)
        classifier_model = keras.Model(classifier_input, x)

        # Then, we compute the gradient of the top predicted class for our input image
        # with respect to the activations of the last conv layer
        with tf.GradientTape() as tape:
            # Compute activations of the last conv layer and make the tape watch it
            last_conv_layer_output = last_conv_layer_model(img_array)
            tape.watch(last_conv_layer_output)
            # Compute class predictions
            preds = classifier_model(last_conv_layer_output)
            top_pred_index = tf.argmax(preds[0])
            top_class_channel = preds[:, top_pred_index]

        # This is the gradient of the top predicted class with regard to
        # the output feature map of the last conv layer
        grads = tape.gradient(top_class_channel, last_conv_layer_output)

        # This is a vector where each entry is the mean intensity of the gradient
        # over a specific feature map channel
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

        # We multiply each channel in the feature map array
        # by "how important this channel is" with regard to the top predicted class
        last_conv_layer_output = last_conv_layer_output.numpy()[0]
        pooled_grads = pooled_grads.numpy()
        for i in range(pooled_grads.shape[-1]):
            last_conv_layer_output[:, :, i] *= pooled_grads[i]

        # The channel-wise mean of the resulting feature map
        # is our heatmap of class activation
        heatmap = np.mean(last_conv_layer_output, axis=-1)

        # np.maximum(heatmap, 0) - This is the ReLU operation to ensure we consider
        # only those features that tend to have a positive effect and increase the
        # probability score for the particular class.
        # For visualization purpose, we will also normalize the heatmap between 0 & 1

        heatmap = np.maximum(heatmap, 0) / np.max(heatmap)

        return heatmap

    def DisplayHeatMap(self, heatmap):
        # Display heatmap
        plt.matshow(heatmap)
        plt.axis("off")
        plt.show()

    @st.cache_data
    def GetSuperImposedCAMImage(_self, heatmap, img):
        # Rescale heatmap to a range 0-255
        heatmap = np.uint8(255 * heatmap)

        # We use jet colormap to colorize heatmap
        jet = cm.get_cmap("jet")

        # We use RGB values of the colormap
        jet_colors = jet(np.arange(256))[:, :3]
        jet_heatmap = jet_colors[heatmap]

        # We create an image with RGB colorized heatmap
        jet_heatmap = keras.preprocessing.image.array_to_img(jet_heatmap)
        jet_heatmap = jet_heatmap.resize((img.shape[1], img.shape[0]))
        jet_heatmap = keras.preprocessing.image.img_to_array(jet_heatmap)

        superImposedImage = cv2.addWeighted(jet_heatmap, 0.2, img, 0.8, 0.0)

        return superImposedImage

    @st.cache_data
    def DisplaySuperImposedImages(_self, image, heatmap, superimposed_img):
        fig, ax = plt.subplots(1, 3, figsize=(15, 15))
        ax[0].imshow(image)
        ax[0].set_title("Original Image")
        ax[1].imshow(heatmap)
        ax[1].set_title("GradCAM Heatmap")
        ax[2].imshow(superimposed_img)
        ax[2].set_title("GradCAM Superimposed Image")
        st.pyplot(fig)
