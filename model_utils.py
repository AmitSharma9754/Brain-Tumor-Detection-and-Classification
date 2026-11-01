"""
model_utils.py
--------------
1. build_cnn_model()        -> defines the CNN architecture from scratch
2. make_gradcam_heatmap()   -> computes a Grad-CAM heatmap for a given image
3. overlay_heatmap()        -> blends the heatmap on top of the original MRI image

CLASS_NAMES order MUST match the order used by
`image_dataset_from_directory` during training (alphabetical by default),
which for this dataset is:
    0 -> glioma
    1 -> meningioma
    2 -> notumor
    3 -> pituitary
"""

import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras import layers, models

CLASS_NAMES = ["glioma", "meningioma", "notumor", "pituitary"]
IMG_SIZE = 150

# Name of the last convolutional layer in the model below.
# If you change the architecture, update this name to match.
LAST_CONV_LAYER_NAME = "last_conv"


def build_cnn_model(input_shape=(IMG_SIZE, IMG_SIZE, 3), num_classes=4):
    """
    A CNN built from scratch (no transfer learning), similar in spirit
    to the original NeuroMap project. 4 conv blocks + dense head.
    """
    model = models.Sequential([
        layers.Input(shape=input_shape),

        # Block 1
        layers.Conv2D(32, (3, 3), activation="relu", padding="same"),
        layers.BatchNormalization(),
        layers.MaxPooling2D(2, 2),

        # Block 2
        layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
        layers.BatchNormalization(),
        layers.MaxPooling2D(2, 2),

        # Block 3
        layers.Conv2D(128, (3, 3), activation="relu", padding="same"),
        layers.BatchNormalization(),
        layers.MaxPooling2D(2, 2),

        # Block 4 - final conv layer (Grad-CAM reads gradients from here)
        layers.Conv2D(256, (3, 3), activation="relu", padding="same", name=LAST_CONV_LAYER_NAME),
        layers.BatchNormalization(),
        layers.MaxPooling2D(2, 2),

        layers.GlobalAveragePooling2D(),
        layers.Dense(256, activation="relu"),
        layers.Dropout(0.4),
        layers.Dense(num_classes, activation="softmax"),
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def make_gradcam_heatmap(img_array, model, last_conv_layer_name=LAST_CONV_LAYER_NAME, pred_index=None):
    """
    img_array: (1, IMG_SIZE, IMG_SIZE, 3) preprocessed batch (values in [0,1])
    Returns a normalized heatmap (H, W) with values in [0, 1].

    NOTE: Instead of touching `model.get_layer(...).output` directly (which
    can raise "layer has never been called" for Sequential models loaded
    from a .keras file in Keras 3), we rebuild a fresh functional graph by
    replaying every layer on a new Input tensor. This always works
    regardless of how the model was saved/loaded.
    """
    inputs = tf.keras.Input(shape=img_array.shape[1:])
    x = inputs
    conv_output = None
    for layer in model.layers:
        x = layer(x)
        if layer.name == last_conv_layer_name:
            conv_output = x

    if conv_output is None:
        raise ValueError(
            f"Layer '{last_conv_layer_name}' not found in model. "
            f"Available layers: {[l.name for l in model.layers]}"
        )

    grad_model = tf.keras.models.Model(inputs=inputs, outputs=[conv_output, x])

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(predictions[0])
        class_channel = predictions[:, pred_index]

    grads = tape.gradient(class_channel, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-8)
    return heatmap.numpy()


def overlay_heatmap(original_img, heatmap, alpha=0.4):
    """
    original_img: (H, W, 3) uint8 RGB image
    heatmap:      (h, w) float array in [0, 1]
    Returns a uint8 RGB image with the heatmap overlaid.
    """
    heatmap_resized = cv2.resize(heatmap, (original_img.shape[1], original_img.shape[0]))
    heatmap_uint8 = np.uint8(255 * heatmap_resized)

    colored_heatmap = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    colored_heatmap = cv2.cvtColor(colored_heatmap, cv2.COLOR_BGR2RGB)

    overlay = colored_heatmap * alpha + original_img * (1 - alpha)
    overlay = np.clip(overlay, 0, 255).astype(np.uint8)
    return overlay