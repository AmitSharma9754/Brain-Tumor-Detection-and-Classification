"""
preprocess.py
--------------
Common image preprocessing used both while training the CNN
and while running inference in the Streamlit app.

Keeping ONE shared function guarantees that the exact same
resizing / scaling logic is used at train time and at
prediction time (a very common source of bugs in ML apps).
"""

import numpy as np
from PIL import Image

IMG_SIZE = 150  # 150x150 input, change here if you retrain with a different size


def preprocess_image(file_or_path):
    """
    Accepts either:
      - a file-like object (e.g. Streamlit's UploadedFile)
      - a path to an image on disk

    Returns:
      img_batch      -> np.array of shape (1, IMG_SIZE, IMG_SIZE, 3), scaled to [0,1]
      original_img   -> np.array (H, W, 3) uint8, the ORIGINAL resized RGB image
                         (used later to overlay the Grad-CAM heatmap)
    """
    img = Image.open(file_or_path).convert("RGB")
    img = img.resize((IMG_SIZE, IMG_SIZE))

    original_img = np.array(img)  # uint8, 0-255, for overlay later

    img_array = original_img.astype("float32") / 255.0
    img_batch = np.expand_dims(img_array, axis=0)  # (1, 150, 150, 3)

    return img_batch, original_img
