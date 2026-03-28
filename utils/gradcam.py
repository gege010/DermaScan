"""
utils/gradcam.py
────────────────
Gradient-weighted Class Activation Mapping (Grad-CAM) for EfficientNetB0.

Grad-CAM uses the gradient of the predicted class score with respect to the
feature maps of the last convolutional layer to highlight discriminative
image regions influencing the model's decision.

Reference: Selvaraju et al. (2017) https://arxiv.org/abs/1610.02391
"""

from __future__ import annotations

import base64
import io
from typing import Optional

import cv2
import numpy as np
import tensorflow as tf
from PIL import Image


class GradCAM:
    """
    Generates Grad-CAM heatmaps for a Keras model.

    Args:
        model: Trained Keras model.
        layer_name: Name of the target convolutional layer.
                    Defaults to the last Conv layer if None.

    Example:
        >>> cam = GradCAM(model, layer_name="top_conv")
        >>> img_array = preprocess_image("photo.jpg")  # shape (1, 224, 224, 3)
        >>> heatmap = cam.generate(img_array, class_idx=5)
        >>> overlay_b64 = cam.to_base64(cam.overlay_heatmap(original_img, heatmap))
    """

    def __init__(self, model: tf.keras.Model, layer_name: Optional[str] = None):
        self.model = model
        self.layer_name = layer_name or self._find_last_conv_layer()
        self._grad_model = self._build_grad_model()

    def _find_last_conv_layer(self) -> str:
        """Auto-detect the last convolutional layer name."""
        for layer in reversed(self.model.layers):
            if isinstance(layer, tf.keras.layers.Conv2D):
                return layer.name
        raise ValueError("No Conv2D layer found in model. Specify layer_name explicitly.")

    def _build_grad_model(self) -> tf.keras.Model:
        """Build a sub-model that outputs (conv_layer_output, final_predictions)."""
        target_layer = self.model.get_layer(self.layer_name)
        return tf.keras.Model(
            inputs=self.model.inputs,
            outputs=[target_layer.output, self.model.output],
        )

    def generate(
        self,
        img_array: np.ndarray,
        class_idx: Optional[int] = None,
    ) -> np.ndarray:
        """
        Compute Grad-CAM heatmap.

        Args:
            img_array: Preprocessed image array of shape (1, H, W, 3).
            class_idx: Target class index. Uses argmax prediction if None.

        Returns:
            Normalized heatmap as float32 array of shape (H, W) in [0, 1].
        """
        img_tensor = tf.cast(img_array, tf.float32)

        with tf.GradientTape() as tape:
            conv_outputs, predictions = self._grad_model(img_tensor)
            if class_idx is None:
                class_idx = int(tf.argmax(predictions[0]))
            class_score = predictions[:, class_idx]

        # Gradients of the class score w.r.t. conv feature map
        grads = tape.gradient(class_score, conv_outputs)

        # Global average pooling of gradients → importance weights
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

        # Weight feature maps by their importance
        conv_outputs = conv_outputs[0]
        heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)

        # ReLU + normalize to [0, 1]
        heatmap = tf.nn.relu(heatmap)
        heatmap = heatmap.numpy()

        if heatmap.max() > 0:
            heatmap /= heatmap.max()

        return heatmap.astype(np.float32)

    @staticmethod
    def overlay_heatmap(
        original_img: np.ndarray,
        heatmap: np.ndarray,
        alpha: float = 0.4,
        colormap: int = cv2.COLORMAP_JET,
    ) -> np.ndarray:
        """
        Superimpose the Grad-CAM heatmap onto the original image.

        Args:
            original_img: Original image as uint8 RGB array (H, W, 3).
            heatmap: Normalized heatmap from generate(), shape (h, w).
            alpha: Opacity of the heatmap overlay (0=invisible, 1=opaque).
            colormap: OpenCV colormap constant.

        Returns:
            Overlay image as uint8 RGB array.
        """
        # Resize heatmap to match original image
        h, w = original_img.shape[:2]
        heatmap_resized = cv2.resize(heatmap, (w, h))

        # Apply colormap and blend
        heatmap_colored = cv2.applyColorMap(
            np.uint8(255 * heatmap_resized), colormap
        )
        heatmap_rgb = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)

        overlay = cv2.addWeighted(original_img, 1 - alpha, heatmap_rgb, alpha, 0)
        return overlay.astype(np.uint8)

    @staticmethod
    def to_base64(img_array: np.ndarray) -> str:
        """Convert a uint8 RGB numpy array to a base64-encoded PNG string."""
        pil_img = Image.fromarray(img_array)
        buffer = io.BytesIO()
        pil_img.save(buffer, format="PNG")
        return base64.b64encode(buffer.getvalue()).decode("utf-8")
