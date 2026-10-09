"""Pneumonia classification and Grad-CAM explanations for chest X-ray images."""
import os
import threading
import warnings

import matplotlib
import numpy as np
import tensorflow as tf
from PIL import Image

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.environ.get('MODEL_PATH', os.path.join(BASE_DIR, 'x-ray-classification.h5'))
MODEL_DRIVE_URL = 'https://drive.google.com/file/d/1O7viAG4WQetngYA0x7edasiaOncK6HWz/view?usp=sharing'

IMG_SIZE = (224, 224)
THRESHOLD = 0.5
# How pixels must be scaled before they reach the model:
#   'raw'          -> 0-255, for models that scale internally (Keras EfficientNet starts with Rescaling layers)
#   'rescale'      -> [0, 1], for models trained with ImageDataGenerator(rescale=1/255) and no Rescaling layer
#   'mobilenet_v2' -> [-1, 1], tf.keras.applications.mobilenet_v2.preprocess_input
#   'auto'         -> 'raw' if the model begins with a Rescaling layer, otherwise 'rescale'
PREPROCESSING = os.environ.get('PREPROCESSING', 'auto')
MAX_OVERLAY_SIDE = 1024

# Harmless: Keras notes that a model saved with inputs=[...] was called with a bare tensor
warnings.filterwarnings('ignore', message='The structure of `inputs` doesn', category=UserWarning)

_model = None
_model_lock = threading.Lock()
_predict_lock = threading.Lock()


class ModelNotAvailable(RuntimeError):
    pass


def load_classifier(path=MODEL_PATH):
    if not os.path.exists(path):
        raise ModelNotAvailable(
            f"Model file not found at {path}. Run `git lfs pull` or download it from {MODEL_DRIVE_URL} "
            "and place it in the project root (or set MODEL_PATH).")
    with open(path, 'rb') as f:
        if f.read(40).startswith(b'version https://git-lfs'):
            raise ModelNotAvailable(
                f"{path} is a Git LFS pointer, not the trained model. Install Git LFS and run "
                f"`git lfs install && git lfs pull`, or download the model from {MODEL_DRIVE_URL}.")
    return tf.keras.models.load_model(path, compile=False)


def get_model():
    """Load the classifier once per process; raises ModelNotAvailable with setup instructions."""
    global _model
    with _model_lock:
        if _model is None:
            _model = load_classifier()
        return _model


def load_image(path):
    """Open an image as RGB. Raises PIL.UnidentifiedImageError / OSError for non-images."""
    with Image.open(path) as img:
        return img.convert('RGB')


def resolve_preprocessing(model, mode=None):
    mode = mode or PREPROCESSING
    if mode != 'auto':
        return mode
    for layer in model.layers[:4]:
        if isinstance(layer, tf.keras.layers.Rescaling):
            return 'raw'
    return 'rescale'


def preprocess(img, mode='rescale'):
    # Nearest-neighbour matches Keras' load_img / flow_from_directory default used during training
    arr = np.asarray(img.resize(IMG_SIZE, Image.NEAREST), dtype=np.float32)
    if mode == 'raw':
        pass
    elif mode == 'mobilenet_v2':
        arr = arr / 127.5 - 1.0
    elif mode == 'rescale':
        arr = arr / 255.0
    else:
        raise ValueError(f"Unknown PREPROCESSING mode: {mode!r}")
    return arr[np.newaxis]


def _single(value):
    # Keras 3 returns one-element lists for models built with inputs=[...] / outputs=[...]
    return value[0] if isinstance(value, (list, tuple)) and len(value) == 1 else value


def pneumonia_probability(predictions):
    # Works for a single sigmoid unit and for 2-way softmax (classes sorted NORMAL, PNEUMONIA)
    return _single(predictions)[:, -1]


def _has_spatial_output(layer):
    if isinstance(layer, tf.keras.layers.InputLayer):
        return False
    try:
        return len(layer.output.shape) == 4
    except (AttributeError, ValueError):
        return False


def _find_target_layer(model):
    """Find the last layer with a 4D (spatial) output, looking inside a nested base model.

    Returns (index of the nested model in model.layers or None, the target layer).
    """
    for index, layer in reversed(list(enumerate(model.layers))):
        if isinstance(layer, tf.keras.Model):
            for inner in reversed(layer.layers):
                if _has_spatial_output(inner):
                    return index, inner
        elif _has_spatial_output(layer):
            return None, layer
    raise ValueError("Could not find a convolutional layer for Grad-CAM")


def _build_forward(model):
    """Return a function mapping an image batch to (target conv activations, predictions)."""
    nested_index, target = _find_target_layer(model)

    if nested_index is None:
        grad_model = tf.keras.Model(_single(model.inputs), [target.output, _single(model.outputs)])
        return lambda x: grad_model(x, training=False)

    # Transfer-learning models usually wrap the base network as one layer, e.g.
    # Sequential([EfficientNetB0(include_top=False), GlobalAveragePooling2D(), Dense(1)]).
    # Run the layers around the base by hand so gradients can reach its inner conv layer.
    base = model.layers[nested_index]
    base_grad = tf.keras.Model(_single(base.inputs), [target.output, _single(base.outputs)])

    def forward(x):
        for layer in model.layers[:nested_index]:
            if not isinstance(layer, tf.keras.layers.InputLayer):
                x = layer(x, training=False)
        conv_output, x = base_grad(x, training=False)
        for layer in model.layers[nested_index + 1:]:
            x = layer(x, training=False)
        return conv_output, x

    return forward


def compute_gradcam(model, img_array, explain_positive=True):
    """Grad-CAM heatmap in [0, 1] at the conv layer's resolution.

    explain_positive=True highlights evidence for pneumonia; False highlights evidence for a normal finding.
    """
    forward = _build_forward(model)
    inputs = tf.convert_to_tensor(img_array)
    with tf.GradientTape() as tape:
        conv_output, predictions = forward(inputs)
        probability = pneumonia_probability(predictions)
        score = probability if explain_positive else 1.0 - probability

    grads = tape.gradient(score, conv_output)
    weights = tf.reduce_mean(grads, axis=(0, 1, 2))
    heatmap = tf.nn.relu(tf.reduce_sum(conv_output[0] * weights, axis=-1)).numpy()

    peak = heatmap.max()
    if not np.isfinite(peak) or peak <= 0:
        return np.zeros_like(heatmap)
    return heatmap / peak


def describe_region(heatmap):
    """Describe where the heatmap is concentrated, e.g. 'lower zone, image-left (patient's right on a PA view)'."""
    if heatmap.max() <= 0:
        return None
    h, w = heatmap.shape
    ys, xs = np.mgrid[0:h, 0:w]
    weights = heatmap / heatmap.sum()
    cy, cx = (weights * ys).sum() / h, (weights * xs).sum() / w

    zone = 'upper zone' if cy < 1 / 3 else 'middle zone' if cy < 2 / 3 else 'lower zone'
    if cx < 0.4:
        side = "image-left (patient's right on a PA view)"
    elif cx > 0.6:
        side = "image-right (patient's left on a PA view)"
    else:
        side = 'central / bilateral'
    return f"{zone}, {side}"


def save_overlay(img, heatmap, out_path, alpha=0.4):
    """Blend a jet-coloured heatmap over the full-resolution image (same aspect ratio as the upload)."""
    img = img.copy()
    img.thumbnail((MAX_OVERLAY_SIDE, MAX_OVERLAY_SIDE))
    heat = Image.fromarray(np.uint8(255 * heatmap)).resize(img.size, Image.BILINEAR)
    colored = matplotlib.colormaps['jet'](np.asarray(heat, dtype=np.float32) / 255.0)[..., :3]
    overlay = (1 - alpha) * np.asarray(img, dtype=np.float32) + alpha * 255 * colored
    Image.fromarray(np.clip(overlay, 0, 255).astype(np.uint8)).save(out_path)


def analyze(image_path, gradcam_path, model=None):
    """Classify an X-ray and write its Grad-CAM overlay. Returns a JSON-serialisable summary."""
    model = model or get_model()
    img = load_image(image_path)
    batch = preprocess(img, resolve_preprocessing(model))

    # Keras models are not guaranteed to be thread-safe; serialise inference
    with _predict_lock:
        probability = float(pneumonia_probability(model(batch, training=False).numpy())[0])
        is_pneumonia = probability > THRESHOLD
        heatmap = compute_gradcam(model, batch, explain_positive=is_pneumonia)

    save_overlay(img, heatmap, gradcam_path)
    return {
        'probability': probability,
        'is_pneumonia': is_pneumonia,
        'confidence': probability if is_pneumonia else 1.0 - probability,
        'threshold': THRESHOLD,
        'attention_region': describe_region(heatmap),
    }
