import io
import os
import sys
import tempfile

# Configure the app before any project module is imported (they read the environment at import time)
TMP = tempfile.mkdtemp(prefix='xray-tests-')
os.environ.update({
    'MODEL_PATH': os.path.join(TMP, 'model.h5'),
    'UPLOAD_FOLDER': os.path.join(TMP, 'uploads'),
    'SECRET_KEY': 'test',
    'GEMINI_API_KEY': '',  # empty values stop load_dotenv from pulling a real key from .env
    'API_KEY': '',
    'TF_CPP_MIN_LOG_LEVEL': '2',
})
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402
import pytest  # noqa: E402
import tensorflow as tf  # noqa: E402
from PIL import Image  # noqa: E402


def make_model(nested=True):
    """Tiny stand-in for the EfficientNetB0 transfer-learning classifier."""
    tf.keras.utils.set_random_seed(0)
    if nested:
        base_in = tf.keras.Input((224, 224, 3))
        x = tf.keras.layers.Conv2D(4, 3, strides=4, activation='relu')(base_in)
        x = tf.keras.layers.Conv2D(8, 3, strides=2, activation='relu', name='out_relu')(x)
        base = tf.keras.Model(base_in, x, name='base')
        return tf.keras.Sequential([
            tf.keras.Input((224, 224, 3)),
            base,
            tf.keras.layers.GlobalAveragePooling2D(),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(1, activation='sigmoid'),
        ])
    inputs = tf.keras.Input((224, 224, 3))
    x = tf.keras.layers.Conv2D(8, 3, strides=4, activation='relu', name='block_conv')(inputs)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    return tf.keras.Model(inputs, tf.keras.layers.Dense(1, activation='sigmoid')(x))


make_model(nested=True).save(os.environ['MODEL_PATH'])


def image_bytes(fmt='PNG', size=(300, 260), mode='L'):
    rng = np.random.default_rng(0)
    img = Image.fromarray(rng.integers(0, 255, size[::-1], dtype=np.uint8), 'L').convert(mode)
    buf = io.BytesIO()
    img.save(buf, fmt)
    buf.seek(0)
    return buf


@pytest.fixture
def xray_path(tmp_path):
    path = tmp_path / 'xray.png'
    path.write_bytes(image_bytes().getvalue())
    return str(path)
