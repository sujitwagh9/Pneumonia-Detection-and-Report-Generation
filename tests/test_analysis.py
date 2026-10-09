import numpy as np
import pytest
import tensorflow as tf
from PIL import Image

import analysis
import evaluate
from conftest import make_model


@pytest.mark.parametrize('nested', [True, False])
def test_gradcam_finds_conv_layer_in_flat_and_nested_models(nested, xray_path):
    model = make_model(nested)
    batch = analysis.preprocess(analysis.load_image(xray_path))
    for positive in (True, False):
        heatmap = analysis.compute_gradcam(model, batch, explain_positive=positive)
        assert heatmap.ndim == 2 and heatmap.shape[0] > 1
        assert np.all(np.isfinite(heatmap))
        assert heatmap.min() >= 0 and heatmap.max() <= 1


def test_gradcam_handles_models_built_with_list_inputs_and_outputs(xray_path, tmp_path):
    # Shape of the real EfficientNet model: functional, inputs=[...], outputs=[...], saved as .h5
    inputs = tf.keras.Input((224, 224, 3))
    x = tf.keras.layers.Rescaling(1 / 255)(inputs)
    x = tf.keras.layers.Conv2D(8, 3, strides=4, activation='relu', name='top_conv')(x)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    path = str(tmp_path / 'list_io.h5')
    tf.keras.Model([inputs], [tf.keras.layers.Dense(1, activation='sigmoid')(x)]).save(path)
    model = analysis.load_classifier(path)

    result = analysis.analyze(xray_path, str(tmp_path / 'cam.png'), model=model)
    assert 0 <= result['probability'] <= 1
    heatmap = analysis.compute_gradcam(model, analysis.preprocess(analysis.load_image(xray_path)))
    assert np.all(np.isfinite(heatmap))


def test_gradcam_returns_zeros_instead_of_nan_when_gradients_vanish():
    inputs = tf.keras.Input((224, 224, 3))
    x = tf.keras.layers.Conv2D(2, 3, kernel_initializer='zeros', name='conv')(inputs)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    model = tf.keras.Model(inputs, tf.keras.layers.Dense(1, activation='sigmoid')(x))
    heatmap = analysis.compute_gradcam(model, np.zeros((1, 224, 224, 3), np.float32))
    assert np.all(heatmap == 0)


def test_load_classifier_explains_git_lfs_pointer(tmp_path):
    pointer = tmp_path / 'model.h5'
    pointer.write_text('version https://git-lfs.github.com/spec/v1\noid sha256:abc\nsize 1\n')
    with pytest.raises(analysis.ModelNotAvailable, match='git lfs pull'):
        analysis.load_classifier(str(pointer))
    with pytest.raises(analysis.ModelNotAvailable, match='not found'):
        analysis.load_classifier(str(tmp_path / 'missing.h5'))


def test_preprocess_modes():
    img = Image.new('RGB', (50, 80), (255, 255, 255))
    assert analysis.preprocess(img, 'rescale').shape == (1, 224, 224, 3)
    assert analysis.preprocess(img, 'rescale').max() == pytest.approx(1.0)
    assert analysis.preprocess(Image.new('RGB', (5, 5)), 'mobilenet_v2').min() == pytest.approx(-1.0)
    assert analysis.preprocess(img, 'raw').max() == pytest.approx(255.0)
    with pytest.raises(ValueError):
        analysis.preprocess(img, 'nope')


def test_auto_preprocessing_uses_raw_pixels_when_model_rescales_itself():
    inputs = tf.keras.Input((224, 224, 3))
    x = tf.keras.layers.Rescaling(1 / 255)(inputs)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    rescaling_model = tf.keras.Model(inputs, tf.keras.layers.Dense(1)(x))
    assert analysis.resolve_preprocessing(rescaling_model, 'auto') == 'raw'
    assert analysis.resolve_preprocessing(make_model(False), 'auto') == 'rescale'
    assert analysis.resolve_preprocessing(rescaling_model, 'rescale') == 'rescale'


def test_pneumonia_probability_supports_sigmoid_and_softmax():
    assert analysis.pneumonia_probability(np.array([[0.8]]))[0] == pytest.approx(0.8)
    assert analysis.pneumonia_probability(np.array([[0.3, 0.7]]))[0] == pytest.approx(0.7)


def test_analyze_reports_confidence_for_the_predicted_class(xray_path, tmp_path, monkeypatch):
    class Fixed(tf.keras.Model):
        def __init__(self, p):
            super().__init__()
            self.p = p

        def call(self, x, training=False):
            return tf.fill((tf.shape(x)[0], 1), self.p)

    monkeypatch.setattr(analysis, 'compute_gradcam', lambda *a, **k: np.ones((7, 7)))
    out = tmp_path / 'cam.png'
    result = analysis.analyze(xray_path, str(out), model=Fixed(0.1))
    assert result['is_pneumonia'] is False
    assert result['confidence'] == pytest.approx(0.9)
    assert Image.open(out).size == (300, 260)  # overlay keeps the upload's aspect ratio

    result = analysis.analyze(xray_path, str(out), model=Fixed(0.8))
    assert result['is_pneumonia'] is True and result['confidence'] == pytest.approx(0.8)


def test_describe_region():
    heatmap = np.zeros((9, 9))
    heatmap[7, 1] = 1
    assert analysis.describe_region(heatmap) == "lower zone, image-left (patient's right on a PA view)"
    heatmap = np.zeros((9, 9))
    heatmap[0, 8] = 1
    assert analysis.describe_region(heatmap).startswith('upper zone, image-right')
    assert analysis.describe_region(np.zeros((3, 3))) is None


def test_evaluation_metrics():
    m = evaluate.compute_metrics([1, 1, 0, 0], [0.9, 0.4, 0.6, 0.1])
    assert m['confusion'] == {'tp': 1, 'fn': 1, 'fp': 1, 'tn': 1}
    assert m['accuracy'] == 0.5 and m['sensitivity'] == 0.5 and m['specificity'] == 0.5
    assert evaluate.roc_auc([1, 1, 0, 0], [0.9, 0.4, 0.6, 0.1]) == pytest.approx(0.75)
    assert evaluate.roc_auc([1, 0], [0.5, 0.5]) == pytest.approx(0.5)
