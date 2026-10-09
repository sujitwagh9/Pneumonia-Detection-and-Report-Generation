from types import SimpleNamespace

from PIL import Image

import report

RESULT = {'probability': 0.87, 'is_pneumonia': True, 'confidence': 0.87, 'threshold': 0.5,
          'attention_region': "lower zone, image-left (patient's right on a PA view)"}


def test_pdf_supports_unicode_markdown_and_missing_report(tmp_path):
    img = tmp_path / 'x.png'
    Image.new('RGB', (400, 500), 'gray').save(img)
    text = "## Summary\n**Bold** finding – SpO₂ ≥ 94 % · “quoted”\n\n- one\n- two\n\n<script>x</script>"
    for body in (text, None):
        out = tmp_path / 'r.pdf'
        report.build_pdf(RESULT, body, str(img), str(img), str(out))
        assert out.read_bytes().startswith(b'%PDF')


def test_markdown_output_escapes_html():
    html = report.markdown_to_html("**hi** <img src=x onerror=alert(1)>")
    assert '<strong>hi</strong>' in html and '<img' not in html


def test_generate_report_without_key_returns_none(monkeypatch):
    monkeypatch.setenv('GEMINI_API_KEY', '')
    monkeypatch.setenv('API_KEY', '')
    assert report.generate_report(RESULT) is None


def test_generate_report_uses_prompt_and_handles_errors(monkeypatch):
    calls = []

    def generate_content(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(text='## Summary\nok')

    client = SimpleNamespace(models=SimpleNamespace(generate_content=generate_content))
    monkeypatch.setattr(report, '_client', lambda: client)
    assert report.generate_report(RESULT) == '## Summary\nok'
    assert '0.870' in calls[0]['contents'] and '87.0%' in calls[0]['contents']
    assert calls[0]['model'] == report.GEMINI_MODEL

    def fail(**kwargs):
        raise RuntimeError('quota exceeded')

    client.models.generate_content = fail
    assert report.generate_report(RESULT) is None


def test_legacy_api_key_variable_is_accepted(monkeypatch):
    monkeypatch.setenv('GEMINI_API_KEY', '')
    monkeypatch.setenv('API_KEY', 'legacy-key')
    assert report._client() is not None
