import os
import re
import time

import pytest

import app as app_module
from conftest import image_bytes

app = app_module.app


@pytest.fixture
def client(monkeypatch):
    app.config.update(TESTING=True, WTF_CSRF_ENABLED=False)
    app_module.limiter.enabled = False
    monkeypatch.setattr(app_module.report, 'generate_report', lambda result: '## Summary\nLooks **fine**.')
    return app.test_client()


def upload(client, data=None, name='xray.png'):
    return client.post('/', data={'file': (data or image_bytes(), name)}, content_type='multipart/form-data')


def wait_until_done(client, run_id, timeout=60):
    deadline = time.time() + timeout
    while time.time() < deadline:
        state = client.get(f'/api/analysis/{run_id}/status').get_json()['state']
        if state in ('done', 'error'):
            return state
        time.sleep(0.2)
    raise AssertionError('analysis did not finish')


def test_full_analysis_flow(client):
    response = upload(client, image_bytes('JPEG', mode='RGB'), 'chest.jpeg')
    assert response.status_code == 302
    run_id = re.search(r'/analysis/([0-9a-f]{32})', response.headers['Location']).group(1)

    assert wait_until_done(client, run_id) == 'done'
    page = client.get(f'/analysis/{run_id}').get_data(as_text=True)
    assert 'Analysis result' in page and '<strong>fine</strong>' in page
    assert 'Model attention concentrated' in page or 'no distinct focus' in page

    assert client.get(f'/analysis/{run_id}/original').mimetype == 'image/jpeg'
    assert client.get(f'/analysis/{run_id}/grad-cam').mimetype == 'image/png'
    pdf = client.get(f'/analysis/{run_id}/pdf')
    assert pdf.mimetype == 'application/pdf' and 'attachment' in pdf.headers['Content-Disposition']

    assert client.post(f'/analysis/{run_id}/delete').status_code == 302
    assert client.get(f'/analysis/{run_id}').status_code == 404


def test_progress_page_while_analysis_runs(client):
    run_id = upload(client).headers['Location'].rsplit('/', 1)[1]
    wait_until_done(client, run_id)
    # Rewind the stored status to check how an in-progress analysis renders
    app_module.set_status(os.path.join(app.config['UPLOAD_FOLDER'], run_id), 'reporting')
    page = client.get(f'/analysis/{run_id}')
    assert page.status_code == 200
    html = page.get_data(as_text=True)
    assert 'Analyzing your X-ray' in html and 'Drafting the written report' in html


def test_missing_report_still_shows_results(client, monkeypatch):
    monkeypatch.setattr(app_module.report, 'generate_report', lambda result: None)
    run_id = upload(client).headers['Location'].rsplit('/', 1)[1]
    assert wait_until_done(client, run_id) == 'done'
    assert 'could not be generated' in client.get(f'/analysis/{run_id}').get_data(as_text=True)


def test_each_upload_gets_its_own_files(client):
    first = upload(client).headers['Location']
    second = upload(client).headers['Location']
    assert first != second


@pytest.mark.parametrize('data,name,status,message', [
    (None, 'scan.txt', 400, 'Unsupported file type'),
    (b'not an image', 'scan.png', 400, 'Could not read that file'),
    (b'0' * (11 * 1024 * 1024), 'scan.png', 413, 'too large'),
])
def test_rejects_bad_uploads(client, data, name, status, message):
    import io
    response = upload(client, io.BytesIO(data) if data else None, name)
    assert response.status_code == status
    assert message in response.get_data(as_text=True)


def test_unknown_or_malformed_ids_are_404(client):
    assert client.get('/analysis/' + '0' * 32).status_code == 404
    assert client.get('/analysis/not-a-uuid').status_code == 404
    assert client.get('/api/analysis/' + 'A' * 32 + '/status').status_code == 404
    run_id = upload(client).headers['Location'].rsplit('/', 1)[1]
    assert client.get(f'/analysis/{run_id}/status.json').status_code == 404


def test_csrf_token_required():
    app.config.update(TESTING=True, WTF_CSRF_ENABLED=True)
    client = app.test_client()
    response = upload(client)
    assert response.status_code == 400 and 'session expired' in response.get_data(as_text=True)

    page = client.get('/').get_data(as_text=True)
    token = re.search(r'name="csrf_token" value="([^"]+)"', page).group(1)
    response = client.post('/', data={'file': (image_bytes(), 'x.png'), 'csrf_token': token},
                           content_type='multipart/form-data')
    assert response.status_code == 302


def test_rate_limit(client):
    app_module.limiter.enabled = True
    app_module.limiter.reset()
    try:
        codes = [upload(client).status_code for _ in range(11)]
    finally:
        app_module.limiter.enabled = False
    assert codes[:10] == [302] * 10 and codes[10] == 429


def test_old_runs_are_cleaned_up(client):
    run_id = upload(client).headers['Location'].rsplit('/', 1)[1]
    wait_until_done(client, run_id)
    directory = os.path.join(app.config['UPLOAD_FOLDER'], run_id)
    old = time.time() - 2 * 3600
    os.utime(directory, (old, old))
    app_module.cleanup_old_runs()
    assert not os.path.exists(directory)
