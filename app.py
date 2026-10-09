"""Web app: upload a chest X-ray, get a pneumonia prediction, Grad-CAM heatmap and a PDF report."""
import json
import logging
import os
import secrets
import shutil
import time
import uuid
from concurrent.futures import ThreadPoolExecutor

from dotenv import load_dotenv

# Load .env before importing modules that read configuration at import time
load_dotenv()

from flask import Flask, abort, jsonify, redirect, render_template, request, send_file, url_for  # noqa: E402
from flask_limiter import Limiter  # noqa: E402
from flask_limiter.util import get_remote_address  # noqa: E402
from flask_wtf.csrf import CSRFError, CSRFProtect  # noqa: E402
from markupsafe import Markup  # noqa: E402
from PIL import Image, UnidentifiedImageError  # noqa: E402
from werkzeug.exceptions import RequestEntityTooLarge  # noqa: E402

import analysis  # noqa: E402
import report  # noqa: E402

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(name)s: %(message)s')
log = logging.getLogger(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

app = Flask(__name__)
app.config.update(
    SECRET_KEY=os.environ.get('SECRET_KEY') or secrets.token_hex(32),
    UPLOAD_FOLDER=os.path.abspath(os.environ.get('UPLOAD_FOLDER', os.path.join(BASE_DIR, 'uploads'))),
    MAX_CONTENT_LENGTH=10 * 1024 * 1024,  # 10 MB
    RETENTION_MINUTES=int(os.environ.get('RETENTION_MINUTES', '60')),
    RATELIMIT_STORAGE_URI=os.environ.get('RATELIMIT_STORAGE_URI', 'memory://'),
)
if not os.environ.get('SECRET_KEY'):
    log.warning("SECRET_KEY is not set; using a random key (forms break across restarts and workers)")
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

csrf = CSRFProtect(app)
limiter = Limiter(get_remote_address, app=app)
executor = ThreadPoolExecutor(max_workers=int(os.environ.get('ANALYSIS_WORKERS', '2')))

# Fail fast with setup instructions if the model is missing; requests show the same message
try:
    analysis.get_model()
except analysis.ModelNotAvailable as e:
    log.error("%s", e)

STEPS = ['queued', 'analyzing', 'reporting', 'pdf', 'done']
STEP_LABELS = {
    'queued': 'Waiting in queue',
    'analyzing': 'Classifying and computing Grad-CAM',
    'reporting': 'Drafting the written report',
    'pdf': 'Building the PDF',
}


@app.context_processor
def template_globals():
    return {'retention_minutes': app.config['RETENTION_MINUTES']}


# ---------- Run storage (one folder per analysis) ----------

def run_dir(run_id):
    try:
        if uuid.UUID(hex=run_id).hex != run_id:
            abort(404)
    except ValueError:
        abort(404)
    path = os.path.join(app.config['UPLOAD_FOLDER'], run_id)
    if not os.path.isdir(path):
        abort(404)
    return path


def write_json(path, data):
    tmp = f"{path}.tmp"
    with open(tmp, 'w') as f:
        json.dump(data, f)
    os.replace(tmp, path)  # atomic, so the status endpoint never reads a half-written file


def read_json(path):
    try:
        with open(path) as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def set_status(directory, state, error=None):
    write_json(os.path.join(directory, 'status.json'), {'state': state, 'error': error})


def find_original(directory):
    for ext in ALLOWED_EXTENSIONS:
        path = os.path.join(directory, f'original.{ext}')
        if os.path.exists(path):
            return path
    return None


def cleanup_old_runs():
    """Delete uploaded X-rays and reports older than RETENTION_MINUTES."""
    cutoff = time.time() - app.config['RETENTION_MINUTES'] * 60
    root = app.config['UPLOAD_FOLDER']
    for name in os.listdir(root):
        path = os.path.join(root, name)
        if os.path.isdir(path) and os.path.getmtime(path) < cutoff:
            shutil.rmtree(path, ignore_errors=True)


def process_run(directory):
    """Background job: classify, explain, write the report and build the PDF."""
    try:
        set_status(directory, 'analyzing')
        original = find_original(directory)
        gradcam = os.path.join(directory, 'grad_cam.png')
        result = analysis.analyze(original, gradcam)

        set_status(directory, 'reporting')
        report_text = report.generate_report(result)

        set_status(directory, 'pdf')
        report.build_pdf(result, report_text, original, gradcam, os.path.join(directory, 'diagnosis_report.pdf'))

        write_json(os.path.join(directory, 'result.json'),
                   {**result, 'report': report_text, 'created_at': time.time()})
        set_status(directory, 'done')
    except Exception:
        log.exception("Analysis failed for %s", directory)
        set_status(directory, 'error', "The analysis failed. Please try again with a different image.")


# ---------- Routes ----------

def render_index(error=None, status=200):
    model_error = None
    try:
        analysis.get_model()
    except analysis.ModelNotAvailable as e:
        model_error = str(e)
    return render_template('index.html', error=error, model_error=model_error), status


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.errorhandler(RequestEntityTooLarge)
def file_too_large(e):
    return render_index("File is too large. The maximum size is 10 MB.", 413)


@app.errorhandler(429)
def rate_limited(e):
    return render_index("Too many uploads. Please wait a minute and try again.", 429)


@app.errorhandler(CSRFError)
def csrf_failed(e):
    return render_index("Your session expired. Please reload the page and try again.", 400)


@app.route('/', methods=['GET'])
def index():
    return render_index()


@app.route('/', methods=['POST'])
@limiter.limit(os.environ.get('UPLOAD_RATE_LIMIT', '10 per minute'))
def upload():
    cleanup_old_runs()
    try:
        analysis.get_model()
    except analysis.ModelNotAvailable:
        return render_index("The model is not installed on this server.", 503)

    file = request.files.get('file')
    if file is None or file.filename == '':
        return render_index("Please choose a chest X-ray image to upload.", 400)
    if not allowed_file(file.filename):
        return render_index("Unsupported file type. Upload a PNG or JPEG image.", 400)

    # Each analysis gets its own folder so concurrent users never see each other's files
    run_id = uuid.uuid4().hex
    directory = os.path.join(app.config['UPLOAD_FOLDER'], run_id)
    os.makedirs(directory)
    ext = file.filename.rsplit('.', 1)[1].lower()
    path = os.path.join(directory, f'original.{ext}')
    file.save(path)

    try:
        with Image.open(path) as img:
            img.verify()
    except (UnidentifiedImageError, OSError):
        shutil.rmtree(directory, ignore_errors=True)
        return render_index("Could not read that file as an image. Please upload a valid chest X-ray.", 400)

    set_status(directory, 'queued')
    executor.submit(process_run, directory)
    return redirect(url_for('view_run', run_id=run_id))


@app.route('/analysis/<run_id>')
def view_run(run_id):
    directory = run_dir(run_id)
    status = read_json(os.path.join(directory, 'status.json')) or {'state': 'queued'}
    if status['state'] == 'error':
        return render_index(status['error'], 500)
    if status['state'] != 'done':
        return render_template('processing.html', run_id=run_id, steps=STEPS, labels=STEP_LABELS,
                               state=status['state'])

    result = read_json(os.path.join(directory, 'result.json'))
    return render_template(
        'result.html',
        run_id=run_id,
        result=result,
        report_html=Markup(report.markdown_to_html(result['report'])) if result['report'] else None,
    )


@app.route('/api/analysis/<run_id>/status')
def run_status(run_id):
    status = read_json(os.path.join(run_dir(run_id), 'status.json')) or {'state': 'queued', 'error': None}
    return jsonify(status)


@app.route('/analysis/<run_id>/delete', methods=['POST'])
def delete_run(run_id):
    shutil.rmtree(run_dir(run_id), ignore_errors=True)
    return redirect(url_for('index'))


REPORT_FILES = {
    'grad-cam': ('grad_cam.png', False),
    'pdf': ('diagnosis_report.pdf', True),
}


@app.route('/analysis/<run_id>/<kind>')
def run_file(run_id, kind):
    directory = run_dir(run_id)
    if kind == 'original':
        path, as_attachment = find_original(directory), False
    elif kind in REPORT_FILES:
        filename, as_attachment = REPORT_FILES[kind]
        path = os.path.join(directory, filename)
    else:
        abort(404)
    if not path or not os.path.exists(path):
        abort(404)
    response = send_file(path, as_attachment=as_attachment)
    response.headers['Cache-Control'] = 'private, no-store'
    return response


if __name__ == '__main__':
    app.run(
        host=os.environ.get('HOST', '127.0.0.1'),
        port=int(os.environ.get('PORT', '5000')),
        debug=os.environ.get('FLASK_DEBUG') == '1',
    )
