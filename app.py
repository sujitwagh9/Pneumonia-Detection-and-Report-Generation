import os
from flask import Flask, render_template, request, redirect, url_for, send_file
from werkzeug.utils import secure_filename
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
from fpdf import FPDF
import google.generativeai as genai
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Flask App Configuration
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Load pre-trained X-ray classification model
model = tf.keras.models.load_model('x-ray-classification.h5')

# Configure Gemini API with API Key from environment
genai.configure(api_key=os.environ['API_KEY'])

def create_pdf_report(prediction, grad_cam_path, detailed_report):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)

    pdf.cell(200, 10, txt="Pneumonia Diagnosis Report", ln=True, align='C')
    pdf.cell(200, 10, txt=f"Diagnosis: {'Pneumonia Detected' if prediction > 0.5 else 'No Pneumonia'}", ln=True)
    
    # Access the prediction value from the numpy array
    prediction_value = prediction[0] if isinstance(prediction, np.ndarray) else prediction
    pdf.cell(200, 10, txt=f"Prediction Confidence: {prediction_value:.2f}", ln=True)

    # Skip Grad-CAM visualization if not generated
    if grad_cam_path:  # Only add Grad-CAM if it exists
        pdf.cell(200, 10, txt="Grad-CAM Visualization:", ln=True)
        pdf.image(grad_cam_path, x=10, y=None, w=190)

    pdf.cell(200, 10, txt="Detailed Diagnosis Report via Gemini API:", ln=True)
    pdf.multi_cell(0, 10, detailed_report)

    pdf_path = os.path.join(app.config['UPLOAD_FOLDER'], "diagnosis_report.pdf")
    pdf.output(pdf_path)
    return pdf_path


def generate_detailed_report(prediction, additional_info):
    try:
        # If prediction is a numpy array, extract its value
        if isinstance(prediction, np.ndarray):
            prediction_value = prediction[0]  # Assuming prediction is a 1D array with one element
        else:
            prediction_value = prediction

        # Initialize the GenerativeModel with the model name
        model = genai.GenerativeModel("gemini-1.5-flash")
        
        # Construct the prompt
        prompt = f"""
        A patient has undergone a chest X-ray analysis.
        Diagnosis: {'Pneumonia Detected' if prediction_value > 0.5 else 'No Pneumonia'}.
        Prediction Confidence: {prediction_value:.2f}.
        Additional Analysis Information: {additional_info}.
        Please generate a detailed medical report with recommendations and insights based on this data.
        """
        
        # Generate the content using the model
        response = model.generate_content(prompt)
        
        # Return the generated report text
        return response.text
    except Exception as e:
        # Return error message if there's an issue
        return f"Error generating report: {str(e)}"

# Flask Routes
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' not in request.files:
            return "No file uploaded", 400
        file = request.files['file']
        if file.filename == '':
            return "No selected file", 400

        file_path = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(file.filename))
        file.save(file_path)

        # Skipping Grad-CAM generation
        # grad_cam_path, prediction = generate_grad_cam(model, file_path)

        # Instead of Grad-CAM, just get prediction
        img = tf.keras.preprocessing.image.load_img(file_path, target_size=(224, 224))
        img_array = tf.keras.preprocessing.image.img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        prediction = model.predict(img_array)[0]

        additional_info = "No additional info available for now."
        detailed_report = generate_detailed_report(prediction, additional_info)

        # Since no Grad-CAM, you can pass a placeholder or skip the grad_cam_path
        grad_cam_path = None  # Or set it to an existing image if needed
        pdf_path = create_pdf_report(prediction, grad_cam_path, detailed_report)

        return redirect(url_for('download_pdf', filename="diagnosis_report.pdf"))

    return render_template('index.html')

@app.route('/download/<filename>')
def download_pdf(filename):
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    if not os.path.exists(file_path):
        return "File not found", 404
    return send_file(file_path, as_attachment=True)



# Run the app
if __name__ == '__main__':
    app.run(debug=True)

