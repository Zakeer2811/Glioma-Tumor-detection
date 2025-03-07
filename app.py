import os
import cv2
from flask import Flask, request, redirect, url_for, render_template_string, send_from_directory
from ultralytics import YOLO

# Configuration
UPLOAD_FOLDER = 'uploads'
RESULT_FOLDER = 'static/results'
MODEL_PATH = r'C:\D_DRIVE\preprocessed2\mini1\train5\weights\best.onnx'

# Create the Flask app
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['RESULT_FOLDER'] = RESULT_FOLDER

# Load the YOLO ONNX model using Ultralytics
model = YOLO(MODEL_PATH)

# HTML templates with CSS styling
index_html = '''
<!doctype html>
<html>
<head>
    <title>Image Detection</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            background: #f2f2f2;
            margin: 0;
            padding: 0;
            text-align: center;
        }
        .container {
            width: 80%;
            margin: auto;
            padding: 20px;
            background: #fff;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            margin-top: 50px;
            border-radius: 8px;
        }
        h1 {
            color: #333;
        }
        form {
            margin-top: 20px;
        }
        input[type="file"] {
            padding: 10px;
            margin-bottom: 20px;
        }
        input[type="submit"] {
            padding: 10px 20px;
            background: #007bff;
            color: #fff;
            border: none;
            border-radius: 4px;
            cursor: pointer;
        }
        input[type="submit"]:hover {
            background: #0056b3;
        }
        .footer {
            margin-top: 30px;
            font-size: 0.9em;
            color: #666;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>Upload an Image for Detection</h1>
        <form method="post" enctype="multipart/form-data">
            <input type="file" name="file" accept="image/*">
            <br>
            <input type="submit" value="Upload">
        </form>
        <div class="footer">
            <p>&copy; 2025 Your Company Name</p>
        </div>
    </div>
</body>
</html>
'''

result_html = '''
<!doctype html>
<html>
<head>
    <title>Detection Result</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            background: #f9f9f9;
            margin: 0;
            padding: 0;
            text-align: center;
        }
        .container {
            width: 90%;
            margin: auto;
            padding: 20px;
            background: #fff;
            margin-top: 50px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        h1 {
            color: #333;
        }
        img {
            max-width: 80%;
            height: auto;
            border: 1px solid #ddd;
            border-radius: 4px;
            padding: 5px;
        }
        a {
            display: inline-block;
            margin-top: 20px;
            padding: 10px 20px;
            background: #28a745;
            color: #fff;
            text-decoration: none;
            border-radius: 4px;
        }
        a:hover {
            background: #218838;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>Detection Result</h1>
        <img src="{{ url_for('static', filename='results/' + filename) }}" alt="Result Image">
        <br><br>
        <a href="{{ url_for('index') }}">Upload Another Image</a>
    </div>
</body>
</html>
'''

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Check for file in request
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)
        if file:
            # Save the uploaded file
            filename = file.filename
            upload_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(upload_path)

            # Run detection on the uploaded image
            results = model.predict(source=upload_path, imgsz=640)
            
            # Force manual saving of the result image:
            # Get the plotted result as a numpy array
            img_result = results[0].plot()
            # Define the output path (static/results folder with same filename)
            output_path = os.path.join(app.config['RESULT_FOLDER'], filename)
            cv2.imwrite(output_path, img_result)

            # Redirect to the result page
            return redirect(url_for('result', filename=filename))
    return render_template_string(index_html)

@app.route('/result/<filename>')
def result(filename):
    return render_template_string(result_html, filename=filename)

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

def create_folders():
    for folder in [UPLOAD_FOLDER, RESULT_FOLDER]:
        if not os.path.exists(folder):
            os.makedirs(folder)

if __name__ == '__main__':
    create_folders()
    app.run(debug=True)
