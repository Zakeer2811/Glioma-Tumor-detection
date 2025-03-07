

---

## Project: YOLO-based Image Detection Web App

### Description
This project implements an image detection system using the YOLO (You Only Look Once) deep learning model to identify and classify objects in images. The application utilizes Flask to create a web interface that allows users to upload images for processing. The backend leverages a trained YOLO model in ONNX format to perform object detection and visualize the results. Users can upload an image, and the system will display the detection results on a new page.

Key features include:
- Upload an image through a simple web interface.
- Use a trained YOLOv5 (ONNX) model to predict objects in the uploaded image.
- Visualize the detection results directly in the browser.
- Option to upload another image for detection.

This app serves as an end-to-end object detection solution, from data upload to result display, with a clean and user-friendly interface.

### Project Structure
- `app.py`: Flask app code for serving the web interface and processing images.
- `uploads/`: Directory to temporarily store uploaded images.
- `static/results/`: Directory to store result images after detection.
- `index_html`: HTML template for the main page where users upload images.
- `result_html`: HTML template for displaying the detection results.

### Installation

1. Clone this repository to your local machine.
2. Install the required dependencies by running:
   ```bash
   pip install -r requirements.txt
   ```
3. Make sure you have a trained YOLO model in ONNX format (`best.onnx`). You can use the Ultralytics YOLO repository for training and exporting the model.

4. Run the Flask app:
   ```bash
   python app.py
   ```

5. Open your browser and navigate to `http://127.0.0.1:5000/` to start uploading images and see the object detection results.

### Requirements

- Python 3.12
- Flask
- OpenCV
- ultralytics (for YOLOv11)
- ONNX Runtime

---


