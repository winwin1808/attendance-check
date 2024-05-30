# Attendance Check System

This project is designed for checking attendance using a camera and face recognition models to detect faces and verify eye and smile detection. The system is implemented in Python 3.11 and uses Flask for the web framework.

## Working Concept

The application captures video input from a webcam and processes each frame to detect faces and smiles. The following steps outline the working concept of the system:

1. **Face Detection**: The system uses Haar cascades to detect faces in each video frame.
2. **Smile Detection**: Simultaneously, it uses another Haar cascade to detect smiles within the detected faces.
3. **Face Recognition**: The detected faces are resized and passed through a K-Nearest Neighbors (KNN) model to identify the person.
4. **Attendance Logging**: Once a face is identified and a smile is detected, the system logs the attendance with the current time.
5. **Model Training**: The application can train the face recognition model using images stored in the `static/faces` directory. This allows adding new users to the system.

## Project Structure

```
attendance-check/
├── Attendance/
│   ├── Attendance-02_23_23.csv
│   ├── Attendance-05_30_24.csv
├── model/
│   ├── face_recognition_model.pkl
│   ├── haarcascade_eye.xml
│   ├── haarcascade_frontalface_default.xml
│   ├── haarcascade_smile.xml
├── static/
│   ├── assets/
│   │   ├── logo.png
│   │   ├── vnuk.png
│   ├── faces/
│   ├── style/
│   │   ├── Welcome.css
├── templates/
│   ├── Home.html
│   ├── WelcomePage.html
├── .gitattributes
├── app.py
├── index.ipynb
├── README.md
```

## How to Run

1. **Install Python 3.11**: Ensure that you have Python 3.11 installed on your system.
2. **Clone the Repository**: Clone this repository to your local machine.
3. **Install Dependencies**: Navigate to the project directory and install the required dependencies using the following command:
    ```bash
    pip install -r requirements.txt
    ```
4. **Run the Application**: Start the Flask application using the following command:
    ```bash
    python app.py
    ```
5. **Access the Application**: Open your web browser and go to `http://localhost:5000` to access the application.

## Demo Images

<div style="display: flex; justify-content: space-between;">
  <img src="/mnt/data/image1.png" alt="Demo Image 1" style="width: 48%;">
  <img src="/mnt/data/image2.png" alt="Demo Image 2" style="width: 48%;">
</div>

## Acknowledgements

This project uses the following resources:
- OpenCV for face and smile detection.
- Flask for the web framework.
- Scikit-learn for the machine learning model.
- Joblib for model serialization.

---