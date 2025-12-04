AttendScan: Secure Facial Verification Attendance System

AttendScan is an end-to-end facial recognition attendance system designed for secure enterprise environments. It features a 3-stage verification pipeline that integrates Liveness Detection (anti-spoofing), Real-Time Emotion Analysis, and Identity Verification using a custom-trained Metric Learning model.

(Note: Replace image_d260aa.png with a screenshot of your actual app GUI if available)

🚀 Key Features

🛡️ Liveness Detection (Anti-Spoofing): A 3D-CNN model analyzes a continuous 24-frame video buffer to distinguish between live subjects and 2D spoof attacks (photos/screens).

🧠 Advanced Face Verification: Uses a custom ResNet50 model trained with Triplet Loss (Metric Learning) to generate 128-dimension "face fingerprints" for high-accuracy matching.

😊 Real-Time Emotion Analysis: Integrates the deepface library to display the user's current emotion (e.g., Happy, Neutral) during the verification process.

💻 User-Friendly GUI: A modern, dark-themed desktop interface built with CustomTkinter.

🛠️ Installation

Prerequisites

Python 3.10 or higher

A working webcam

1. Clone the Repository

git clone [https://github.com/saturnn29/Face-Verification-Attendance-System](https://github.com/saturnn29/Face-Verification-Attendance-System)
cd AttendScan


2. Set Up Virtual Environment (Recommended)

It is highly recommended to use a virtual environment to avoid dependency conflicts.

# Windows
python -m venv .venv
.venv\Scripts\activate

# Mac/Linux
python3 -m venv .venv
source .venv/bin/activate


3. Install Dependencies

This project requires specific libraries like tensorflow-cpu, customtkinter, and deepface.

pip install -r requirements.txt


▶️ Usage

1. Initialize the Database

Before running the app for the first time, initialize the local database files:

python database.py


2. Run the Application

Launch the main GUI:

python main.py


3. How to Use

Start Camera: Click the "Start Camera" button to activate the webcam.

Enrollment: To add a new user, click "Enroll New Employee." Follow the prompts to enter ID, Name, and Department. Look at the camera to capture your face.

Verification: Click "Verify Attendance." The system will:

Check if you are a live person (Anti-Spoofing).

Analyze your emotion.

Verify your identity against the database.

Display the result in the status log.

🏗️ System Architecture

The system processes verification requests through a strictly ordered 3-stage pipeline:

Stage 1: Liveness Gatekeeper

Input: 24-frame video buffer.

Model: 3D-CNN (loaded from livenessmodel.py).

Action: If spoof score > 0.50, the process aborts immediately.

Stage 2: Emotion Analyzer

Input: Current face crop.

Model: deepface (Ensemble).

Action: Logs dominant emotion (e.g., "Happy").

Stage 3: Identity Verification

Input: Current face crop (preprocessed).

Model: Custom ResNet50 Metric Learning model.

Action: Generates embedding -> Calculates Cosine Similarity -> Matches if score > Threshold.

📊 Model Performance

Two verification models were trained and compared for this project:

Model Type

Architecture

AUC Score

Status

Metric Learning

ResNet50 + Triplet Loss

0.80

✅ Selected

Classification

ResNet50 + Softmax

0.70

❌ Discarded

The Metric Learning model was selected for its superior ability to distinguish between known and unknown identities.

📂 Project Structure

main.py: Entry point for the application.

attendance_app.py: Handles the GUI and user interaction logic.

face_recognition_backend.py: The core "brain" containing all ML logic and pipeline orchestration.

livenessmodel.py: Definition of the 3D-CNN architecture.

database.py: Utility script to reset/initialize the database.

models/: Directory containing trained .keras and .h5 model files.

database/: Directory storing employees.json and face_database.pkl.

📄 License

This project is for academic/educational purposes.

Author: [Your Name]
