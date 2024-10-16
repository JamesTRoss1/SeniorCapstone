#import tensorflow as tf
import cv2
from deepface import DeepFace
from mtcnn import MTCNN
import google.generativeai as genai
#import audio_tester
import subprocess

# Function to calculate the variance of Laplacian to measure image blurriness
def blur_measure(image):
    """
    Calculates the variance of Laplacian to measure image blurriness

    :param numpy.ndarray image: Input image
    :return: Variance of Laplacian
    :rtype: float
    """
    return cv2.Laplacian(image, cv2.CV_64F).var()

def convert_mp4_to_wav(mp4_file):
    """Converts an MP4 file to WAV format without deleting the original.

    Args:
        mp4_file (str): The path to the MP4 file.

    Returns:
        str: The path to the newly created WAV file.
    """

    wav_file = os.path.splitext(mp4_file)[0] + ".wav"
    command = ["ffmpeg", "-i", mp4_file, "-acodec", "pcm_s16le", "-f", "wav", wav_file]

    try:
        subprocess.run(command, check=True)
        return wav_file
    except subprocess.CalledProcessError as e:
        print(f"Error converting MP4 to WAV: {e}")
        return None

# Function to determine emotions of a collection of faces in a frame
def emotion_faces(faces, video_path):
    preds = DeepFace.analyze(faces, actions=['emotion'], detector_backend="mtcnn", enforce_detection=False)
    emotions = {}
    weights = {'sad': 1, 'angry': 1, 'surprise': 1, 'fear': 1, 'happy': 1, 'disgust': 1, 'neutral': 1}

    for pred in preds:
        if pred["face_confidence"] < 0.8:
            pass
        else:
            emotions = {k: emotions.get(k, 0) + pred['emotion'].get(k, 0) for k in set(emotions) | set(pred['emotion'])}

    emotions = {k: emotions.get(k, 0) * weights.get(k, 0) for k in set(emotions) & set(weights)}

    if not emotions:
        #wav_file = convert_mp4_to_wav(video_path)

        #if wav_file is not None:
            # file_name, dnn_output, gemini_output = audio_tester.__emotion_audio__(wav_file)
            # gemini_output = str(gemini_output).strip().lower()
            # if gemini_output in weights.keys():
            #     return gemini_output, {}

        return "Unknown", {}

    return max(emotions, key=emotions.get), emotions

# Function to generate content using Google Gemini
def generate_gemini_content(sentence, emotion):
    #can generate key here: https://aistudio.google.com/app/apikey
    genai.configure(api_key="AIzaSyDLhkSOGqSLLDAKwrPbOY7D743wka7YDQs")  # Replace "YOUR_API_KEY" with your actual API key. (generated from link above)

    for m in genai.list_models():
        if 'generateContent' in m.supported_generation_methods:
            print(m.name)
    print()

    model = genai.GenerativeModel('gemini-1.5-pro-latest')
    safe = [
        {
            "category": "HARM_CATEGORY_DANGEROUS",
            "threshold": "BLOCK_NONE",
        },
        {
            "category": "HARM_CATEGORY_HARASSMENT",
            "threshold": "BLOCK_NONE",
        },
        {
            "category": "HARM_CATEGORY_HATE_SPEECH",
            "threshold": "BLOCK_NONE",
        },
        {
            "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
            "threshold": "BLOCK_NONE",
        },
        {
            "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
            "threshold": "BLOCK_NONE",
        },
    ]

    prompt = f"Prompt: Please alter the following sentence, considering the specified crowd sentiment and aiming to evoke a more positive emotional response. Aim to maintain the original meaning while adjusting the tone, vocabulary, or context as needed. Parameters: Sentence: {sentence} Crowd Sentiment: {emotion} Instructions: Emotional Elevation: If the crowd sentiment is negative, aim to elevate the emotional tone of the response.  This might involve: Offering hope or optimism: Suggesting positive possibilities or future outcomes. Providing comfort or reassurance: Acknowledging the negative emotions and offering support or understanding. Shifting the focus: Directing attention towards more positive aspects of the situation or related topics. Contextual Relevance: Ensure the response is relevant to the input sentence and the overall theme or topic. Nuance and Sensitivity: Be mindful of the specific context and nuances of the situation to avoid inappropriate or insensitive responses."

    response = model.generate_content(prompt, safety_settings=safe)

    return response.text

import sys
from PyQt6.QtWidgets import QApplication, QWidget, QLabel, QVBoxLayout,  QPushButton, QMainWindow, QLineEdit, QTextEdit
from PyQt6.QtCore import QThread, pyqtSignal
import os

class EmotionWorker(QThread):
    finished = pyqtSignal()
    progress = pyqtSignal(int)
    emotion = pyqtSignal(str)
    def run(self):
        output = {}
        dir_name = "audio"

        for root, dirs, files in os.walk(dir_name):
            for count, file in enumerate(files):
                if file.endswith(".mp4"):
                    # Load the video
                    video_path = root + "/" + file
                    print(str(video_path))
                    video_capture = cv2.VideoCapture(video_path)

                    # Set frame rate
                    fps = 23
                    frame_count = 0

                    # Create MTCNN detector
                    detector = MTCNN()

                    # Initialize emotion list for the entire video
                    video_emotions = []

                    # Iterate through frames
                    while True:
                        ret, frame = video_capture.read()
                        if not ret:
                            break

                        if frame_count % fps == 0:
                            print(f"Processing frame {frame_count}")

                            # Detect faces
                            faces = detector.detect_faces(frame)
                            cropped_faces = []

                            # Extract faces and append to cropped_faces list
                            for i, face in enumerate(faces):
                                x, y, w, h = face['box']
                                cropped_face = frame[y:y + h, x:x + w]
                                cropped_faces.append(cropped_face)

                                # Save detected face as JPEG file
                                emotion, _ = emotion_faces(cropped_face, video_path)
                                # if emotion:
                                #     cv2.imwrite(f"face_{frame_count}_{i+1}_{emotion}.jpg", cropped_face)

                            # Check if any faces were detected
                            if len(cropped_faces) == 0:
                                print("No faces detected in this frame.")
                                break
                                continue  # Skip this frame if no faces are detected

                            # Output the list of cropped faces
                            for i, face in enumerate(cropped_faces):
                                emotion, _ = emotion_faces(face, video_path)
                                if emotion:
                                    video_emotions.append(emotion)
                        
                        frame_count += 1
                    print("Done Processing " + str(video_path))
                    # Close video capture
                    print(video_capture.get(cv2.CAP_PROP_FPS))
                    video_capture.release()
                    cv2.destroyAllWindows()

                    # Output overall emotion for the video
                    if video_emotions:

                        overall_emotion = max(set(video_emotions), key=video_emotions.count)
                        print("Overall Emotion for the Video:", overall_emotion)
                        self.emotion.emit(str(overall_emotion))

                        output.update({video_path : overall_emotion})
                        self.progress.emit(count)

                        print(str(output))
                        self.finished.emit()


class QMainWindow(QWidget):
    def __init__(self):
        super().__init__()
        layout = QVBoxLayout()
        self.message = "Welcome to the emotion detection message manipulator.\nPress start to begin detecting emotions. Enter your sentence you would like manipulated and press submit when ready.\n"
        self.label = QTextEdit(self.message)
        self.label.setReadOnly(True)
        layout.addWidget(self.label)

        self.setLayout(layout)
        self.input_box= QLineEdit()
        self.input_box.setStyleSheet('background-color:white')
        self.setWindowTitle("Emotion Detection Message Manipulator")
        self.setGeometry(200,200, 400, 400)
        self.setStyleSheet('background-color:blue; color:black; font-weight: bold; font-size: 16px')
        self.worker = EmotionWorker()
        self.worker.finished.connect(self.worker_finished)
        self.worker.progress.connect(self.update_progress)
        self.worker.emotion.connect(self.set_emotion)
        self.emotion = None


        self.start_button = QPushButton("Start")
        self.submit_button = QPushButton("Submit")
        self.start_button.clicked.connect(self.start_worker)
        self.submit_button.clicked.connect(self.submit_input)
        layout.addWidget(self.start_button)
        layout.addWidget(self.submit_button)
        layout.addWidget(self.input_box)
        self.user_input = None

    def start_worker(self):
        self.worker.start()

    def update_progress(self, value):
        self.label.append(f"Progress: {value}%")
        

    def worker_finished(self):
        self.label.append("finished!")
    

        



    def set_emotion(self, emotion):
        self.label.append("Emotion Detection is complete: recreating your sentence with emotional sentiment.")
        while self.user_input is None:
            print("stuck here")
        resp = generate_gemini_content(self.user_input, emotion)
        self.label.append(f"Your new sentence is: \n\"{resp}\" \ndue to the rooms emotion being: {emotion}")
        print("done")
    def submit_input(self):
        self.user_input = self.input_box.text()
        self.label.append(f"You input \" {self.user_input} \" as your sentence.")

if __name__ == "__main__":
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)
    window = QMainWindow()

    window.show()
    sys.exit(app.exec())