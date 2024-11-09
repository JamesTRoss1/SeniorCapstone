import sys
from PyQt6.QtWidgets import QApplication, QWidget, QLabel, QVBoxLayout, QPushButton, QStackedWidget
from PyQt6.QtCore import QThread, pyqtSignal
import tensorflow as tf
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
# Function to determine emotions of a collection of faces in a frame
def emotion_faces(faces):
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
        return "Unknown", {}
    return max(emotions, key=emotions.get), emotions
# Function to generate content using Google Gemini
def generate_gemini_content(sentence, emotion):
    #can generate key here: https://aistudio.google.com/app/apikey
    genai.configure(api_key="AIzaSyDAVBCGTHDB4gos1MO7pU-6sOccFakqEc0")  # Replace "YOUR_API_KEY" with your actual API key. (generated from link above)
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
    prompt = f"I have a sentence: \"{sentence}\". The current emotional sentiment of the environment is {emotion}. Can you rewrite the sentence to better mediate this emotional sentiment while conveying the same core message? Output only the best option, which can be multiple sentences long, that will best improve the emotion of the environment. Do not include an explanation or more than one option."
    response = model.generate_content(prompt, safety_settings=safe)
    return response.text
# Load the video
def process_video(video_path):
    video_capture = cv2.VideoCapture(video_path)
    #video_capture = cv2.VideoCapture(1)
    # Set frame rate
    fps = 2
    frame_count = 0
    # Create MTCNN detector
    detector = MTCNN()
    # Initialize emotion list for the entire video
    video_emotions = []
    # Iterate through frames
    while video_capture.isOpened():
        ret, frame = video_capture.read()
        if not ret:
            break
        # Capture frame every 'fps' seconds
        frame_count += 1
        if frame_count % int(video_capture.get(cv2.CAP_PROP_FPS) * fps) == 0:
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
                emotion, _ = emotion_faces(cropped_face)
                if emotion:
                    cv2.imwrite(f"face_{frame_count}_{i+1}_{emotion}.jpg", cropped_face)
            # Check if any faces were detected
            if len(cropped_faces) == 0:
                print("No faces detected in this frame.")
                continue  # Skip this frame if no faces are detected
            # Output the list of cropped faces
            for i, face in enumerate(cropped_faces):
                emotion, _ = emotion_faces(face)
                if emotion:
                    video_emotions.append(emotion)
    # Close video capture
    video_capture.release()
    cv2.destroyAllWindows()
    # Output emotion composition for the video
    if video_emotions:
        emotion_counts = {emotion: video_emotions.count(emotion) for emotion in set(video_emotions)}
        total_emotions = len(video_emotions)

        emotion_percentages = {}
        print("Emotion breakdown:")
        for emotion, count in emotion_counts.items():
            percentage = (count / total_emotions) * 100
            emotion_percentages[emotion] = percentage
            print(f"{emotion}: {percentage:.2f}%")

        return emotion_percentages
        # if overall_emotion == 'happy':
        #     overall_emotion = 'neutral'
        # if overall_emotion == 'sad':
        #     overall_emotion = 'happy'
        # sentence = input("Enter a sentence: ")
        # generated_sentence = generate_gemini_content(sentence, overall_emotion)
        # print("Generated Sentence:", generated_sentence)
    else:
        print("No emotions detected in the video.")

class Screen1(QWidget):
    def __init__(self, stack, screen2):
        super().__init__()
        self.stack = stack
        self.screen2 = screen2

        layout = QVBoxLayout()
        label = QLabel("Welcome to the Adaptive Announcement System! Please click start to begin crowd emotion detection.")
        start_button = QPushButton("Start")
        start_button.clicked.connect(self.start_emotion_detection)

        layout.addWidget(label)
        layout.addWidget(start_button)
        self.setLayout(layout)

    def start_emotion_detection(self):
        emotion_percentages = process_video("video_3.mp4")
        self.screen2.set_emotion_data(emotion_percentages)
        self.stack.setCurrentIndex(1)

class Screen2(QWidget):
    def __init__(self, stack):
        super().__init__()
        self.stack = stack
        layout = QVBoxLayout()
        self.label = QLabel("Emotion detection is now complete!\n")
        self.label1 = QLabel("")
        self.option_1_button = QPushButton("Select option 1")
        self.option_1_button.clicked.connect(self.go_to_screen3)

        self.label2 = QLabel("")
        option_2_button = QPushButton("Select option 2")
        option_2_button.clicked.connect(self.go_to_screen3)
        
        layout.addWidget(self.label)
        layout.addWidget(self.label1)
        layout.addWidget(self.option_1_button)
        layout.addWidget(self.label2)
        layout.addWidget(option_2_button)
        self.setLayout(layout)

    def set_emotion_data(self, emotion_data):
        # Update the label or UI with emotion percentages
        emotion_text = "\n".join([f"{emotion}: {percent:.2f}%" for emotion, percent in emotion_data.items()])
        self.label1.setText(f"Option 1 - use complete emotion breakdown:\n{emotion_text}")
        overall_emotion = max(emotion_data, key=emotion_data.get)
        self.label2.setText(f"Option 2 - use maximum overall emotion:\n{overall_emotion}")
        print("Overall Emotion for the Video:", overall_emotion)

    def go_to_screen3(self):
        # Switch to screen 3 after selecting an option
        self.stack.setCurrentIndex(2)

class Screen3(QWidget):
    #TODO: Add in to link the screen's text output with gemini 
    def __init__(self, stack):
        super().__init__()
        layout = QVBoxLayout()
        label = QLabel("Temporary screen for Google Gemini output")
        layout.addWidget(label)
        self.setLayout(layout)

class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.stack = QStackedWidget()
        self.screen2 = Screen2(self.stack)
        self.screen1 = Screen1(self.stack, self.screen2)
        self.screen3 = Screen3(self.stack)

        # self.screen1 = Screen1(self.stack)
        # self.screen2 = Screen2(self.stack)
        # self.screen3 = Screen3(self.stack)

        self.stack.addWidget(self.screen1)
        self.stack.addWidget(self.screen2)
        self.stack.addWidget(self.screen3)

        layout = QVBoxLayout()
        layout.addWidget(self.stack)
        self.setLayout(layout)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
