import sys
from PyQt6.QtWidgets import QApplication, QWidget, QLabel, QVBoxLayout, QPushButton, QStackedWidget, QLineEdit
from PyQt6.QtCore import QThread, pyqtSignal
import tensorflow as tf
import cv2
from deepface import DeepFace
from mtcnn import MTCNN
import google.generativeai as genai
#import audio_tester
import subprocess
import multiprocessing
import os


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
    genai.configure(api_key="YOUR_API_KEY")  # Replace "YOUR_API_KEY" with your actual API key. (generated from link above)
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
        emotion_percentages = process_video("video.mp4")
        self.screen2.set_emotion_data(emotion_percentages)
        self.stack.setCurrentIndex(1)


class EmotionWorker(QThread):
    finished = pyqtSignal()
    progress = pyqtSignal(int)
    image = pyqtSignal(list)
    
    def __init__(self):
        QThread.__init__(self)
        
    def run(self):
        deepface_result = None
        gemini_result = None 
         
        audio_file = "video.mp4"
        self.progress.emit(1)
        
        with multiprocessing.Pool(processes=1) as pool:
            deepface_result = pool.apply(process_video, args=(audio_file,))
            print(str(deepface_result))
            
        self.progress.emit(50)
        emotion = str(deepface_result.get(audio_file, None))
        sentence = str("Your flight is delayed")
        if emotion is not None:
            gemini_result = generate_gemini_content(sentence, emotion)
        
        self.finished.emit()
        

class Screen2(QWidget):
    #Note: when data is passed from one screen to another, need to have the screen the data needs to 'go to' passed in as an arugument here**
    def __init__(self, stack, screen3):
        super().__init__()
        self.stack = stack
        self.screen3 = screen3
        layout = QVBoxLayout()
        self.label = QLabel("Emotion detection is now complete!\n")
        self.label1 = QLabel("")
        self.option_1_button = QPushButton("Select option 1")
        self.option_1_button.clicked.connect(self.go_to_screen3_completebreakdown)

        self.label2 = QLabel("")
        option_2_button = QPushButton("Select option 2")
        option_2_button.clicked.connect(self.go_to_screen3_maxemotion)
        
        layout.addWidget(self.label)
        layout.addWidget(self.label1)
        layout.addWidget(self.option_1_button)
        layout.addWidget(self.label2)
        layout.addWidget(option_2_button)
        self.setLayout(layout)

    def set_emotion_data(self, emotion_data):
        self.emotion_data = emotion_data #defining for later use**
        # Update the label or UI with emotion percentages
        emotion_text = "  ".join([f"{emotion}: {percent:.2f}%" for emotion, percent in emotion_data.items()])
        self.label1.setText(f"Option 1 - use complete emotion breakdown:\n{emotion_text}")
        overall_emotion = max(emotion_data, key=emotion_data.get)
        self.label2.setText(f"Option 2 - use maximum overall emotion:\n{overall_emotion}")
        print("Overall Emotion for the Video:", overall_emotion)

    def go_to_screen3_completebreakdown(self):
        # Switch to screen 3 after selecting an option
        #pass full emotion breakdown to screen 3 in this case
        emotion_text_gemini = "\n".join([f"{emotion}: {percent:.2f}%" for emotion, percent in self.emotion_data.items()])
        self.screen3.set_emotions_forgemini(emotion_text_gemini)
        self.stack.setCurrentIndex(2)

    def go_to_screen3_maxemotion(self):
        # Switch to screen 3 after selecting an option
        #pass the overall emotion to screen 3 in this case
        max_emotion = max(self.emotion_data, key=self.emotion_data.get)
        self.screen3.set_emotions_forgemini(max_emotion)
        self.stack.setCurrentIndex(2)

class Screen3(QWidget):
    def __init__(self, stack):
        super().__init__()
        self.stack = stack
        layout = QVBoxLayout()
        self.label = QLabel("Generating emotionally altered text screen. Please enter sentence in input box below then press the 'generate emotionally altered announcement' button. You may alter the text in the input box and re-generate the emotionally altered text")
        #layout.addWidget(label)
        self.labelemotionbreakdown = QLabel("")
        self.input_box = QLineEdit()
        self.submit_button = QPushButton("Generate emotionally altered announcement")
        self.labelgeneratedtext = QLabel("")
        self.submit_button.clicked.connect(self.gemini_connection)

        layout.addWidget(self.label)
        layout.addWidget(self.labelemotionbreakdown)
        layout.addWidget(self.labelgeneratedtext)
        layout.addWidget(self.input_box)
        layout.addWidget(self.submit_button)

        self.setLayout(layout)
    def set_emotions_forgemini(self, emotion_topass_togemini):
        self.emotion_forgemini = emotion_topass_togemini
        self.labelemotionbreakdown.setText = (f"Using emotion breakdown:\n{self.emotion_forgemini}")

    def gemini_connection(self):
        user_input = self.input_box.text()
        resp = generate_gemini_content(user_input, self.emotion_forgemini)
        self.labelgeneratedtext.setText(f"Generated content: {resp}")

class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setGeometry(200,200, 400, 400)
        self.setStyleSheet('background-color:lightblue; color:black; font-weight: bold; font-size: 16px')
        self.stack = QStackedWidget()
        self.screen3 = Screen3(self.stack)
        self.screen2 = Screen2(self.stack, self.screen3)
        self.screen1 = Screen1(self.stack, self.screen2)
        
        self.worker = EmotionWorker()
        

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
