import sys
from PyQt6.QtWidgets import QApplication, QWidget, QLabel, QVBoxLayout, QPushButton, QStackedWidget, QLineEdit, QRadioButton
from PyQt6.QtCore import QThread, pyqtSignal, QObject, QSize
from PyQt6.QtGui import QIcon
import tensorflow as tf
import cv2
from deepface import DeepFace
from mtcnn import MTCNN
import google.generativeai as genai
#import audio_tester
import subprocess
import multiprocessing
import os
import math 
import time

audio_file = "audio/video.mp4"
frame = None 

def detect_blur_laplacian(img, threshold=30):
    """Detects image blur using the Laplacian filter.

    Args:
        img: The input image.
        threshold: The variance threshold for determining blurriness.

    Returns:
        True if the image is blurry, False otherwise.
    """

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    fm = cv2.Laplacian(gray, cv2.CV_64F).var()
    print(str(fm))
    return True if fm < threshold else False


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
    preds = DeepFace.analyze(faces, actions=['emotion', 'age'], detector_backend="mtcnn", enforce_detection=False)
    emotions = {}
    weights = {'sad': 1, 'angry': 1, 'surprise': 1, 'fear': 1, 'happy': 1, 'disgust': 1, 'neutral': 1}
    for pred in preds:
        #ignore the prediction if the confidence is less than 0.8 or the face is less than 13 years old
        if pred["face_confidence"] < 0.7 or pred["age"] < 13:
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
    genai.configure(api_key="AIzaSyAdwzIQZXJx48UyP10eXfUdWn2qlZSu6Os")  # Replace "YOUR_API_KEY" with your actual API key. (generated from link above)
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

#captures 10 second video and saves to .mp4 file - can modify this as needed
def capture_video(video_path, duration=10):
    # Open webcam, 0 is the default webcam
    video_capture = cv2.VideoCapture(0)
    # codec and create VideoWriter object definitions
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for .mp4 file format
    # Frames per second,  higher than the fps limit in the main block of code to gather a more detailed/precise video and then extract needed frames from there
    #Also note: 20 fps should result in a smooth video**
    fps = 20.0  
    frame_width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(video_path, fourcc, fps, (frame_width, frame_height))
    start_time = time.time()
    
    #loop to keep grabbing/saving video as 
    while int(time.time() - start_time) < duration:
        ret, frame = video_capture.read()
        if not ret:
            print("Failed to grab frame")
            break
        out.write(frame)
        # Optional - shows the video/the frames being captured during those 10 sec**
        cv2.imshow('Frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Release everything when the job is finished
    video_capture.release()
    out.release()
    cv2.destroyAllWindows()

# Load the video
def process_video(self, video_path):
    global frame 
    
    video_capture = cv2.VideoCapture(video_path)
    #video_capture = cv2.VideoCapture(1)
    # Set frame rate
    fps = 2
    frame_count = 0
    seconds = 1
    # Create MTCNN detector
    detector = MTCNN()
    # Initialize emotion list for the entire video
    video_emotions = []
    # Iterate through frames and record progress 
    max_progress = int(seconds*int(video_capture.get(cv2.CAP_PROP_FPS))//fps)
    progress_counter = 0
    
    while video_capture.isOpened():
        ret, frame = video_capture.read()
        if not ret:
            break
        
        # Capture frame every 'fps' seconds; check if frame is not blurry 
        if not detect_blur_laplacian(frame):
            frame_count += 1
            progress_counter += 1
        else:
            print("Frame is blurry. Dropping frame.")
            continue
            
        self.progress.emit(int((progress_counter / max_progress) * 100))
        if frame_count >= seconds*int(video_capture.get(cv2.CAP_PROP_FPS))//fps:
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
                emotion, _ = emotion_faces(cropped_face)
                '''
                if emotion:
                    cv2.imwrite(f"face_{frame_count}_{i+1}_{emotion}.jpg", cropped_face)
                    '''
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

        main_layout = QVBoxLayout()

        # Top section with 3 labels
        top_layout = QVBoxLayout()
        top_layout.setContentsMargins(0, 0, 0, 0)  # Set outer margins to zero
        top_layout.setSpacing(0)  # Set spacing between labels to zero

        top_layout.addWidget(QLabel("Adaptive Announcement System"), alignment=Qt.AlignmentFlag.AlignCenter)
        top_layout.addWidget(QLabel("Welcome!"), alignment=Qt.AlignmentFlag.AlignCenter)
        top_layout.addWidget(QLabel("Please choose the input method"), alignment=Qt.AlignmentFlag.AlignCenter)

        main_layout.addLayout(top_layout)

        # Middle section with a horizontal box containing two labels
        middle_layout = QHBoxLayout()
        middle_layout.addWidget(QLabel("Live video"))
        middle_layout.addWidget(QLabel("Upload video"))
        main_layout.addLayout(middle_layout)

        # Another horizontal box with two buttons shaped as icons
        button_layout = QHBoxLayout()
        
        # Create first icon button
        icon_button1 = QPushButton()
        original_icon1 = QPixmap("C:/Users/tous1/Desktop/Capstone/Workspace/camera_icon.png")
        resized_icon1 = original_icon1.scaled(40, 40, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
        icon_button1.setIcon(QIcon(resized_icon1))
        icon_button1.setFixedSize(40, 40)
        icon_button1.setStyleSheet("background-color: transparent; border: none;")  # Styling for icon button
        icon_button1.clicked.connect(self.toggle_start_button)  # Connect to a function

        # Create second icon button
        icon_button2 = QPushButton()
        original_icon2 = QPixmap("C:/Users/tous1/Desktop/Capstone/Workspace/film_icon.png")  # Use a different icon if needed
        resized_icon2 = original_icon2.scaled(40, 40, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
        icon_button2.setIcon(QIcon(resized_icon2))
        icon_button2.setFixedSize(40, 40)
        icon_button2.setStyleSheet("background-color: transparent; border: none;")  # Styling for icon button
        icon_button2.clicked.connect(self.toggle_file_input)  # Connect to a function

        self.file_input = QLineEdit()
        self.file_input.setPlaceholderText("Enter video file name")
        self.file_input.setVisible(False)

        # Add buttons to the layout
        button_layout.addWidget(icon_button1)
        button_layout.addWidget(icon_button2)
        main_layout.addLayout(button_layout)

        # Bottom section with a label and a button
        bottom_layout = QHBoxLayout()
        bottom_label = QLabel("click START to begin crowd emotion detection")
        bottom_button = QPushButton("START")
        bottom_button.clicked.connect(self.start_emotion_detection)  # Connect to a function
        bottom_layout.addWidget(bottom_label)
        bottom_layout.addWidget(bottom_button)
        main_layout.addLayout(bottom_layout)

        # Set the main layout for the window
        self.setLayout(main_layout)

    def toggle_start_button(self):
        # Show the start button if a selection has been made
        self.start_button.setVisible(self.live_button.isChecked() or self.preset_button.isChecked())

    def toggle_file_input(self):
        # Show file input field if "Use pre-set video file" is selected
        self.file_input.setVisible(self.preset_button.isChecked())
        self.toggle_start_button()

    def start_emotion_detection(self):
        if self.live_button.isChecked():
            # Live video
            capture_video('captured_video.mp4', duration=10)
            video_source = "captured_video.mp4"
        elif self.preset_button.isChecked():
            video_source = self.file_input.text()  # Pre-set video file name
        #emotion_percentages = process_video("video_3.mp4")
        emotion_percentages = process_video(video_source)
        self.screen2.set_emotion_data(emotion_percentages)
        self.stack.setCurrentIndex(1)


class EmotionWorker(QThread):
    finished = pyqtSignal()
    progress = pyqtSignal(int)
    image = pyqtSignal(list)
    emotion = pyqtSignal(dict)
    
    def __init__(self):
        QThread.__init__(self)
        
    def run(self):
        deepface_result = None
        self.progress.emit(0)
        '''
        with multiprocessing.Pool(processes=1) as pool:
            deepface_result = pool.apply(process_video, args=(audio_file,))
            print(str(deepface_result))
            '''
        deepface_result=process_video(self, audio_file)
        print(type(deepface_result))
        self.emotion.emit(deepface_result)
        
import sys
from PyQt6.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget, QPushButton, QSlider, QHBoxLayout
from PyQt6.QtMultimedia import QMediaPlayer
from PyQt6.QtMultimediaWidgets import QVideoWidget
from PyQt6.QtCore import QUrl, Qt
from PyQt6.QtGui import QImage, QPixmap

class DisplayImageWidget(QWidget):
    def __init__(self):
        super(DisplayImageWidget, self).__init__()
        self.image = None  # Initialize image to None
        self.frame = QLabel()
        self.layout = QHBoxLayout(self)
        self.layout.addWidget(self.frame)

    def set_image(self, image):
        self.image = image
        if self.image is not None:
            # Convert the image to a QImage
            self.convert = QImage(self.image, self.image.shape[1], self.image.shape[0], self.image.strides[0], QImage.Format.Format_BGR888)
            
            # Calculate the scaled size to fit within the frame
            frame_width, frame_height = self.frame.size().width(), self.frame.size().height()
            image_width, image_height = self.convert.size().width(), self.convert.size().height()
            scale_factor = min(frame_width / image_width, frame_height / image_height)
            scaled_width = int(image_width * scale_factor)
            scaled_height = int(image_height * scale_factor)

            # Scale the image
            scaled_image = self.convert.scaled(scaled_width, scaled_height, Qt.AspectRatioMode.KeepAspectRatio)

            # Set the scaled image to the label
            self.frame.setPixmap(QPixmap.fromImage(scaled_image))
        else:
            self.frame.clear()

    def get_image(self):
        return self.image

class VideoPlayer(QWidget):
    def __init__(self, file_path):
        global frame 
        super().__init__()

        self.media_player = QMediaPlayer()
        self.media_player.setSource(QUrl.fromLocalFile(file_path))
        print(str(dir(self.media_player)))
        print(str(self.media_player.hasVideo()))
        print(str(self.media_player.duration()))
        
        self.video_widget = QVideoWidget()
        print(str(dir(self.video_widget)))
        self.media_player.setVideoOutput(self.video_widget)
        
        self.media_player.play()
        
class Screen2(QWidget):
    #Note: when data is passed from one screen to another, need to have the screen the data needs to 'go to' passed in as an arugument here**
    def __init__(self, stack, screen3):
        global audio_file 
        super().__init__()
        self.stack = stack
        self.screen3 = screen3
        layout = QVBoxLayout()
        self.label = QLabel("Emotion detection is now processing, please wait for your results!\n", self)
        self.label1 = QLabel("")
        self.option_1_button = QPushButton("Select option 1")
        self.option_1_button.clicked.connect(self.go_to_screen3_completebreakdown)

        self.label2 = QLabel("")
        option_2_button = QPushButton("Select option 2")
        option_2_button.clicked.connect(self.go_to_screen3_maxemotion)
        
        self.label3 = QLabel()
        
        layout.addWidget(self.label)
        layout.addWidget(self.label1)
        layout.addWidget(self.option_1_button)
        layout.addWidget(self.label2)
        layout.addWidget(option_2_button)
        layout.addWidget(self.label3)
        
        self.video_player_widget = DisplayImageWidget()
        layout.addWidget(self.video_player_widget)
        
        self.setLayout(layout)


    def set_emotion_data(self, emotion_data):
        self.emotion_data = emotion_data #defining for later use**
        # Update the label or UI with emotion percentages
        if self.emotion_data != None:
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

    def restart(self):
        self.emotion_data =None
        self.label1.setText("")
        self.label2.setText("")
        self.label.setText("Emotion detection is now processing, please wait for your results!\n")

    def update_emotion(self, emotion):
        print(emotion)
        
        self.label.setText("Emotion detection is now complete!\n")
        self.set_emotion_data(emotion)
        
    def update_progress(self, value):
        global frame 
        self.label.setText(f"Progress: {value}%")
        if frame is not None: 
            self.video_player_widget.set_image(frame)


class Screen3(QWidget):
    def __init__(self, stack):
        super().__init__()
        self.stack = stack
        layout = QVBoxLayout()
        self.label = QLabel("Generating emotionally altered text screen. \nPlease enter sentence in input box below then press the 'generate emotionally altered announcement' button. \nYou may alter the text in the input box and re-generate the emotionally altered text")
        #layout.addWidget(label)
        self.labelemotionbreakdown = QLabel("")
        self.input_box = QLineEdit()
        self.submit_button = QPushButton("Generate emotionally altered announcement")
        self.labelgeneratedtext = QLabel("")
        self.labelgeneratedtext.setWordWrap(True)
        self.submit_button.clicked.connect(self.gemini_connection)

        #Restart Button
        self.restart_button = QPushButton("Restart")
        self.restart_button.clicked.connect(self.restart)

        layout.addWidget(self.label)
        layout.addWidget(self.labelemotionbreakdown)
        layout.addWidget(self.labelgeneratedtext)
        layout.addWidget(self.input_box)
        layout.addWidget(self.submit_button)
        layout.addWidget(self.restart_button)

        self.setLayout(layout)
    def set_emotions_forgemini(self, emotion_topass_togemini):
        self.emotion_forgemini = emotion_topass_togemini
        self.labelemotionbreakdown.setText = (f"Using emotion breakdown:\n{self.emotion_forgemini}")

    def restart(self):
        self.stack.setCurrentIndex(0)


    def gemini_connection(self):
        user_input = self.input_box.text()
        resp = generate_gemini_content(user_input, self.emotion_forgemini)
        self.labelgeneratedtext.setText(f"Generated content: {resp}")

class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setFixedSize(800, 800)
        self.setStyleSheet('background-color:lightblue; color:black; font-weight: bold; font-size: 16px')
        self.stack = QStackedWidget()
        self.screen3 = None
        self.worker = None
        self.screen3 = Screen3(self.stack)
        self.screen2 = Screen2(self.stack, self.screen3)
        self.screen1 = Screen1(self.stack, self.screen2)

        self.worker = EmotionWorker()
        
        self.worker.emotion.connect(self.screen2.update_emotion)
        self.worker.progress.connect(self.screen2.update_progress)
        # self.screen1 = Screen1(self.stack)
        # self.screen2 = Screen2(self.stack)
        # self.screen3 = Screen3(self.stack)

        self.stack.addWidget(self.screen1)
        self.stack.addWidget(self.screen2)
        self.stack.addWidget(self.screen3)

        layout = QVBoxLayout()
        layout.addWidget(self.stack)
        self.setLayout(layout)
    
    def start_worker(self):
        self.worker.start()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())