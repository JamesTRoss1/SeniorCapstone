import sys
from PyQt6.QtWidgets import QHBoxLayout, QApplication, QWidget, QLabel, QVBoxLayout, QPushButton, QStackedWidget, QLineEdit, QRadioButton, QFrame, QFileDialog, QCheckBox, QButtonGroup
from PyQt6.QtCore import QThread, pyqtSignal,  Qt, QUrl, QProcess, QCoreApplication
from PyQt6.QtGui import QFont, QPixmap, QImage, QPixmap
from PyQt6.QtMultimedia import QMediaPlayer
from PyQt6.QtMultimediaWidgets import QVideoWidget
import cv2
from deepface import DeepFace
from mtcnn import MTCNN
import google.generativeai as genai
#import audio_tester
import subprocess
import os 
import time
import re
import pyttsx3
engine = pyttsx3.init()

audio_file = ""
live_video = False
frame = None 
tts = True 

# Disable
def blockPrint():
    sys.stdout = open(os.devnull, 'w')

# Restore
def enablePrint():
    sys.stdout = sys.__stdout__

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
def generate_gemini_content(sentence, emotion, location):
    start = time.time()
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
    prompt = f"Prompt: Please alter the following sentence, considering the specified crowd sentiment, aiming to evoke a more positive emotional response, and taking into account the user's location. Aim to maintain the original meaning while adjusting the tone, vocabulary, or context as needed. Parameters: Sentence: {sentence} Crowd Sentiment: {emotion} User Location: {location} Instructions: If the crowd sentiment is negative, aim to elevate the emotional tone of the response. This might involve offering hope or optimism, providing comfort or reassurance, or shifting the focus towards more positive aspects of the situation or related topics. Ensure the response is relevant to the input sentence and the overall theme or topic. Incorporate location-specific references or cultural nuances to enhance the response's relevance and impact. For example, in California, reference local landmarks, popular culture, or current events; in Georgia, highlight Southern hospitality, historical significance, or outdoor activities. Be mindful of the specific context and nuances of the situation to avoid inappropriate or insensitive responses. Consider the cultural sensitivities and preferences of the user's location. If no location is provided then give a generic response. In all cases, only provide a single response, do not give options."
    response = model.generate_content(prompt, safety_settings=safe)
    cleaned_response = re.sub(r'[\n\\/\t]', ' ', str(response.text).strip())
    cleaned_response = cleaned_response.replace('/', ' ')
    print(str(cleaned_response))
    print(str(time.time() - start))
    return cleaned_response

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

def process_video(self, video_path):
    global frame, live_video 
    
    video_capture = None 
    start = time.time()
    
    if live_video:
        print("Using Live Video")
        video_capture = cv2.VideoCapture(0)
    else:
        video_capture = cv2.VideoCapture(video_path)
    
    # Set frame rate
    FRAMES = 10 
    frame_count = 0
    # Create MTCNN detector
    detector = MTCNN()
    # Initialize emotion list for the entire video
    video_emotions = []
    
    # Iterate through frames and record progress 
    max_progress = FRAMES
    progress_counter = 0
    
    while video_capture.isOpened():
        if frame_count == FRAMES:
            break 
        
        ret, frame = video_capture.read()
        if not ret:
            print("End of video")
            break
        
        # Capture frame every 'fps' seconds; check if frame is not blurry 
        if not detect_blur_laplacian(frame, 30):
            start_ = time.time()
            # Detect faces
            faces = detector.detect_faces(frame)
            cropped_faces = []
            self.progress.emit(int((progress_counter / max_progress) * 100))

            # Check if any faces were detected
            if len(faces) == 0:
                print("No faces detected in this frame.")
                continue  # Skip this frame if no faces are detected
            else:
                frame_count += 1
                progress_counter += 1
                print(f"Processing frame {frame_count}")
                
                # Extract faces and append to cropped_faces list
                for i, face in enumerate(faces):
                    x, y, w, h = face['box']
                    cropped_face = frame[y:y + h, x:x + w]
                    cropped_faces.append(cropped_face)
                    # Save detected face as JPEG file
                    emotion, _ = emotion_faces(cropped_face)
            
            # Output the list of cropped faces
            for i, face in enumerate(cropped_faces):
                emotion, _ = emotion_faces(face)
                if emotion:
                    video_emotions.append(emotion)
            print(str(time.time() - start_))
                    
        else:
            print("Frame is blurry. Dropping frame.")
            continue
        
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
        print(str(time.time() - start))
        return emotion_percentages
    else:
        print("No emotions detected in the video.")

class Screen1(QWidget):
    def __init__(self, stack, screen2):
        super().__init__()
        self.stack = stack
        self.screen2 = screen2

        layout = QVBoxLayout()
        layout.addSpacing(20)

        # First label: Big heading with custom color
        heading_label = QLabel("Welcome to the Adaptive Announcement System!")
        heading_font = QFont("Arial", 32, QFont.Weight.Bold)  # Bigger heading font size
    
        heading_label.setFont(heading_font)  # Set the font for the heading
        heading_label.setStyleSheet("color: #007BFF;") 
        heading_label.setAlignment(Qt.AlignmentFlag.AlignCenter)  # Center the heading
        heading_label.setWordWrap(True)  # Allow wrapping if necessary
        heading_label.setMargin(20)  # Adding margin around the heading label

        # Create a container (box) for the heading label
        heading_box = QFrame()
        heading_box.setFrameShape(QFrame.Shape.StyledPanel)  # Make it a styled box
        heading_box.setFrameShadow(QFrame.Shadow.Raised)  # Optional: Add shadow
        heading_box.setStyleSheet("""
            background-color: #ECF0F1;  /* Light gray background */
            border: 2px solid #007BFF;   /* Blue border */
            border-radius: 10px;         /* Rounded corners */
            padding: 10px;               /* Padding inside the box */
        """)
        
        heading_box_layout = QVBoxLayout()  # Create a layout for the heading box
        heading_box_layout.addSpacing(20)
        heading_box_layout.addWidget(heading_label)  # Add the heading label to the layout
        heading_box.setLayout(heading_box_layout)  # Set the layout for the QFrame

        # Second label: Subheading or description with smaller font
        self.description_label = QLabel("Select the type of video input")
        self.style_label(self.description_label)

        # Image Addition (PyQt6)
        image_label = QLabel()
        pixmap = QPixmap("images/announcement.jpg")  # Replace with your image path
        # Optionally resize the image to fit well in the layout
        pixmap = pixmap.scaled(300, 300, Qt.AspectRatioMode.KeepAspectRatio)
        image_label.setPixmap(pixmap)
        image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        # Create video type selection radio buttons
        self.button_group = QButtonGroup()
        self.live_button = QRadioButton("Use live video footage")
        self.preset_button = QRadioButton("Use pre-set video file")
        self.button_group.addButton(self.live_button)
        self.button_group.addButton(self.preset_button)
        self.button_group.setExclusive(True)
        
        # Create a horizontal layout for radio buttons to center them
        radio_layout = QHBoxLayout()
        radio_layout.addWidget(self.live_button)
        radio_layout.addWidget(self.preset_button)
        radio_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)  # Center the radio buttons
        radio_layout.setSpacing(20)  # Add spacing between the radio buttons

        # File input (hidden initially)
        self.file_input = QLabel()
        self.file_input.setVisible(False)
        self.style_label(self.file_input)

        # Start button (initially hidden)
        self.start_button = QPushButton("Start")
        self.start_button.setVisible(False)
        self.start_button.clicked.connect(self.start_emotion_detection)
        self.style_button(self.start_button)  # Apply button styling

        # Connect video type selection buttons
        self.live_button.toggled.connect(self.toggle_start_button)
        self.preset_button.toggled.connect(self.toggle_file_input)

        # Add widgets to the main layout
        layout.addWidget(heading_box)  # Heading in a styled box
        layout.addWidget(self.description_label)  # Description label
        layout.addWidget(image_label)  # Image display
        layout.addLayout(radio_layout)  # Add radio buttons layout
        layout.addWidget(self.file_input)  # File input field
        layout.addWidget(self.start_button)  # Start button

        layout.setSpacing(30)  # Set spacing between widgets
        layout.setContentsMargins(50, 40, 50, 40)  # Add margins around layout

        self.setLayout(layout)

    def style_button(self, button):
        button.setStyleSheet("""
             QPushButton {
                background-color: #3498DB;  /* Bright blue background */
                color: white;                /* White text */
                border: none;                /* No border */
                padding: 12px 20px;          /* Larger padding for the button */
                font-size: 18px;             /* Increase font size */
                border-radius: 8px;          /* Rounded corners */
                min-width: 200px;            /* Ensure the button width is enough */
            }
            QPushButton:hover {
                background-color: #2980B9;   /* Darker blue on hover */
            }
        """)
        
    def style_label(self, label):
        description_font = QFont("Arial", 18)  # Slightly smaller font for description
        label.setFont(description_font)  # Apply the smaller font
        label.setAlignment(Qt.AlignmentFlag.AlignCenter)  # Center align the text
        label.setStyleSheet("color: #34495E;")  # Slightly lighter color for description
        label.setWordWrap(True)  # Allow wrapping if necessary
        label.setMargin(10)  # Add some margin for breathing space

    def toggle_start_button(self):
        # Show the start button if a selection has been made
        self.description_label.setText("Please click 'Start' to begin crowd emotion detection")
        self.start_button.setVisible(self.live_button.isChecked() or self.preset_button.isChecked())

    def toggle_file_input(self):
        # Show file input field if "Use pre-set video file" is selected
        if self.preset_button.isChecked():
            file_dialog = QFileDialog()
            file_dialog.setNameFilter("MP4 Files (*.mp4)")
            file_dialog.setFileMode(QFileDialog.FileMode.ExistingFile)

            if file_dialog.exec():
                self.selected_file = file_dialog.selectedFiles()[0]
                self.file_input.setText("Video File Path: " + str(self.selected_file))
                self.file_input.setVisible(self.preset_button.isChecked())
            
        self.toggle_start_button()

    def start_emotion_detection(self):
        if self.live_button.isChecked():
            # Live video
            global live_video, audio_file
            audio_file = ""
            live_video = True
        elif self.preset_button.isChecked():
            audio_file = self.selected_file
        #emotion_percentages = process_video("video_3.mp4")
        main_window=self.stack.parent()
        main_window.start_worker(audio_file)
        self.stack.setCurrentIndex(1)
        self.screen2.restart()


class EmotionWorker(QThread):
    finished = pyqtSignal()
    progress = pyqtSignal(int)
    image = pyqtSignal(list)
    emotion = pyqtSignal(dict)
    audiofile = pyqtSignal(str)
    def __init__(self, audio_file):
        QThread.__init__(self)
        self.audio_file = audio_file
    def run(self):
        deepface_result = None
        self.progress.emit(0)
        '''
        with multiprocessing.Pool(processes=1) as pool:
            deepface_result = pool.apply(process_video, args=(audio_file,))
            print(str(deepface_result))
            '''
        deepface_result=process_video(self, self.audio_file)
        print(type(deepface_result))
        self.emotion.emit(deepface_result)
        


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
            
            # Calculate the scaled size to fit within the frame, but prioritize filling the frame
            frame_width, frame_height = self.frame.size().width(), self.frame.size().height()
            image_width, image_height = self.convert.size().width(), self.convert.size().height()
            scale_factor = max(frame_width / image_width, frame_height / image_height)
            scaled_width = int(image_width * scale_factor)
            scaled_height = int(image_height * scale_factor)

            # Scale the image
            scaled_image = self.convert.scaled(scaled_width, scaled_height, Qt.AspectRatioMode.IgnoreAspectRatio)

            # Set the scaled image to the label
            self.frame.setPixmap(QPixmap.fromImage(scaled_image))
            self.frame.update()
        else:
            self.frame.clear()

    def clear_image(self):
        self.frame.setPixmap(None)
        
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
        self.layout = QVBoxLayout()
        self.layout.setSpacing(20)
        self.label = QLabel("Emotion detection is now processing, please wait for your results!\n", self)
        self.style_label(self.label)
        
        self.label1 = QLabel("")
        self.style_label(self.label1)
        
        self.option_1_button = QPushButton("Complete Emotion Breakdown")
        self.option_1_button.clicked.connect(self.go_to_screen3_completebreakdown)
        self.option_1_button.hide()
        self.label2 = QLabel("")
        self.style_label(self.label2)
        
        self.option_2_button = QPushButton("Max Emotion in Video")
        self.option_2_button.clicked.connect(self.go_to_screen3_maxemotion)
        self.option_2_button.hide()
        
        self.label3 = QLabel()
        self.style_label(self.label3)
        
        self.layout.addWidget(self.label)
        self.layout.addWidget(self.label1)
        self.layout.addWidget(self.option_1_button)
        self.layout.addWidget(self.label2)
        self.layout.addWidget(self.option_2_button)
        self.layout.addWidget(self.label3)
        
        self.video_player_widget = DisplayImageWidget()
        self.layout.addWidget(self.video_player_widget)
        
        self.checkBox = QCheckBox('Text-To-Speech', self)
        self.layout.addWidget(self.checkBox)
        
        self.setLayout(self.layout)
        
    def style_button(self, button):
        button.setStyleSheet("""
             QPushButton {
                background-color: #3498DB;  /* Bright blue background */
                color: white;                /* White text */
                border: none;                /* No border */
                padding: 12px 20px;          /* Larger padding for the button */
                font-size: 18px;             /* Increase font size */
                border-radius: 8px;          /* Rounded corners */
                min-width: 200px;            /* Ensure the button width is enough */
            }
            QPushButton:hover {
                background-color: #2980B9;   /* Darker blue on hover */
            }
        """)
        
    def style_label(self, label):
        description_font = QFont("Arial", 18)  # Slightly smaller font for description
        label.setFont(description_font)  # Apply the smaller font
        label.setAlignment(Qt.AlignmentFlag.AlignCenter)  # Center align the text
        label.setStyleSheet("color: #34495E;")  # Slightly lighter color for description
        label.setWordWrap(True)  # Allow wrapping if necessary
        label.setMargin(10)  # Add some margin for breathing space


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
        global tts
        emotion_text_gemini = "\n".join([f"{emotion}: {percent:.2f}%" for emotion, percent in self.emotion_data.items()])
        self.screen3.set_emotions_forgemini(emotion_text_gemini)
        tts=self.checkBox.isChecked()
        self.stack.setCurrentIndex(2)

    def go_to_screen3_maxemotion(self):
        # Switch to screen 3 after selecting an option
        #pass the overall emotion to screen 3 in this case
        global tts 
        max_emotion = max(self.emotion_data, key=self.emotion_data.get)
        self.screen3.set_emotions_forgemini(max_emotion)
        tts = self.checkBox.isChecked()
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
        self.option_1_button.show()
        self.option_2_button.show()
        
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
        layout.setSpacing(20)
        self.label = QLabel("Generating emotionally altered text screen. Please enter sentence in input box below then press the 'generate emotionally altered announcement' button. You may alter the text in the input box and re-generate the emotionally altered text")
        self.style_label(self.label)
        
        self.labelemotionbreakdown = QLabel("")
        self.style_label(self.labelemotionbreakdown)
        
        self.input_box = QLineEdit()
        self.input_box.setStyleSheet("""
            QLineEdit {
                padding: 10px;
                border: 1px solid #BDC3C7;
                border-radius: 5px;
                background-color: #ECF0F1;
                font-size: 16px;
            }
        """)
        
        self.labellocation = QLabel("Please put your location in the box below for a more personalized statement. Leave blank for generic response without location consideration.")
        self.labellocation.setWordWrap(True)
        self.style_label(self.labellocation)
        
        self.input_loc_box = QLineEdit()
        self.input_loc_box.setStyleSheet("""
            QLineEdit {
                padding: 10px;
                border: 1px solid #BDC3C7;
                border-radius: 5px;
                background-color: #ECF0F1;
                font-size: 16px;
            }
        """)
        
        self.submit_button = QPushButton("Generate emotionally altered announcement")
        self.style_button(self.submit_button)
        
        self.labelgeneratedtext = QLabel("")
        self.labelgeneratedtext.setWordWrap(True)
        self.style_label(self.labelgeneratedtext)
        
        self.submit_button.clicked.connect(self.gemini_connection)

        #Restart Button
        self.restart_button = QPushButton("Restart")
        self.style_button(self.restart_button)
        self.restart_button.clicked.connect(self.restart)

        layout.addWidget(self.label)
        layout.addWidget(self.labelemotionbreakdown)
        layout.addWidget(self.labelgeneratedtext)
        layout.addWidget(self.input_box)
        layout.addWidget(self.labellocation)
        layout.addWidget(self.input_loc_box)
        layout.addWidget(self.submit_button)
        layout.addWidget(self.restart_button)
        self.setLayout(layout)
        
    def style_button(self, button):
        button.setStyleSheet("""
             QPushButton {
                background-color: #3498DB;  /* Bright blue background */
                color: white;                /* White text */
                border: none;                /* No border */
                padding: 12px 20px;          /* Larger padding for the button */
                font-size: 18px;             /* Increase font size */
                border-radius: 8px;          /* Rounded corners */
                min-width: 200px;            /* Ensure the button width is enough */
            }
            QPushButton:hover {
                background-color: #2980B9;   /* Darker blue on hover */
            }
        """)
        
    def style_label(self, label):
        description_font = QFont("Arial", 18)  # Slightly smaller font for description
        label.setFont(description_font)  # Apply the smaller font
        label.setAlignment(Qt.AlignmentFlag.AlignCenter)  # Center align the text
        label.setStyleSheet("color: #34495E;")  # Slightly lighter color for description
        label.setWordWrap(True)  # Allow wrapping if necessary
        label.setMargin(10)  # Add some margin for breathing space
        
    def set_emotions_forgemini(self, emotion_topass_togemini):
        self.emotion_forgemini = emotion_topass_togemini
        self.labelemotionbreakdown.setText = (f"Using emotion breakdown:\n{self.emotion_forgemini}")

    def restart(self):
        QCoreApplication.quit()
        QProcess.startDetached(sys.executable, sys.argv)


    def gemini_connection(self):
        user_input = self.input_box.text()
        location_input = self.input_loc_box.text()
        resp = generate_gemini_content(user_input, self.emotion_forgemini, location_input)
        self.labelgeneratedtext.setText(f"Generated content: {resp}")

        if tts:
            voices = engine.getProperty('voices')       #getting details of current voice
            engine.setProperty('rate', 125)     # setting up new voice rate
            engine.setProperty('voice', voices[1].id)   #changing index, changes voices. 1 for female
            engine.say(resp)
            engine.runAndWait()

class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setFixedSize(1000, 1000)
        self.setStyleSheet('background-color:lightblue; color:black; font-weight: bold; font-size: 16px')
        self.stack = QStackedWidget()
        self.screen3 = None
        self.worker = None
        self.screen3 = Screen3(self.stack)
        self.screen2 = Screen2(self.stack, self.screen3)
        self.screen1 = Screen1(self.stack, self.screen2)

        self.stack.addWidget(self.screen1)
        self.stack.addWidget(self.screen2)
        self.stack.addWidget(self.screen3)

        layout = QVBoxLayout()
        layout.addSpacing(20)
        layout.addWidget(self.stack)
        self.setLayout(layout)
    
    def start_worker(self, audio_file):
        self.worker = EmotionWorker(audio_file)
        self.worker.emotion.connect(self.screen2.update_emotion)
        self.worker.progress.connect(self.screen2.update_progress)        
        self.worker.start()

blockPrint()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
