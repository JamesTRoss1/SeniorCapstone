import cv2
from deepface import DeepFace
from mtcnn import MTCNN
import google.generativeai as genai

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
    genai.configure(api_key="AIzaSyBiMZ9M0EotI0WlHclUKsJvgp_pfu_hGpc")  # Replace "YOUR_API_KEY" with your actual API key

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
video_path = "video_3.mp4"
video_capture = cv2.VideoCapture(video_path)

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

# Output overall emotion for the video
if video_emotions:
    overall_emotion = max(set(video_emotions), key=video_emotions.count)
    print("Overall Emotion for the Video:", overall_emotion)
    sentence = input("Enter a sentence: ")
    generated_sentence = generate_gemini_content(sentence, overall_emotion)
    print("Generated Sentence:", generated_sentence)
else:
    print("No emotions detected in the video.")
