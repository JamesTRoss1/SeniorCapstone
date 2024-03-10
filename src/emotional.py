import cv2 
from deepface import DeepFace 
from itertools import repeat

def start():
    """
    Builds the model globally as to only be called once 
    """

    global model, emotionlabels
    model = DeepFace.build_model("Emotion")
    emotion_labels = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']

def emotion_faces(faces, frame):
    """
    Determines the emotions of a collection of faces in a frame 

    :param list faces: The faces of the frame, with their coordinates in the frame
    :param opencv-frame-object frame: frame to pull the faces from 
    :return: score of each emotions, most prevalent emotions 
    :rtype: list, list
    """

    emotion_scores = list(repeat(0, len(emotion_labels)))

    for (x, y, w, h) in faces:
        # Extract the face ROI (Region of Interest)
        face = frame[y:y + h, x:x + w]

        # Resize the face ROI to match the input shape of the model
        face = cv2.resize(face, (48, 48), interpolation=cv2.INTERAREA)

        # Normalize the resized face image
        face = face / 255.0

        # Reshape the image to match the input shape of the model
        face = face.reshape(1, 48, 48, 1)

        # Predict emotions using the pre-trained model
        preds = model.predict(face)[0]

        emotion_idx = preds.argmax()
        emotion = emotion_labels[emotion_idx]
        emotion_scores[emotion_idx] = emotion_scores[emotion_idx] + 1 

    #Determine the most prevalent emotion collected 
    return emotion_scores, [emotion_labels[i] for i, x in enumerate(emotion_scores) if x == max(emotion_scores)]
