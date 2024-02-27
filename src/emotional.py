import cv2 
from deepface import DeepFace 
from itertools import repeat

def __start__():
    """
    Builds the model globally as to only be called once 
    """

    global model, emotion_labels
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
        face_roi = frame[y:y + h, x:x + w]

        # Resize the face ROI to match the input shape of the model
        resized_face = cv2.resize(face_roi, (48, 48), interpolation=cv2.INTER_AREA)

        # Normalize the resized face image
        normalized_face = resized_face / 255.0

        # Reshape the image to match the input shape of the model
        reshaped_face = normalized_face.reshape(1, 48, 48, 1)

        # Predict emotions using the pre-trained model
        preds = model.predict(reshaped_face)[0]
        emotion_idx = preds.argmax()
        emotion = emotion_labels[emotion_idx]
        emotion_scores[emotion_idx] = emotion_scores[emotion_idx] + 1 

    #Determine the most prevalent emotion collected 
    return emotion_scores, [emotion_labels[i] for i, x in enumerate(a) if x == max(a)]