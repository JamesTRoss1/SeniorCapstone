import cv2 
from deepface import DeepFace 
from itertools import repeat
import numpy
import time 

def emotion_faces(faces):
    """
    Determines the emotions of a collection of faces in a frame 

    :param list faces: a numpy array in BGR format, base64 encoded image, or path to the image 
    :return: string of the most prevalent emotion, dictionary of weighted emotion sums of all faces in an image 
    :rtype: str, dict
    """

    #preds is a list of dictionaries; refer to https://github.com/serengil/deepface/blob/master/deepface/DeepFace.py for detailed documentation of analyze()
    preds = DeepFace.analyze(faces, actions = ['emotion'], detector_backend="mtcnn")
    emotions = {}
    weights = {'sad': 1, 'angry': 1, 'surprise': 1, 'fear': 1, 'happy': 1, 'disgust': 1, 'neutral': 1}

    for pred in preds:
        print("Face Analysis: " + str(pred))

        #reliability of the model, don't process bad data. need to check if 0.8 is too high or too low 
        if pred["face_confidence"] < 0.8:
            pass 
        else:
            emotions = {k: emotions.get(k, 0) + pred['emotion'].get(k, 0) for k in set(emotions) | set(pred['emotion'])}

    #weight the sums 
    emotions = {k: emotions.get(k, 0) * weights.get(k, 0) for k in set(emotions) & set(weights)}

    #Determine the most prevalent emotion collected 
    return max(emotions, key=emotions.get), emotions