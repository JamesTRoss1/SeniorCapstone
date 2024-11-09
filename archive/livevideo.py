import cv2
from deepface import DeepFace


## This method has been updated to just return the emotions, and remove unconfident faces 
## That DeepFace  originally included.
def emotion_faces(preds):
    emotions = {}
    weights = {'sad': 1, 'angry': 1, 'surprise': 1, 'fear': 1, 'happy': 1, 'disgust': 1, 'neutral': 1}
    i = 0
    temp = preds
    for pred in preds:
        ## may want to raise the confidence measure to 0.9
        if pred["face_confidence"] < 0.8:
            # removes the face that did not meet the confidence number.
            temp = temp.pop(i)
        else:
            emotions = {k: emotions.get(k, 0) + pred['emotion'].get(k, 0) for k in set(emotions) | set(pred['emotion'])}
            i=i+1    
    #update preds to the list containing higher confidence level faces
    preds = temp
    emotions = {k: emotions.get(k, 0) * weights.get(k, 0) for k in set(emotions) & set(weights)}

    if not emotions:
        return "Unknown", {}

    return max(emotions, key=emotions.get), emotions
## use the face cascade to detect faces for example
video_capture=cv2.VideoCapture(1)

if not video_capture.isOpened():
    video_capture=cv2.VideoCapture(0)
if not video_capture.isOpened():
    raise IOError("Unable to open the webcam")
#read face  

while True:
    ret, frame = video_capture.read()

    ## Move the facial detection and analysis outside of the def so
    ## We can have access to the
    preds = DeepFace.analyze(frame, actions=['emotion', 'age'] , detector_backend="mtcnn", enforce_detection=False)
    ## retrieve group emotion and 
    emotion, _ = emotion_faces(preds)


    # Draws a bounding box around each persons face.
    for faces in preds:

        cv2.rectangle(frame,(faces['region']['x'],faces['region']['y']),(faces['region']['x']+
            faces['region']['w'],faces['region']['y']+faces['region']['h']),(0,255,0),2)
    fontstyle = cv2.FONT_HERSHEY_TRIPLEX
    #putText(img, text, org, font, fontScale,color, thickness, parameters of the image.)
    ##need to write for multiple people.
    
    for faces in preds:
        #labels the persons emotion around their face
        cv2.putText(frame, faces['dominant_emotion'], (faces['region']['x'], faces['region']['y']), fontstyle, 1, (255,0,0), 1, cv2.LINE_AA);

    ## put the overall group emotion in the title of the visual application
    cv2.imshow(f'Overall group emotion: {emotion}', frame)
    
    ## wait key is = to ten seconds to allow for viewing, 1= 1 millisecond
    ## if you want to stop the program, press s
    ## if you are done viewing the image, press any other key to continue.
    if cv2.waitKey(10000) & 0xFF==ord('s'):
        break
    video_capture.release()
cv2.destroyAllWindows()