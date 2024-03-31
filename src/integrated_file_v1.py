def blur_measure(image):
    """
    Calculates the variance of Laplacian to measure image blurriness

    :param numpy.ndarray image: Input image
    :return: Variance of Laplacian
    :rtype: float
    """
    return cv2.Laplacian(image, cv2.CV_64F).var()

def emotion_faces(faces):
    """
    Determines the emotions of a collection of faces in a frame

    :param list faces: a numpy array in BGR format, base64 encoded image, or path to the image
    :return: string of the most prevalent emotion, dictionary of weighted emotion sums of all faces in an image
    :rtype: str, dict
    """

    #preds is a list of dictionaries; refer to https://github.com/serengil/deepface/blob/master/deepface/DeepFace.py for detailed documentation of analyze()
    preds = DeepFace.analyze(faces, actions=['emotion'], detector_backend="mtcnn", enforce_detection=False)
    emotions = {}
    weights = {'sad': 1, 'angry': 1, 'surprise': 1, 'fear': 1, 'happy': 1, 'disgust': 1, 'neutral': 1}

    for pred in preds:
        print("Face Analysis: " + str(pred))

        # reliability of the model, don't process bad data. need to check if 0.8 is too high or too low
        if pred["face_confidence"] < 0.8:
            pass
        else:
            emotions = {k: emotions.get(k, 0) + pred['emotion'].get(k, 0) for k in set(emotions) | set(pred['emotion'])}

    # weight the sums
    emotions = {k: emotions.get(k, 0) * weights.get(k, 0) for k in set(emotions) & set(weights)}

    # Check if emotions dictionary is empty
    if not emotions:
        return "Unknown", {}

    # Determine the most prevalent emotion collected
    return max(emotions, key=emotions.get), emotions

# Load the video
video_path = "video_3.mp4"
video_capture = cv2.VideoCapture(video_path)

# Set frame rate
# Frame rate (frames per second)
fps = 2
frame_count = 0

# Create MTCNN detector
detector = MTCNN()

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
        for face in faces:
            x, y, w, h = face['box']
            cropped_face = frame[y:y + h, x:x + w]

            # Check blurriness of the face
            #Can include this later as needed**
            # blur_num = blur_measure(cropped_face)
            # if blur_num < 100:  # Adjust threshold as needed, from testing threshold of 50 or less seems reasonable
            #     print(f"Face too blurry, skipping. measure: {blur_num}")
            #     continue  # Skip this face if too blurry

            cropped_faces.append(cropped_face)

        # Check if any faces were detected
        if len(cropped_faces) == 0:
            print("No faces detected in this frame.")
            continue  # Skip this frame if no faces are detected

        # Output the list of cropped faces
        for i, face in enumerate(cropped_faces):
            emotion, emotions = emotion_faces(face)
            if emotion:
                cv2.imwrite(f"face_{frame_count}_{i+1}_{emotion}.jpg", face)
                print(f"Emotion of face {i+1} in frame {frame_count}: {emotion}, Emotion Details: {emotions}")

video_capture.release()
cv2.destroyAllWindows()

