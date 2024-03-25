import cv2

# Load the video
video_path = "video.mp4"
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
        # Detect faces
        faces = detector.detect_faces(frame)

        cropped_faces = []

        # Extract faces and append to cropped_faces list
        for face in faces:
            x, y, w, h = face['box']
            cropped_faces.append(frame[y:y+h, x:x+w])

        # Output the list of cropped faces
        for i, face in enumerate(cropped_faces):
            cv2.imwrite(f"face_{frame_count}_{i+1}.jpg", face)

        print(f"Faces extracted from frame {frame_count}")

video_capture.release()
cv2.destroyAllWindows()
