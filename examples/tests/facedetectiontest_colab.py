from mtcnn import MTCNN
import cv2

# Load the image
image = cv2.imread("group.jpg")

# Create MTCNN detector
detector = MTCNN()

# Detect faces
faces = detector.detect_faces(image)

cropped_faces = []

# Extract faces and append to cropped_faces list
for face in faces:
    x, y, w, h = face['box']
    cropped_faces.append(image[y:y+h, x:x+w])

# Output the list of cropped faces
for i, face in enumerate(cropped_faces):
    cv2.imwrite(f"face_{i+1}.jpg", face)

print("List of faces extracted successfully!")
