!pip install mtcnn
from IPython.display import display, Javascript
from google.colab.output import eval_js
from base64 import b64decode
from mtcnn import MTCNN
import numpy as np
import cv2
import os


# Initialize the MTCNN detector
detector = MTCNN()

# JavaScript code to access the webcam and capture frames
js = Javascript('''
    let video;
    let frameCounter = 0;

    async function startCamera() {
        video = document.createElement('video');
        document.body.appendChild(video);
        const stream = await navigator.mediaDevices.getUserMedia({video: true});
        video.srcObject = stream;

        // Wait for the video to be fully loaded before setting the canvas dimensions
        await new Promise((resolve) => {
            video.onloadedmetadata = () => {
                resolve();
            };
        });

        await video.play();

        // Resize the output to fit the video element.
        google.colab.output.setIframeHeight(document.documentElement.scrollHeight, true);
    }

    async function captureFrame() {
        if (!video) {
            throw new Error('Video element not initialized');
        }

        const canvas = document.createElement('canvas');
        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;
        canvas.getContext('2d').drawImage(video, 0, 0);
        const imgData = canvas.toDataURL('image/jpeg');
        return imgData;
    }
''')

# Display the JavaScript code
display(js)

# Execute the startCamera function to initialize the camera
eval_js('startCamera()')

# Function to decode base64 image and convert it to OpenCV format
def decode_image(img_data):
    img_bytes = b64decode(img_data.split(',')[1])
    nparr = np.frombuffer(img_bytes, np.uint8)
    img_np = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return img_np

# Create directory to save frames
output_dir = 'detected_faces'
os.makedirs(output_dir, exist_ok=True)

# Main loop for real-time face detection
frame_counter = 0
while True:
    try:
        # Capture frame from the webcam
        img_data = eval_js('captureFrame()')

        # Decode the image and convert to OpenCV format
        frame = decode_image(img_data)

        # Detect faces using MTCNN
        faces = detector.detect_faces(frame)

        # Draw bounding boxes around the faces
        for face in faces:
            x, y, w, h = face['box']
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        # Save frame with detected faces
        for face in faces:
            x, y, w, h = face['box']
            face_img = frame[y:y+h, x:x+w]
            cv2.imwrite(os.path.join(output_dir, f"frame_{frame_counter}.jpg"), face_img)
            frame_counter += 1

        # Display the resulting frame
        cv2.imshow('Real-time Face Detection', frame)

        # Press 'q' key to exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    except Exception as e:
        print(f"Error: {e}")
        break

# Close OpenCV windows
cv2.destroyAllWindows()

