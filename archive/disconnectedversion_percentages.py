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
def generate_gemini_content(sentence):
    #can generate key here: https://aistudio.google.com/app/apikey
    genai.configure(api_key="AIzaSyDAVBCGTHDB4gos1MO7pU-6sOccFakqEc0")  # Replace "YOUR_API_KEY" with your actual API key. (generated from link above)

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

    prompt = f"I have a sentence: \"{sentence}\". The current emotional sentiment of the environment is 50% surprise, 50% happy. Can you rewrite the sentence to better mediate this emotional sentiment while conveying the same core message? Output only the best option, which can be multiple sentences long, that will best improve the emotion of the environment. Do not include an explanation or more than one option."
    response = model.generate_content(prompt, safety_settings=safe)

    return response.text

# Output overall emotion for the video
sentence = input("Enter a sentence: ")
generated_sentence = generate_gemini_content(sentence)
print("Generated sentence:", generated_sentence)
