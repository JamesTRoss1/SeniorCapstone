import pathlib
import textwrap
import os 
import google.generativeai as genai

#requires python 3.9 or greater 
#get api key here: https://aistudio.google.com/app/apikey

def __start__():
    global model, safe

    genai.configure(api_key="AIzaSyDysb2hlF6rCfuuFHL9QRpoUp8CFzbAFBo")

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

def __generate__():
    global model, safe 

    sentence = "Your flight has been delayed"
    emotion = "angry"
    prompt = "I have a sentence: \"{}\".  The current emotional sentiment of the environment is {}.  Can you rewrite the sentence to better mediate this emotional sentiment while conveying the same core message? Output only the best option, which can be multiple sentences long, that will best improve the emotion of the environment. Do not include an explanation or more than one option.".format(sentence, emotion)
    response = model.generate_content(prompt, safety_settings=safe)

    print(str(response.text))
    return str(response.text)

def __main__():
    __start__()
    __generate__()

if __name__ == "__main__":
    __main__()