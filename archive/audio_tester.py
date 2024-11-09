import sys
import argparse 
import os 
import subprocess
import time 
import shutil
import pathlib
from pydub import AudioSegment 
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier, GradientBoostingClassifier, BaggingClassifier
from sklearn.ensemble import AdaBoostRegressor, RandomForestRegressor, GradientBoostingRegressor, BaggingRegressor
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.neural_network import MLPClassifier, MLPRegressor
import traceback
from asteroid.models import BaseModel, DPRNNTasNet
from collections import Counter
import soundfile as sf
import librosa
import noisereduce as nr
import google.generativeai as genai
import speech_recognition as sr
from textblob import TextBlob
import contextlib
import json 
import csv 
from tensorflow.keras.layers import LSTM, GRU

relative_path = 'asteroid/egs/wsj0-mix-var/Multi-Decoder-DPRNN'
script_dir = os.path.dirname(os.path.abspath(__file__))  # Directory of the current script
full_path = os.path.abspath(os.path.join(script_dir, relative_path))
print(str(full_path))
sys.path.append(full_path)
import torch, torchaudio
import argparse
import os
from model import MultiDecoderDPRNN

import nltk 
nltk.download('vader_lexicon')
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# Get the directory of the current script
script_dir = os.path.dirname(os.path.abspath(__file__))
# Construct the path to the folder you want to import from
folder_path = os.path.join(script_dir, 'emotion-recognition-using-speech')
# Add the folder path to sys.path
sys.path.append(folder_path)

from emotion_recognition import EmotionRecognizer
from deep_emotion_recognition import DeepEmotionRecognizer

print("Finished imports")

dnn_output = None 
gemini_output = None


class Analyzer:
    def __init__(self):
        pass 
    
    def __analyze_text__(self, text):
        """
        Transcribes audio from a file and performs sentiment analysis on the transcribed text.

        Args:
            text (str): text to be analyzed.

        Returns:
            float of the polarity of the emotion or None 
        """

        try:
            
            if not isinstance(text, str):
                return None

            # Perform sentiment analysis with TextBlob
            blob = TextBlob(text)
            sentiment = blob.sentiment
            
            # Prepare the results
            results = {
                'transcribed_text': text,
                'sentiment_polarity': sentiment.polarity,
                'sentiment_subjectivity': sentiment.subjectivity
            }
            print(str(results))
            # return results
            return float(sentiment.polarity)

            #Perform sentiment analysis with Vader 
            # analyzer = SentimentIntensityAnalyzer()
            # results = analyzer.polarity_scores(text)
            # print(str(results))

        except Exception as e:
            print(str(e))
            return None 

    def convert_to_wav(self, input_file, output_file):
        """Converts an audio file to WAV format.

        Args:
            input_file: The path to the input audio file.
            output_file: The path to the output WAV file.
        """
        sound = AudioSegment.from_file(input_file)
        sound = sound.set_channels(1)
        sound.export(output_file, format="wav")

class DNNAnalyzer(Analyzer):
    def __init__(self, file_name, model, dnn, preprocess=True):
        self.file_name = file_name
        self.__init__deep_model__(dnn)
        self.__init_audio_split__(model)
        if preprocess:
            self.__preprocess_file__(self.file_name)
            
    def get_emotion(self):
        multiple_emotions = self.__split_and_analyze__(self.file_name)
        aggregate_emotion = self.__aggregate__(multiple_emotions)
        # return aggregate_emotion, multiple_emotions
        return aggregate_emotion
        
    def __preprocess_file__(self, file_name):
        print("Starting conversion")
        new_extension = "wav"
        path = pathlib.Path(file_name)
        new_path = path.with_suffix(f".{new_extension}")
        self.convert_to_wav(path, new_path)
        self.file_name = str(new_path)   
        print("Finished conversion") 
        
    def __split__(self, file_name):
        #tries to make a files directory or clears directory if empty 
        print("Cleaning up directory")
        try:
            shutil.rmtree("files")
        except Exception as e:
            pass
        
        try:
            os.mkdir("files")
        except Exception as e:
            print(str(e))
            
        #separates models
        print("Finished cleaning up directory") 
        print("Splitting " + str(file_name))

        try:
            mixture, sample_rate = torchaudio.load(file_name) 
            mixture = mixture.transpose(0, 0)
            outputs_est = self.split_model.separate(mixture).cpu()
            for i, source in enumerate(outputs_est):
                torchaudio.save(f"files/{i}.wav", source[None], sample_rate)
        except Exception as e:
            self.split_model.separate(file_name, resample=True, force_overwrite=True, output_dir = "files")

        print("Finished splitting " + str(file_name))

    def __preprocess_noise__(self, file_name, out_file):
        try:
            # Load the audio file
            audio, sr = librosa.load(file_name, sr=None)
            # Reduce noise
            audio_denoised = nr.reduce_noise(y=audio, sr=sr)
            # Save the denoised audio
            sf.write(out_file, audio_denoised, sr)
        except Exception as e:
            print(str(e))

    def __preprocess_gain__(self, file_name, out_file):
        # Load the audio file
        audio = AudioSegment.from_file(file_name)

        # Normalize audio
        normalized_audio = audio.apply_gain(-audio.dBFS)  # Normalize to 0 dBFS

        # Save the normalized audio
        normalized_audio.export(out_file, format="wav")


    def __analyze__(self, file_name):
        print("Analyzing " + str(file_name))
        analysis_ = []
        
        # self.__preprocess_noise__(file_name, file_name)
        # self.__preprocess_gain__(file_name, file_name)

        audio = AudioSegment.from_file(str(file_name))
        if audio.channels > 1:
            mono_audio = audio.set_channels(1)
            mono_audio.export(str(file_name), format='wav')

        try:
            analysis_.append(self.rec.predict(file_name))
            print("Prediction: ", analysis_[-1])
        except Exception as e:
            print(str(e))
        
        try:
            analysis_.append(self.rec.predict_proba(file_name))
            print("Prediction Confidence: ", analysis_[-1])
        except Exception as e:
            print(str(e))  

        return analysis_

    def __split_and_analyze__(self, file_name):
        self.__split__(file_name)
        base_path = os.getcwd() 
        dir_path = base_path + "/files"
        analysis = []
        files = [f for f in os.listdir(dir_path) if os.path.isfile(os.path.join(dir_path, f))]
        for file in files:
            file_path = os.path.join(dir_path, file)
            analysis.append(self.__analyze__(file_path))
        print(str(analysis))
        print("Finished Splitting and Analyzing")
        return analysis

    def __aggregate__(self, emotion_list):
        """
        Aggregates a list of tuples (emotion, probability) into a single emotion with the highest total probability.

        Args:
            emotion_list: A list of tuples (most prevalent emotion, probability of each emotion).

        Returns:
            A tuple (emotion, total_probability).
        """

        emotion_probs = {}
        for emotion, probability in emotion_list:
            for k, v in probability.items():
                if k in emotion_probs:
                    emotion_probs[k] += v
                else:
                    emotion_probs[k] = v

        max_emotion = max(emotion_probs, key=emotion_probs.get)
        max_probability = emotion_probs[max_emotion] / float(len(emotion_list))

        return max_emotion, max_probability

    def __init_audio_split__(self, model):
        print("Initializing Asteroid Model")
        self.split_model = model
        print("Finished Asteroid Model")

    #Note that while it works for the DNN, it doesn't work for Gemini. fix later
    def convert_mp4_to_audio(self, input_file, output_file, format='mp3'):
        """
        Convert MP4 file to an audio file (MP3 or WAV) using ffmpeg.

        :param input_file: Path to the input MP4 file.
        :param output_file: Path to the output audio file.
        :param format: Audio format for the output file. Default is 'mp3'.
        """
        # Define the ffmpeg command
        command = [
            'ffmpeg',
            '-i', input_file,      # Input file
            '-vn',                 # No video
            '-acodec', 'libmp3lame' if format == 'mp3' else 'pcm_s16le',  # Audio codec
            '-ar', '44100',        # Audio sample rate
            '-ac', '2',            # Number of audio channels
            output_file            # Output file
        ]
        
        # Run the ffmpeg command
        subprocess.run(command, check=True)

    def __init__deep_model__(self, dnn):
        print("Initializing Deep Emotional Recognizer Model")
        # default parameters (LSTM: 128x2, Dense:128x2)
        self.rec = dnn 
        # train the model
        self.rec.train()
        # get the accuracy
        print(self.rec.test_score())

    def __parse_text_from_audio__(self, audio_file_path):
        recognizer = sr.Recognizer()

        with sr.AudioFile(audio_file_path) as source:
            audio_data = recognizer.record(source)
            text = recognizer.recognize_google(audio_data)
            return text 

class GeminiAnalyzer(Analyzer):
    def __init__(self, converted_file):
        self.converted_file = converted_file
        self.emotion = None
        self.text = None
        self.polarity = None

    def get_emotion(self):
        try:
            self.emotion = self.analyze_emotion()
        except Exception as e:
            print(str(e))
        
        try:
            self.text, self.polarity = self.analyze_text()
        except Exception as e:
            print(str(e))
            
        # return self.emotion, self.text, self.polarity
        return self.emotion 

    def analyze_emotion(self):
        print("Starting Gemini Emotional Analysis")
        gemini_emotion = str(self.generate_gemini_content(self.converted_file, "What is the emotional sentiment expressed in this audio file? Reply with only one word from the following list that closely expresses the audio: happy, sad, angry, neutral, surprise")).strip().lower()
        print("Finished Gemini Emotional Analysis. Emotion: " + str(gemini_emotion))
        return gemini_emotion

    def analyze_text(self):
        print("Starting Gemini Text Analysis With TextBlob")
        gemini_transcribed_text = str(self.generate_gemini_content(self.converted_file, "Give me transcribed text of this audio clip, Only include the transcribed text in your response. Clean it up since this text will be passed to a emotion recognition script that only accepts alphabetical characters.")).strip()
        print("Finished Gemini Text Analysis. Text: " + str(gemini_transcribed_text))
        gemini_polarity_score = super().__analyze_text__(gemini_transcribed_text)
        return gemini_transcribed_text, gemini_polarity_score

    # Function to generate content using Google Gemini
    def generate_gemini_content(self, audio_file, sentence):
        try:
            #can generate key here: https://aistudio.google.com/app/apikey
            genai.configure(api_key="AIzaSyDysb2hlF6rCfuuFHL9QRpoUp8CFzbAFBo")  # Replace "YOUR_API_KEY" with your actual API key

            for m in genai.list_models():
                if 'generateContent' in m.supported_generation_methods:
                    print(m.name)
            print()

            # model = genai.GenerativeModel('gemini-1.5-pro-latest')
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

            myfile = genai.upload_file(audio_file)
            print(f"{myfile=}")

            model = genai.GenerativeModel("gemini-1.5-flash")
            result = model.generate_content([myfile, sentence], safety_settings=safe)
            print(str(dir(result.candidates[0])))
            print("Safety Ratings: " + str(result.candidates[0].safety_ratings))
            print(f"{result.text=}")
            return result.text

        except Exception as e:
            print(str(e))
            return None

def __emotion_audio_test__(path, isDir=False):
    out = {}
    
    try:
        # mpariente/DPRNNTasNet-ks2_WHAM_sepclean and JorisCos/ConvTasNet_Libri2Mix_sepclean_16k is best 
        # look into libri3mix sepnoisy 16k and libri3mix sepclean 16k 
        models = []
        models.append(BaseModel.from_pretrained("mpariente/DPRNNTasNet-ks2_WHAM_sepclean"))
        models.append(BaseModel.from_pretrained("JorisCos/ConvTasNet_Libri2Mix_sepclean_16k"))
        models.append(MultiDecoderDPRNN.from_pretrained("JunzheJosephZhu/MultiDecoderDPRNN").eval())
        models.append(BaseModel.from_pretrained("JorisCos/ConvTasNet_Libri2Mix_sepnoisy_8k"))
        models.append(BaseModel.from_pretrained("JorisCos/ConvTasNet_Libri3Mix_sepnoisy_16k"))
        models.append(BaseModel.from_pretrained("JorisCos/ConvTasNet_Libri2Mix_sepnoisy_16k"))
        models.append(BaseModel.from_pretrained("JorisCos/ConvTasNet_Libri3Mix_sepclean_16k"))
        
        modelStrings = []
        modelStrings.append("mpariente/DPRNNTasNet-ks2_WHAM_sepclean")
        modelStrings.append("JorisCos/ConvTasNet_Libri2Mix_sepclean_16k")
        modelStrings.append("JunzheJosephZhu/MultiDecoderDPRNN")
        modelStrings.append("JorisCos/ConvTasNet_Libri2Mix_sepnoisy_8k")
        modelStrings.append("JorisCos/ConvTasNet_Libri3Mix_sepnoisy_16k")
        modelStrings.append("JorisCos/ConvTasNet_Libri2Mix_sepnoisy_16k")
        modelStrings.append("JorisCos/ConvTasNet_Libri3Mix_sepclean_16k")
        
        dnn_models = []
        dnn_models.append(DeepEmotionRecognizer(emotions=['angry', 'sad', 'neutral', 'ps', 'happy'], balance=False, n_rnn_layers=1, n_dense_layers=1, rnn_units=128, dense_units=128, verbose=1))
        dnn_models.append(DeepEmotionRecognizer(emotions=['angry', 'sad', 'neutral', 'ps', 'happy'], balance=False, n_rnn_layers=2, n_dense_layers=2, rnn_units=128, dense_units=128, verbose=1))
        dnn_models.append(DeepEmotionRecognizer(emotions=['angry', 'sad', 'neutral', 'ps', 'happy'], balance=False, n_rnn_layers=3, n_dense_layers=3, rnn_units=128, dense_units=128, verbose=1))
        dnn_models.append(DeepEmotionRecognizer(emotions=['angry', 'sad', 'neutral', 'ps', 'happy'], balance=False, n_rnn_layers=1, n_dense_layers=1, rnn_units=256, dense_units=256, verbose=1))
        dnn_models.append(DeepEmotionRecognizer(emotions=['angry', 'sad', 'neutral', 'ps', 'happy'], balance=False, n_rnn_layers=2, n_dense_layers=2, rnn_units=256, dense_units=256, verbose=1))
        dnn_models.append(DeepEmotionRecognizer(emotions=['angry', 'sad', 'neutral', 'ps', 'happy'], balance=False, n_rnn_layers=3, n_dense_layers=3, rnn_units=256, dense_units=256, verbose=1))
        dnn_models.append(DeepEmotionRecognizer(emotions=['angry', 'sad', 'neutral', 'ps', 'happy'], balance=False, n_rnn_layers=1, n_dense_layers=1, rnn_units=128, dense_units=128, cell=GRU, verbose=1))
        dnn_models.append(DeepEmotionRecognizer(emotions=['angry', 'sad', 'neutral', 'ps', 'happy'], balance=False, n_rnn_layers=2, n_dense_layers=2, rnn_units=128, dense_units=128, cell=GRU, verbose=1))
        dnn_models.append(DeepEmotionRecognizer(emotions=['angry', 'sad', 'neutral', 'ps', 'happy'], balance=False, n_rnn_layers=3, n_dense_layers=3, rnn_units=128, dense_units=128, cell=GRU, verbose=1))
        dnn_models.append(DeepEmotionRecognizer(emotions=['angry', 'sad', 'neutral', 'ps', 'happy'], balance=False, n_rnn_layers=1, n_dense_layers=1, rnn_units=256, dense_units=256, cell=GRU, verbose=1))
        dnn_models.append(DeepEmotionRecognizer(emotions=['angry', 'sad', 'neutral', 'ps', 'happy'], balance=False, n_rnn_layers=2, n_dense_layers=2, rnn_units=256, dense_units=256, cell=GRU, verbose=1))
        dnn_models.append(DeepEmotionRecognizer(emotions=['angry', 'sad', 'neutral', 'ps', 'happy'], balance=False, n_rnn_layers=3, n_dense_layers=3, rnn_units=256, dense_units=256, cell=GRU, verbose=1))
        
        for dmodel in dnn_models:
            try:
                dmodel.train()
            except Exception as e:
                print(str(e))
        
        total = len(models)
        total_dnn = len(dnn_models)
        
        for j, dmodel in enumerate(dnn_models):
            for i, model in enumerate(models):
                with open(os.devnull, 'w') as devnull:
                    with contextlib.redirect_stdout(devnull):
                        try:
                            if isDir:
                                out.update({str(modelStrings[i]) + "_" + str(dnn_models[j].model_name) : __emotion_audio_dir__(path, model, dmodel, True)})
                            else:
                                out.update({str(modelStrings[i]) + "_" + str(dnn_models[j].model_name) : __emotion_audio__(path, model, dmodel, True)})
                        except Exception as e:
                            print(str(e))
                            continue
                print(str(i + 1) + "/" + str(total) + " on dnn model " + str(j + 1) + "/" + str(total_dnn))
    except Exception as e:
        print(str(e))
    finally:
        return out 
            

def __emotion_audio__(file_name, model=MultiDecoderDPRNN.from_pretrained("JunzheJosephZhu/MultiDecoderDPRNN").eval(), dnn=DeepEmotionRecognizer(emotions=['angry', 'sad', 'neutral', 'ps', 'happy'], balance=False, n_rnn_layers=2, n_dense_layers=2, rnn_units=128, dense_units=128, verbose=1), onlyDNN=False):
    global dnn_output, gemini_output
    
    try:
        if dnn_output is None:
            dnn_emotion = DNNAnalyzer(file_name, model, dnn, True)
        
        dnn_emotion.file_name = file_name
        dnn_emotion.__preprocess_file__(dnn_emotion.file_name)    
        dnn_output = dnn_emotion.get_emotion()
        
    except Exception as e:
        print(str(e))
        dnn_output = None   
    
    if not onlyDNN:
        try:
            if gemini_emotion is None:
                gemini_emotion = GeminiAnalyzer(file_name)
            
            gemini_emotion.converted_file = file_name
            gemini_output = gemini_emotion.get_emotion()
        except Exception as e:
            print(str(e))
            gemini_output = None

    print("Aggregate Emotion From DNN: " + str(dnn_output))
    print("Aggregate Emotion From Gemini: " + str(gemini_output))
    return file_name, dnn_output, gemini_output

def __emotion_audio_dir__(dir_name, model=MultiDecoderDPRNN.from_pretrained("JunzheJosephZhu/MultiDecoderDPRNN").eval(), dnn=DeepEmotionRecognizer(emotions=['angry', 'sad', 'neutral', 'ps', 'happy'], balance=False, n_rnn_layers=2, n_dense_layers=2, rnn_units=128, dense_units=128, verbose=1), onlyDNN=False):
    outputs = []
    for root, dirs, files in os.walk(dir_name):
        for file in files:
            if file.endswith(".mp3") or file.endswith(".wav"):
                file_path = os.path.join(root, file)
                file_path, dnn_output, gemini_output = __emotion_audio__(file_path, model, dnn, onlyDNN)
                outputs.append((file_path, dnn_output, gemini_output))
    print(str(outputs))
    return outputs

def __main__():
    #get arguments 
    parser = argparse.ArgumentParser(
                        prog='Emotional Audio-Analyzer',
                        description='Splits audio into individual waveforms, passes into emotional analyzer model, and returns the emotion')
    parser.add_argument('-i', '--input', help="input file to split and analyze", dest="input", required=True)
    parser.add_argument('--dir', help="the input arg becomes a directory target", dest="dir", action="store_true")
    parser.add_argument('--test', help="run an extensive test on the dnn", dest="test", action="store_true")
    parser.add_argument('--overwrite', help="truncate the csv file before writing", dest="overwrite", action="store_true")
    args = parser.parse_args()
    
    try: 
        file_name = str(args.input)
        out = {}
        
        if args.dir:
            if args.test:
                out = __emotion_audio_test__(file_name, True)
            else:
                out.update({file_name : __emotion_audio_dir__(file_name)})
        else:
            if args.test:
                out = __emotion_audio_test__(file_name, False)
            else:
                out.update({file_name : __emotion_audio__(file_name)})
        
        try:
            print("Writing CSV")
            csv_name = "results.csv"
            
            if args.test:
                csv_name = "test_results.csv"
                
            # Create a list of tuples from the dictionary items
            data = [(key, value) for key, value in out.items()]

            if args.overwrite:
                with open(csv_name, 'w'):
                    pass 
            
            # Open the CSV file in write mode
            with open(csv_name, 'a+', newline='') as csvfile:
                # Create a CSV writer
                writer = csv.writer(csvfile)
                writer.writerows(data)

        except ValueError as e:
            print(f"Error: {e}")
        except Exception as e:
            print(f"An unexpected error occurred: {e}")

    except Exception as e:
        tb_str = traceback.format_exception(etype=type(e), value=e, tb=e.__traceback__)
        print("".join(tb_str))
    
if __name__ == "__main__":
    __main__() 
