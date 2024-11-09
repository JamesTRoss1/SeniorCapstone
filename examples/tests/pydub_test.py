# install the following: pydub, PyAudio, ffmpeg (python3 -m pip install ffmpeg), simpleaudio

#import os
#import platform
#from pathlib import Path

from pydub import AudioSegment
from pydub.playback import play

"""platform_name = platform.system()
AudioSegment.ffmpeg = path_to_ffmpeg(platform_name)
if platform_name == 'Windows':
    os.environ["PATH"] += os.pathsep + str(Path(path_to_ffmpeg(platform_name)).parent)
else:
    os.environ["LD_LIBRARY_PATH"] += ":" + str(Path(path_to_ffmpeg(platform_name)).parent)

    
def path_to_ffmpeg(platform):
    SCRIPT_DIR = Path(__file__).parent 
    if platform == 'Windows':
        return str(Path(SCRIPT_DIR, "win", "ffmpeg", "ffmpeg.exe"))
    elif platform == 'Darwin':
        return str(Path(SCRIPT_DIR, "mac", "ffmpeg", "ffmpeg"))
    else:
        return str(Path(SCRIPT_DIR, "linux", "ffmpeg", "ffmpeg"))"""

#song = AudioSegment.from_mp3(r'C:\Users\kaush\Desktop\capstone\SeniorCapstone\examples\assets\roll.mp3')

wav_file = AudioSegment.from_file(file = r"C:\Users\kaush\Desktop\capstone\SeniorCapstone\examples\assets\audio.wav", 
                                  format = "wav") 
 

#play(song)
play(wav_file)
print("playing")