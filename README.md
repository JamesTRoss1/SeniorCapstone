# Adaptive Announcement System (Honeywell)

## Description: 

The adaptive announcement system involves taking input from either an audio or visual device and then applying computer vision to the captured photo to recognize multiple faces and people in the image. The system will then take the recognized faces and determine the emotional status of each person in the environment. After the emotions of each person is determined, the system will apply an averaging algorithm to determine the mean emotion of the room. With the average emotion of the room determined, this output gets fed into a response generated tuned with a language model which will output a uniquely tuned statement depending on the emotion of the room. This language model will be trained on generating statements that will hopefully positively impact the room group emotional environment through the choice of words selected by the model. The system will either output a text statement or a vocal statement. This idea is new as many airport announcement systems, which often are delivering unfortunate news that may cause anger or dismay, fail to consider the audience and fail to make the attempt to relate to the current environment with a certain empathy that passengers need or demand at that current point in time. 

## Requirements:
- Windows 

## Installation:

1. Create conda environment 
```
conda create -n myenv 
```

2. Activate the environment 
```
conda activate myenv 
```
3. Install the necessary packages
```
conda install -f requirements.txt 
```
4. Navigate to the src directory and run emotion_analysis.py 
```
python emotion_analysis.py
```
