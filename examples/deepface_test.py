#imports 
from deepface import DeepFace 

#launches livestream video and performs facial analysis and recognition 
#probably most efficient to use open-cv instead and just pass in single images of faces one at a time 
#this technique could allow for multitprocessing since each process could process a different face in an image
#and then come together to aggregate the room's emotional sentiment 
DeepFace.stream(db_path="examples/deepface_db")