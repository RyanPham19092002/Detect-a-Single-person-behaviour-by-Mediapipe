# Detect-a-Single-person-behaviour-by-Mediapipe
This is a project detect single person behaviour using Mediapipe and LSTM. This project has 4 classes which are "Clapping", "Swinging hand", "Shaking head", "Nothing"

Mediapipe :
MediaPipe offers open source cross-platform, customizable ML solutions for live and streaming media. Some solutions of Mediapipe :
_Selfie Segmentation
_Face Mesh
_Hand tracking
_Human Pose Detection and Tracking 
_Hair Segmentation
_Object Detection and Tracking
_Face Detection
_Holistic Tracking
_3D Object Detection
And more solutions
In this project, we will use "Human Pose Detection and Tracking" solutions to get data of keypoint to make file weights by training data and print keypoint of skeleton on screen when we use webcam to detect behaviour.

![image](https://user-images.githubusercontent.com/122810752/232667756-f4eb6782-aaeb-4baf-bd18-3891dda9ecb6.png)

Source : https://developers.google.com/mediapipe/solutions/vision/pose_landmarker/

LSTM : Long short term memory 
This is an artificial neural network used in the fields of artificial intelligence and deep learning. It is a variety of RNN (recurrent neural network ) that are capable of learning long-term dependencies, especially in sequence prediction problems. LSTM has feedback connections, i.e., it is capable of processing the entire sequence of data, apart from single data points such as images.
The central role of an LSTM model is held by a memory cell known as a ‘cell state’ that maintains its state over time. The cell state is the horizontal line that runs through the top of the below diagram. It can be visualized as a conveyor belt through which information just flows, unchanged.

![image](https://user-images.githubusercontent.com/122810752/232667313-d8f8520e-7419-4865-bffb-23ae049b0366.png)

Information can be added to or removed from the cell state in LSTM and is regulated by gates. These gates optionally let the information flow in and out of the cell. It contains a pointwise multiplication operation and a sigmoid neural net layer that assist the mechanism.

![image](https://user-images.githubusercontent.com/122810752/232668048-fc6f4c64-fc73-4ed1-b34e-1486c10f9cf6.png)

The sigmoid layer gives out numbers between zero and one, where zero means ‘nothing should be let through,’ and one means ‘everything should be let through.’

Source : https://intellipaat.com/blog/what-is-lstm/

Steps to do this project : 

+ First Step : Preparing some packages
Mediapipe
Pandas
Opencv-python-headless
tensorflow
scikit-learn

+ Second Step : Make data
Read 600 frames from webcam and print keypoint in skeleton on screen by code of mediapipe and some defines : 
"
mpPose = mp.solutions.pose
pose = mpPose.Pose()
mpDraw = mp.solutions.drawing_utils 
"
When webcam turn on, you need to do an action that you want to detect after training model (in this project is 4 actions).
After reading 600 frames, the webcam will turn off and then the data of action will save a point coordinates of 33 keypoints which each other has 4 values, including : x,y,z,visibility in file text has name : label + ".txt" (which label is the name of each action).

+ Thrid Step: Train LSTM model
_With files txt has data of actions, we get this data in a second column form of file txt and save it in numpy dataset
We make a loop to make the data input and the data output, input data has 10 frames length . After that, we will make the y numpy array to save output is a number to categorize of each class when prdeict (0,1,2..).
_Build the LSTMs model and get the file weights .h5 

+ Fourth Step : Inference LSTM
Load model with file weights we had in step 3, predict the actions , draw keypoint in skeleton and label on screen based on model predict.
