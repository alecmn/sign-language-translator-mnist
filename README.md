# sign-language-translator-mnist

This repository contains software and data for performing sign language translation training and testing on a mnist sign language dataset. Additionally, there is support for a live camera feed sign language translation.



# Requirements and setup
You will need to install sklearn, numpy, pytorch, and onnx

# Data
The data used is taken from a kaggle dataset called [Sign Language MNIST](https://www.kaggle.com/datasets/datamunge/sign-language-mnist?datasetId=3258&select=amer_sign2.png) which is an MNIST adaptation of a sign language dataset.


# Reproducing the experiments

Running the system can be split up into 3 steps:

1. To train the network run [train.py](train.py) where you can specify any of the models to be used that can be found in [networks.py](networks.py). 
2. Running [evaluation.py](evaluation.py) gives you the results of testing the network on the test set and also create a production version of the model with onnx.
3. For a live feed where it translate your hand signal in real-time you can run the [live_translation.py](live_translation.py) file after doing the previous two steps.
# Contact us
This repository is created by Alec Nonnemaker and Sara Boby.
Feel free to email us if you have any questions at **alec.michael@live.nl** or **sboby1720@gmail.com**






