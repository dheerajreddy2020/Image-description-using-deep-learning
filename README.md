# Image Caption Generator
The goal of this project is to describe a scene from an image

## Usage scenarios
1. Self Driving Cars – If we can develop the scene around the car, it can boost the self driving system
2. Aid to the blind – We can create a guide to the blind by first converting scene to text and then converting text to speech
3. CCTV cameras – Alarms can be raised using this technology if we detect malicious activity going on somewhere

## Model Architecture
![model-architecture](https://github.com/dheerajreddy2020/Image-description-using-deep-learning/blob/master/Model-Architecture.PNG)

## Available Dataset
Flickr-30k : Over 30,000 images with more 5 descriptions for each image.

### Encoder Model:
This model is used to extract the features from an image to later pass it onto the sentence generation model
In this project xception model is used as the encoder model to extract features from the images
Reference : https://www.tensorflow.org/api_docs/python/tf/keras/applications/Xception

### Decoder model:
This is a sentence generation model. 
The sentence is generated word by word as shown in the example below using the LSTM model to predict the next word.

![sentence-generation](https://github.com/dheerajreddy2020/Image-description-using-deep-learning/blob/master/Sentence-generator.PNG)
