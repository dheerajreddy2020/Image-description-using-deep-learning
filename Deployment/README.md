## Image-Caption-Generator-Flask-Deployment
This is a demo project to elaborate how Machine Learn Models are deployed on production using Flask API

### Prerequisites
You must have OpenCV, Tensorflow to run the deep-learning model and Flask (for API) installed.

### Project Structure
This project has four major parts :
1. app.py - This contains Flask APIs that receives the uploaded image through GUI or API calls, and computes the precited sentence based on the model and returns it.
2. templates - This folder contains the HTML template to allow user to upload images to generate a description.
3. '.pkl' files - Two .pkl files which are the dictionaries to convert from word to embedding vector and from embedding vector to word.
4. '.h5' files - Two .h5 files which are the deep learning models, one model is to create the features for encoder model

### Running the project
1. Run app.py using below command to start Flask API
```
python app.py
```
By default, flask will run on port 5000.

2. Navigate to URL http://localhost:5000

Upload a valid image and hit Predict.
If everything goes well, you should  be able to see the predcited image description
