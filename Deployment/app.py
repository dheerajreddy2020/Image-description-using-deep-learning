from __future__ import division, print_function
# coding=utf-8
import sys
import glob
import re
import numpy as np


# Flask utils
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer

from pickle import load
import cv2
import os
import matplotlib.pyplot as plt
from keras_preprocessing.image import img_to_array
import numpy as np


def load_descriptions(filename):
	# load all features
	descriptions = load(open(filename, 'rb'))
	# filter features
	return descriptions

from numpy import argmax
from tensorflow.keras.preprocessing.sequence import pad_sequences
def generate_description(model, photo, wordtoix, ixtoword, max_length, evaluation=False):
	# loop for ever over images
  # retrieve the photo feature
  in_text = 'startseq'
  #photo = photos[im_id]
  for i in range(max_length):
    sequence = [wordtoix[word] for word in in_text.split(' ') if word in wordtoix]
    sequence = pad_sequences([sequence], maxlen=max_length)
    yhat = model.predict([photo,sequence], verbose=0)
    yhat = argmax(yhat)
    word = ixtoword[yhat]
    if word is None:
      break
    in_text += ' '+word
    if word == 'endseq':
      break
  word_list = in_text.split(' ')[1:-1]
  final_sentence = ' '.join(word for word in word_list)
  if evaluation:
    return in_text
  else:
    return final_sentence

from tensorflow.keras.applications.xception import preprocess_input
def get_encoder_features(image,model):
  image = cv2.resize(image,(299,299))
  image = img_to_array(image)
  # reshape data for the model
  image = np.expand_dims(image, axis=0)
  # prepare the image for the model
  image = preprocess_input(image)
  #image = image/image.max()
  # get features
  feature = model.predict(image, verbose=0)
  np.reshape(feature, feature.shape[1])
  return feature

wordtoix = load_descriptions('wordtoix.pkl')
ixtoword = load_descriptions('ixtoword.pkl')
max_length = 74

## Loading Encoder and Decoder Models
from keras.models import load_model
encoder_model = load_model('Xception_encoder.h5')
decoder_model = load_model('Xception_200dwordvec_model.h5')

app = Flask(__name__)

@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def upload():
	if request.method == 'POST':
		# Get the file from post request
		f = request.files['file']
        # Save the file to ./uploads
		basepath = os.getcwd()
		try:
			os.mkdir('uploads')
		except:
			next
		file_path = os.path.join(
			basepath, 'uploads', secure_filename(f.filename))
		f.save(file_path)
		img = cv2.imread(file_path)
		plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
		feature = get_encoder_features(img,encoder_model)
		output = generate_description(decoder_model,feature, wordtoix, ixtoword, max_length)
		return output
	return None


if __name__ == '__main__':
    app.run(debug=True)

