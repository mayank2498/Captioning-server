
from pickle import load
from keras.models import load_model
#### extract features from each photo in the directory

from os import listdir
from keras.applications.vgg16 import VGG16
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.applications.vgg16 import preprocess_input
from keras.models import Model
from keras.models import model_from_json
from keras.preprocessing.sequence import pad_sequences
from numpy import argmax

from keras import backend as K






tokenizer = load(open('/home/mayank/Desktop/NeuralNetwork/Image Captioning/tokenizer.pkl', 'rb'))
print("tokenizer loaded")



def extract_features(image,model):
	# load the model
	
	# re-structure the model
	model.layers.pop()
	model = Model(inputs=model.inputs, outputs=model.layers[-1].output)
	# load the photo
	#image = load_img(filename, target_size=(224, 224))
	# convert the image pixels to a numpy array
	#image = img_to_array(image)
	# reshape data for the model
	image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
	
	# prepare the image for the VGG model
	image = preprocess_input(image)
	# get features
	feature = model.predict(image, verbose=0)
	return feature



###########################################################


# map an integer to a word
def word_for_id(integer, tokenizer):
	for word, index in tokenizer.word_index.items():
		if index == integer:
			return word
	return None
 
# generate a description for an image
def generate_desc(model, tokenizer, photo, max_length):
	# seed the generation process
	in_text = 'startseq'
	# iterate over the whole length of the sequence
	for i in range(max_length):
		# integer encode input sequence
		sequence = tokenizer.texts_to_sequences([in_text])[0]
		# pad input
		sequence = pad_sequences([sequence], maxlen=max_length)
		# predict next word
		yhat = model.predict([photo,sequence], verbose=0)
              
		# convert probability to integer
		yhat = argmax(yhat)
		# map integer to word
		word = word_for_id(yhat, tokenizer)
		# stop if we cannot map the word
		if word is None:
			break
		# append as input for generating the next word
		in_text += ' ' + word
		# stop if we predict the end of the sequence
		if word == 'endseq' or word == 'startseq':
			break
	return in_text
 

def predict(image):
	

	# load the tokenizer
	# pre-define the max sequence length (from training)
	max_length = 34
	# load the model


	model = VGG16()
	print("VGG model loaded!")
	

	# load and prepare the photograph
	photo = extract_features(image,model)
	print("features of photo extracted")
	# generate descriptio)n


		# load json and create model
	json_file = open('/home/mayank/Desktop/NeuralNetwork/Image Captioning/very_model.json', 'r')
	loaded_model_json = json_file.read()
	json_file.close()
	loaded_model = model_from_json(loaded_model_json)
	# load weights into new model
	weights = '/home/mayank/Desktop/NeuralNetwork/Image Captioning/predictions/training2/model_8.h5'
	loaded_model.load_weights(weights)
	print("Loaded model from disk")

	description = generate_desc(loaded_model, tokenizer, photo, max_length)
	K.clear_session()
	return description
	