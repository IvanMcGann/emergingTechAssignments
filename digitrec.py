# Adapted from https://github.com/ianmcloughlin/jupyter-teaching-notebooks/blob/master/mnist.ipynb
import keras as kr
from keras.models import load_model
import sklearn.preprocessing as pre
import gzip
import numpy as np

#import matplotlib.pyplot as plt
from PIL import Image

# Initialize a global enconder.
with gzip.open('data/t10k-labels-idx1-ubyte.gz', 'rb') as f:
		test_lbl = f.read()
test_lbl = np.array(list(test_lbl[ 8:])).astype(np.uint8)
encoder = pre.LabelBinarizer()
encoder.fit(test_lbl)

# Global model variable
model = None

def load():
	global model
	filename = input("Please enter a HDF5 file to load: ")
	model = load_model(filename)
	model.summary()

def construct():
	global model
	if model:
		confirmation = input("\nDo you want to delete the saved model? (y/n) ")
		if confirmation == "y":
			del model
		elif confirmation == "n":
			return
	
	model = kr.models.Sequential()
			
	model.add(kr.layers.Dense(units=600, activation='linear', input_dim=784))
	model.add(kr.layers.Dense(units=400, activation='relu'))

	model.add(kr.layers.Dense(units=10, activation='softmax'))
	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
	
	model.summary
	

def train():
	global model
	global encoder
	
	if not model:
		print("No model found. Please create or load your model first")
		return
	
	with gzip.open('data/train-images-idx3-ubyte.gz', 'rb') as f:
		train_img = f.read()

	with gzip.open('data/train-labels-idx1-ubyte.gz', 'rb') as f:
		train_lbl = f.read()
		
	train_img = ~np.array(list(train_img[16:])).reshape(60000, 28, 28).astype(np.uint8) / 255.0
	train_lbl =  np.array(list(train_lbl[ 8:])).astype(np.uint8)

	inputs = train_img.reshape(60000, 784)
	outputs = encoder.transform(train_lbl)

	model.summary()
	
	model.fit(inputs, outputs, epochs=2, batch_size=100)
	
def test():
	global model
	global encoder
	global test_lbl
	
	if not model:
		print("No model found. Please load a model first")
		return
	
	with gzip.open('data/t10k-images-idx3-ubyte.gz', 'rb') as f:
		test_img = f.read()
		
	test_img = ~np.array(list(test_img[16:])).reshape(10000, 784).astype(np.uint8) / 255.0
	
	model.summary()
	
	result_set = (encoder.inverse_transform(model.predict(test_img)) == test_lbl).sum()
	percent = (result_set/10000)*100
	print("\nThe model has succesfully made ", result_set, " predictions out of 10000 tests (", percent, "%)")
	

def save():
	global model

	if not model:
		print("No model found, create or load the model")
		return

	filename = input("Please enter the name of the model: ")
	model.save(filename)
	
def png_read():
	if not model:
		print("No model found, create or load the model")
		return
	
	filename = input("Please enter the name of the PNG image file: ")
	img = Image.open(filename).convert("L")
	
	img_width = 28
	img_height = 28
	
	if (img.size[0] != 28) or (img.size[1] != 28):
		img.thumbnail((img_width,img_height), Image.ANTIALIAS)
	
	one_dim =  img_width*img_height
	
	im2arr = np.array(img.getdata())
	im2arr = np.array(list(im2arr)).reshape(1, one_dim).astype(np.uint8) / 255.0

	predImg = model.predict(im2arr)
	result_set = encoder.inverse_transform(predImg)

	print("The program predicts the image is a:", result_set)


choice = True
while choice:
	print("""
	1. Load Model
	2. Create a NN model
	3. Train with MNIST training images
	4. Test using MNIST testing images
	5. Save model
	6. Read and predict from a PNG file
	7. Exit
	""")
	choice = input("Menu: ")
	
	if choice == "1":
		load()
	elif choice =="2":
		construct()
	elif choice =="3":
		train()
	elif choice=="4":
		test()
	elif choice =="5":
		save()
	elif choice =="6":
		png_read()
	elif choice=="7":
		choice = None
	else:
		print("Invalid choice, enter a valid number.") 