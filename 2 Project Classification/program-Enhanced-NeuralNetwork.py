#!/usr/bin/python
#
#  The program was written by Raymond WONG.
#  The program is used for illustrating how to write a program in Keras for Neural Network
#

from keras.models import Sequential
from keras.layers import Dense
from keras.models import model_from_json
import numpy

# to train a model
def trainModel(trainingDataFilename):
	# to set the "fixed" seed of a random number generator used in the "optimization" tool
	# in the neural network model
	# The reason why we fix this is to reproduce the same output each time we execute this program
	# In practice, you could set it to any number (or, the current time) (e.g., "numpy.random.seed(time.time())")
	numpy.random.seed(11)
	
	# Step 1: to load the data
	print("  Step 1: to load the data...")
	#   Step 1a: to read the dataset with "numpy" function
	print("    Step 1a: to read the dataset with \"nump\" function...")
	dataset = numpy.loadtxt(trainingDataFilename, delimiter=",")
	
	#   Step 1b: to split the dataset into two datasets, namely the input attribute dataset (X) and the target attribute dataset (Y)
	print("    Step 1b: to split the dataset into two datasets, namely the input attribute dataset (X) and the target attribute dataset (Y)...")
	X = dataset[:,0:8]
	Y = dataset[:,8]
	
	# Step 2: to define the model
	print("  Step 2: to define the model...")
	model = Sequential()
	model.add(Dense(12, input_dim=8, activation='relu'))
	model.add(Dense(8, activation='relu'))
	model.add(Dense(1, activation='sigmoid'))
	
	# Step 3: to compile the model
	print("  Step 3: to compile the model...")
	model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
	
	# Step 4: To fit the model
	print("  Step 4: to fit the model...")
	model.fit(X, Y, validation_split=0.2, epochs=150, batch_size=10)
	
	# Step 5: To evaluate the model
	print("  Step 5: to evaluate the model...")
	scores = model.evaluate(X, Y)
	print("")
	print("{}: {}".format(model.metrics_names[1], scores[1]*100))
	
	return model

# to save a model
def saveModel(model, modelFilenamePrefix):
	# Step 1: to save the model structure to a file in the JSON format
	print("  Step 1: to save the model structure to a file in the JSON format...")
	structureFilename = modelFilenamePrefix + ".json"
	model_json = model.to_json()
	with open(structureFilename, "w") as f:
	    f.write(model_json)
	
	# Step 2: to save the model weight information to a file in the HDF5 format
	print("  Step 2: to save the model weight information to a file in the HDF5 format...")
	weightFilename = modelFilenamePrefix + ".h5"
	model.save_weights(weightFilename)

# to read a model
def readModel(modelFilenamePrefix):
	# Step 1: to load the model structure from a file in the JSON format
	print("  Step 1: to load the model structure from a file in the JSON format...")
	structureFilename = modelFilenamePrefix + ".json"
	with open(structureFilename, "r") as f:
		model_json = f.read()
	model = model_from_json(model_json)
	
	# Step 2: to load the model weight information from a file in the HDF5 format
	print("  Step 2: to load the model weight information from a file in the HDF5 format...")
	weightFilename = modelFilenamePrefix + ".h5"
	model.load_weights(weightFilename)
	
	return model
	
# to predict the target attribute of a new dataset based on a model	
def predictNewDatasetFromModel(newInputAttributeDataFilename, newTargetAttributeDataFilename, model):

	# Step 1: to load the new data (input attributes)
	print("  Step 1: to load the new data with \"nump\" function...")
	newX = numpy.loadtxt(newInputAttributeDataFilename, delimiter=",")
		
	# Step 2: to predict the target attribute of the new data based on a model
	print("  Step 2: Step 2: to predict the target attribute of the new data based on a model...")
	newY = model.predict(newX, batch_size=10)
	
	# Step 3: to save the predicted target attribute of the new data into a file
	print("  Step 3: to save the predicted target attribute of the new data into a file...")
	numpy.savetxt(newTargetAttributeDataFilename, newY, delimiter=",", fmt="%.10f")
		
# the main function
def main():
	trainingDataFilename = "Training-Dataset1-NoOfDim-8-Target-Binary.csv"
	newInputAttributeDataFilename = "New-Dataset1-NoOfDim-8-Target-None.csv"
	newTargetAttributeDataFilename = "New-Dataset1-Target-Output.csv"
	modelFilenamePrefix = "neuralNetworkModel"
	
	# Phase 1: to train the model
	print("Phase 1: to train the model...")
	model = trainModel(trainingDataFilename)
	
	# Phase 2: to save the model to a file
	print("Phase 2: to save the model to a file...")
	saveModel(model, modelFilenamePrefix)

	# Phase 3: to read the model from a file
	print("Phase 3: to read the model from a file...")
	model = readModel(modelFilenamePrefix)	
	
	# Phase 4: to predict the target attribute of a new dataset based on a model
	print("Phase 4: to predict the target attribute of a new dataset based on a model...")
	predictNewDatasetFromModel(newInputAttributeDataFilename, newTargetAttributeDataFilename, model)

main()

