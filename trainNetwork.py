from keras import models, layers, optimizers
from keras.utils import to_categorical
from keras.datasets import mnist
from keras.preprocessing.image import ImageDataGenerator

# Import mnist dataset
(trainImages, trainLabels), (testImages, testLabels) = mnist.load_data()

# Reshape & Normalize images
trainImages = trainImages.reshape(60000, 28, 28, 1)
trainImages.astype("float32") / 255
testImages = testImages.reshape(10000, 28, 28, 1)
testImages.astype("float32") / 255

# Convert labels to one-hot encoded form
trainLabels = to_categorical(trainLabels)
testLabels = to_categorical(testLabels)

# Create generators for for train and test sets - apply image manipulations to the train set but not test
gen = ImageDataGenerator(rotation_range=8, width_shift_range=0.08, shear_range=0.3, height_shift_range=0.08, zoom_range=0.08)
test_gen = ImageDataGenerator()

# Connect generators to images & labels
trainGenerator = gen.flow(trainImages, trainLabels, batch_size=64)
testGenerator = test_gen.flow(testImages, testLabels)

# Convert labels to softmax categorical form
trainLabels = to_categorical(trainLabels)
testLabels = to_categorical(testLabels)

# Initialise neural network
network = models.Sequential()
network.add(layers.Conv2D(32, (3,3), activation="relu", input_shape=(28, 28,1)))
network.add(layers.BatchNormalization(axis=-1))
network.add(layers.MaxPooling2D(2, 2))
network.add(layers.Conv2D(64, (3,3), activation="relu"))
network.add(layers.BatchNormalization(axis=-1))
network.add(layers.MaxPooling2D(2, 2))
network.add(layers.Flatten())
network.add(layers.Dense(128, activation="relu"))
network.add(layers.Dropout(0.2))
network.add(layers.Dense(10, activation="softmax"))
network.summary()

# Compile network
network.compile(loss="categorical_crossentropy", optimizer="Adam", metrics=["acc"])

# Train from generators (// = divide and floor)
network.fit_generator(trainGenerator, steps_per_epoch=60000//64, epochs=5, validation_data=testGenerator, validation_steps=10000 // 64)

# Convert network description to json
modelJSON = network.to_json()
with open("model.json", "w") as json_file:
    json_file.write(modelJSON)

# Convert weights to .h5 file
network.save("network.h5")
print("Network successfully saved to disk")
