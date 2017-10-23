import csv
import cv2
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Flatten, Lambda, Cropping2D, Convolution2D, MaxPooling2D, PReLU
from keras import regularizers
from keras.regularizers import l2
from keras.layers.normalization import BatchNormalization
from keras import initializations
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import sklearn
import matplotlib.pyplot as plt
import random

samples = []
images, measurments = [], []
augmented_images, augmented_measurments = [], []
absulut = 0.2
correction = 0.2
gray = [0.299, 0.587, 0.114]

with open('full_driving_log.csv') as csvfile:    
    reader = csv.reader(csvfile)
    for line in reader:
        if line[3] == 'steering': continue
        measurment = float(line[3])
        if abs(measurment) == 1  and random.random() < 0.5:
            continue
        if abs(measurment) < 0.2  and random.random() < 0.95:
            continue
        if 0.2 < abs(measurment) < 0.6 and random.random() < 0.75:
            continue
        twice = True
        while  twice:
            twice = False
            for i in range(3):
                source_path = line[i]
                filename = source_path.split('\\')[-1]
                filename = filename.split('/')[-1]
                filename = filename.replace(' ', "")
                current_path = 'full_IMG/' + filename
                image = cv2.imread(current_path)
                if type(image) is None: continue
                if i == 1: measurment = measurment + correction
                if i == 2: measurment = measurment - correction
                measurments.append(measurment)
                images.append(image)

    for image, measurment in zip(images, measurments):
            augmented_images.append(image)
            augmented_measurments.append(measurment)
            if abs(measurment) > absulut:
                augmented_images.append(cv2.flip(image, 1))
                augmented_measurments.append(measurment * -1.0)

for i, j in zip(augmented_images, augmented_measurments):
    samples.append([i, j])
print ('len samples:', len(samples))
train_samples, validation_samples = train_test_split(samples, test_size=0.2)

def generator(samples, batch_size=64, augment=True):
    num_samples = len(samples)
    print ('generator num_samples:', num_samples)
    while 1: # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            X_train = np.array([row[0] for row in batch_samples])
            y_train = np.array([row[1] for row in batch_samples])
            if type(X_train) is None: continue
            X_train = np.dot(X_train[...,:3], gray)
            X_train = np.reshape(X_train, X_train.shape + (1,))
            if X_train[0].shape != (160, 320, 1): continue
            yield sklearn.utils.shuffle(X_train, y_train)

train_generator = generator(train_samples, batch_size=64)
validation_generator = generator(validation_samples, batch_size=64)

# plt.hist(augmented_measurments, bins = 5,label = 'train_Y')
# plt.xlabel("angle_size")
# plt.ylabel("number of samples")
# plt.legend(numpoints = 1)
# plt.show()

# model = Sequential()
# model.add(Lambda(lambda x: x / 127.5 - 1, input_shape=(160, 320, 1)))
# model.add(Cropping2D(cropping=((70, 25), (0, 0))))
# model.add(Convolution2D(3, 5, 5, activation='relu'))
# model.add(Convolution2D(24, 5, 5, activation='relu'))
# model.add(Convolution2D(36, 5, 5, activation='relu'))
# model.add(Convolution2D(48, 3, 3, activation='relu'))
# model.add(Convolution2D(64, 3, 3, activation='relu'))
# model.add(Flatten())
# model.add(Dense(100))
# model.add(Dense(50))
# model.add(Dense(10))
# model.add(Dense(1))
# model.compile(loss='mse', optimizer='adam')

# # model.fit(X_train, y_train, validation_split=0.2, shuffle=True)

models = Sequential()
models.add(Cropping2D(cropping=((50,20), (1,1)), input_shape=(160, 320, 1)))
models.add(Lambda(lambda x: x/255.0 - 0.5))
models.add(Convolution2D(24,5,5, subsample=(2,2), init=initializations.he_uniform))
models.add(PReLU())
models.add(BatchNormalization())
models.add(Convolution2D(36,5,5, subsample=(2,2), init=initializations.he_uniform))
models.add(PReLU())
models.add(BatchNormalization())
models.add(Convolution2D(48,5,5, subsample=(2,2), init=initializations.he_uniform))
models.add(PReLU())
models.add(BatchNormalization())
models.add(Convolution2D(64,3,3, init=initializations.he_uniform))
models.add(PReLU())
models.add(BatchNormalization())
# models.add(MaxPooling2D()) #
models.add(Flatten())
models.add(Dense(100, init=initializations.he_uniform))
models.add(BatchNormalization()) #
models.add(Dense(50, init=initializations.he_uniform))
models.add(BatchNormalization()) #
models.add(Dense(10, init=initializations.he_uniform))
models.add(BatchNormalization()) #
models.add(Dense(1, init=initializations.he_uniform))
models.compile(optimizer='adam', loss='mse')

history_object = models.fit_generator(train_generator, samples_per_epoch = len(train_samples), validation_data = validation_generator, nb_val_samples = len(validation_samples),
	nb_epoch=10, verbose=1)


### print the keys contained in the history object
print(history_object.history.keys())

### plot the training and validation loss for each epoch
# plt.plot(history_object.history['loss'])
# plt.plot(history_object.history['val_loss'])
# plt.title('model mean squared error loss')
# plt.ylabel('mean squared error loss')
# plt.xlabel('epoch')
# plt.legend(['training set', 'validation set'], loc='upper right')
# plt.show()

models.save('model.h5')