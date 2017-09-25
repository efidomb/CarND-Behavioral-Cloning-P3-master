import csv
import cv2
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Flatten, Lambda, Cropping2D, Convolution2D, MaxPooling2D
from keras import regularizers
from keras.regularizers import l2
from keras.layers.normalization import BatchNormalization
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import sklearn
import matplotlib.pyplot as plt

samples = []
with open('driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        samples.append(line)

print ('len samples:', len(samples))
train_samples, validation_samples = train_test_split(samples, test_size=0.2)
correction = 0.2
gray = [0.299, 0.587, 0.114]

def generator(samples, batch_size=64):
    num_samples = len(samples)
    print ('generator num_samples:', num_samples)
    while 1: # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images, measurments = [], []
            for sample in batch_samples:
                # if sample[6] == 0: continue
                for i in range(3):
                    source_path = sample[i]
                    filename = source_path.split('\\')[-1]
                    filename = filename.replace(' ', "")
                    if filename[0] != 'I':
                        filename = 'IMG/' + filename
                    # if filename[0] == 'I':
                    #     current_path = 'full_' + filename
                    # else:
                    #     current_path = 'full_IMG/' + filename
                    # current_path = 'IMG/' + filename
                    current_path = filename
                    image = cv2.imread(current_path)
                    if type(image) is None: continue
                    images.append(image)
                    if sample[3] == 'steering':
                        measurment = 0
                    else:
                        measurment = float(sample[3])
                    if i == 1: measurment = measurment + correction
                    if i == 2: measurment = measurment - correction
                    measurments.append(measurment)

            augmented_images, augmented_measurments = [], []
            for image, measurment in zip(images, measurments):
                augmented_images.append(image)
                augmented_measurments.append(measurment)
                augmented_images.append(cv2.flip(image, 1))
                augmented_measurments.append(measurment * -1.0)

            X_train = np.array(augmented_images)
            y_train = np.array(augmented_measurments)
            if type(X_train) is None: continue
            X_train = np.dot(X_train[...,:3], gray)
            X_train = np.reshape(X_train, X_train.shape + (1,))
            if X_train[0].shape != (160, 320, 1): continue
            yield sklearn.utils.shuffle(X_train, y_train)

train_generator = generator(train_samples, batch_size=64)
validation_generator = generator(validation_samples, batch_size=64)

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
models.add(Convolution2D(24,5,5, subsample=(2,2), activation='relu', W_regularizer = l2(0.001)))
models.add(BatchNormalization())
models.add(Convolution2D(36,5,5, subsample=(2,2), activation='relu', W_regularizer = l2(0.001)))
models.add(BatchNormalization())
models.add(Convolution2D(48,5,5, subsample=(2,2), activation='relu', W_regularizer = l2(0.001)))
models.add(BatchNormalization())
models.add(Convolution2D(64,3,3,activation='relu', W_regularizer = l2(0.001)))
models.add(BatchNormalization())
# models.add(MaxPooling2D()) #
models.add(Flatten())
models.add(Dense(100, W_regularizer = l2(0.001)))
# models.add(BatchNormalization()) #
models.add(Dense(50, W_regularizer = l2(0.001)))
# models.add(BatchNormalization()) #
models.add(Dense(10, W_regularizer = l2(0.001)))
# models.add(BatchNormalization()) #
models.add(Dense(1, W_regularizer = l2(0.001)))
models.compile(optimizer='adam', loss='mse', metrics=['accuracy'])

history_object = models.fit_generator(train_generator, samples_per_epoch = len(train_samples), validation_data = validation_generator, nb_val_samples = len(validation_samples),
	nb_epoch=5, verbose=1)

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