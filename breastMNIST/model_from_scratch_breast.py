from pathlib import Path
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
import tensorflow as tf
import medmnist
from medmnist import INFO
from keras.preprocessing.image import ImageDataGenerator
from sklearn.utils import class_weight
import numpy as np

np.random.seed(69)

data_flag = 'breastmnist'
download = True


info = INFO[data_flag]
task = info['task']
n_channels = info['n_channels']
n_classes = len(info['label'])

DataClass = getattr(medmnist, info['python_class'])
#load data
train_dataset = DataClass(split='train', download=download)
val_dataset = DataClass(split='val', download=download)
test_dataset = DataClass(split='test', download=download)
# print('Training Dataset:')
# print(train_dataset)
# print(train_dataset.imgs.shape)
if train_dataset.imgs.ndim == 3:
    train_dataset.imgs = np.expand_dims(train_dataset.imgs, axis=-1)
  # print("Adding channel to images...")

# print('Validation Dataset:')
# print(val_dataset)
# print(val_dataset.imgs.shape)
if val_dataset.imgs.ndim == 3:
    val_dataset.imgs = np.expand_dims(val_dataset.imgs, axis=-1)
# print('Testing Dataset:')
# print(test_dataset)
# print(test_dataset.imgs.shape)
if test_dataset.imgs.ndim == 3:
    test_dataset.imgs = np.expand_dims(test_dataset.imgs, axis=-1)
#preprocessing
train_dataset.imgs = train_dataset.imgs/255.0
val_dataset.imgs = val_dataset.imgs/255.0
test_dataset.imgs = test_dataset.imgs/255.0
#class weight imbalance
breast_class_weights = class_weight.compute_class_weight('balanced',
                                                         classes = np.unique(train_dataset.labels[:,0]),
                                                         y = train_dataset.labels[:, 0])


# print(breast_class_weights)

weights = { 0 : breast_class_weights[0], 1 : breast_class_weights[1] }

# print(weights)
print("Preprocessing and augmentation of images with Standardisation, Rotation and Horizontal Flips..")
datagen = ImageDataGenerator(rotation_range=10,
                             horizontal_flip=True)
my_callbacks = [
    tf.keras.callbacks.EarlyStopping(patience=10)
    # tf.keras.callbacks.ModelCheckpoint(filepath='model.{epoch:02d}-{val_loss:.2f}.h5'),
]

model = Sequential()

model.add(Conv2D(32, (3, 3), padding="same", activation="relu", input_shape=(28, 28, 1)))
model.add(MaxPooling2D(pool_size=(2,2), strides=(1,1)))

model.add(Conv2D(32, (3, 3), padding="same", activation="relu"))
model.add(Conv2D(64, (3, 3), padding="same", activation="relu"))
model.add(Conv2D(128, (3, 3), padding="same", activation="relu"))
model.add(MaxPooling2D(pool_size=(2,2), strides=(1,1)))

model.add(Flatten())
model.add(Dense(128, activation="relu"))
model.add(Dropout(0.5))
model.add(Dense(1, activation="sigmoid"))

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

model.fit(datagen.flow(train_dataset.imgs, train_dataset.labels),
                    epochs=50,
                    callbacks=my_callbacks,
                    validation_data=(val_dataset.imgs, val_dataset.labels),
                    shuffle=True,
                    class_weight=weights)

#saving neural network structure
model_structure = model.to_json()
f = Path("model_from_scratch.json_breast.json")
f.write_text(model_structure)

#save neural network's saved weights
model.save_weights("model_from_scratch_breast.h5")