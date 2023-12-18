from pathlib import Path

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from sklearn.utils import class_weight
import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
import medmnist
from medmnist import INFO
from keras.preprocessing.image import ImageDataGenerator

data_flag = 'bloodmnist'
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
#preprocessing
train_dataset.imgs = train_dataset.imgs/255.0
val_dataset.imgs = val_dataset.imgs/255.0
test_dataset.imgs = test_dataset.imgs/255.0
#padding images
train_dataset.imgs = np.pad(train_dataset.imgs, ((0,0),(2,2),(2,2),(0,0)), 'constant')
val_dataset.imgs = np.pad(val_dataset.imgs, ((0,0),(2,2),(2,2),(0,0)), 'constant')
#onehot encoding
train_labels = to_categorical(train_dataset.labels,num_classes=8)
val_labels = to_categorical(val_dataset.labels, num_classes=8)
datagen = ImageDataGenerator(rotation_range = 30,brightness_range=[0.4,1.5],horizontal_flip = True)

blood_class_weights = class_weight.compute_class_weight('balanced',
                                                         classes = np.unique(train_dataset.labels[:,0]),
                                                         y = train_dataset.labels[:, 0])
my_callbacks = [
    tf.keras.callbacks.EarlyStopping(patience=10)
    # tf.keras.callbacks.ModelCheckpoint(filepath='model.{epoch:02d}-{val_loss:.2f}.h5'),
]

print(blood_class_weights)

weights = {0 : blood_class_weights[0],
           1 : blood_class_weights[1],
           2 : blood_class_weights[2],
           3 : blood_class_weights[3],
           4 : blood_class_weights[4],
           5 : blood_class_weights[5],
           6 : blood_class_weights[6],
           7 : blood_class_weights[7] }
np.random.seed(69)
#model
vgg16 = tf.keras.applications.vgg16
conv_model = vgg16.VGG16(weights='imagenet',
                         include_top=False,
                         input_shape=(32,32,3))
for layer in conv_model.layers:
    layer.trainable = False
x = keras.layers.Flatten()(conv_model.output)
x = keras.layers.Dense(128, activation='relu')(x)
x = keras.layers.Dense(64, activation='relu')(x)
x = keras.layers.Dense(64, activation='relu')(x)
x = keras.layers.Dense(32, activation='relu')(x)
x = keras.layers.Dropout(0.5)(x)
predictions = keras.layers.Dense(8, activation='softmax')(x)
full_model = keras.models.Model(inputs=conv_model.input, outputs=predictions)
full_model.summary()
#compile model
full_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
#training
history = full_model.fit(train_dataset.imgs, train_labels,
                    epochs=100,
                    # callbacks=my_callbacks,
                    validation_data=(val_dataset.imgs, val_labels),
                    shuffle=True,
                    class_weight=weights)

#saving neural network structure
model_structure = full_model.to_json()
f = Path("vgg16.json")
f.write_text(model_structure)

#save neural network's saved weights
full_model.save_weights("vgg16.h5")
