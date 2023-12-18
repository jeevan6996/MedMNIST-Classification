from pathlib import Path
from keras.layers import Dense, Conv2D
import numpy as np
import medmnist
from medmnist import INFO

from keras.preprocessing.image import ImageDataGenerator
# MODEL - XCEPTION

import tensorflow as tf
from tensorflow.keras.layers import Input,Dense,Conv2D,Add
from tensorflow.keras.layers import SeparableConv2D,ReLU
from tensorflow.keras.layers import BatchNormalization,MaxPool2D
from tensorflow.keras.layers import GlobalAvgPool2D
from tensorflow.keras import Model
from sklearn.utils import class_weight
import numpy as np

tf.random.set_seed(100)

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

if train_dataset.imgs.ndim == 3:
    print("Adding channel to images...")
    train_dataset.imgs = np.expand_dims(train_dataset.imgs, axis=-1)
if val_dataset.imgs.ndim == 3:
    print("Adding channel to images...")
    val_dataset.imgs = np.expand_dims(val_dataset.imgs, axis=-1)
if test_dataset.imgs.ndim == 3:
    print("Adding channel to images...")
    test_dataset.imgs = np.expand_dims(test_dataset.imgs, axis=-1)
#preprocessing
datagen = ImageDataGenerator(rescale=1. / 255,
                             rotation_range=10,
                             horizontal_flip=True)

#class weight imbalance
breast_class_weights = class_weight.compute_class_weight('balanced',
                                                         classes = np.unique(train_dataset.labels[:,0]),
                                                         y = train_dataset.labels[:, 0])

weights = { 0 : breast_class_weights[0], 1 : breast_class_weights[1] }

# Conv-Batch Norm block
def conv_bn(x, filters, kernel_size, strides=1):
    x = Conv2D(filters=filters,
               kernel_size = kernel_size,
               strides=strides,
               padding = 'same',
               use_bias = False)(x)
    x = BatchNormalization()(x)
    return x
# SeparableConv-Batch Norm block

def sep_bn(x, filters, kernel_size, strides=1):
    x = SeparableConv2D(filters=filters,
                        kernel_size = kernel_size,
                        strides=strides,
                        padding = 'same',
                        use_bias = False)(x)
    x = BatchNormalization()(x)
    return x


# Entry Flow

def entry_flow(x):
    x = conv_bn(x, filters=32, kernel_size=3, strides=2)
    x = ReLU()(x)
    x = conv_bn(x, filters=64, kernel_size=3, strides=1)
    tensor = ReLU()(x)

    x = sep_bn(tensor, filters=128, kernel_size=3)
    x = ReLU()(x)
    x = sep_bn(x, filters=128, kernel_size=3)
    x = MaxPool2D(pool_size=3, strides=2, padding='same')(x)

    tensor = conv_bn(tensor, filters=128, kernel_size=1, strides=2)
    x = Add()([tensor, x])

    x = ReLU()(x)
    x = sep_bn(x, filters=256, kernel_size=3)
    x = ReLU()(x)
    x = sep_bn(x, filters=256, kernel_size=3)
    x = MaxPool2D(pool_size=3, strides=2, padding='same')(x)

    tensor = conv_bn(tensor, filters=256, kernel_size=1, strides=2)
    x = Add()([tensor, x])

    x = ReLU()(x)
    x = sep_bn(x, filters=728, kernel_size=3)
    x = ReLU()(x)
    x = sep_bn(x, filters=728, kernel_size=3)
    x = MaxPool2D(pool_size=3, strides=2, padding='same')(x)

    tensor = conv_bn(tensor, filters=728, kernel_size=1, strides=2)
    x = Add()([tensor, x])
    return x
# Middle Flow

def middle_flow(tensor):
    for _ in range(8):
        x = ReLU()(tensor)
        x = sep_bn(x, filters = 728, kernel_size = 3)
        x = ReLU()(x)
        x = sep_bn(x, filters = 728, kernel_size = 3)
        x = ReLU()(x)
        x = sep_bn(x, filters = 728, kernel_size = 3)
        x = ReLU()(x)
        tensor = Add()([tensor,x])
    return tensor


# Exit Flow

def exit_flow(tensor):
    x = ReLU()(tensor)
    x = sep_bn(x, filters=728, kernel_size=3)
    x = ReLU()(x)
    x = sep_bn(x, filters=1024, kernel_size=3)
    x = MaxPool2D(pool_size=3, strides=2, padding='same')(x)

    tensor = conv_bn(tensor, filters=1024, kernel_size=1, strides=2)
    x = Add()([tensor, x])

    x = sep_bn(x, filters=1536, kernel_size=3)
    x = ReLU()(x)
    x = sep_bn(x, filters=2048, kernel_size=3)
    x = GlobalAvgPool2D()(x)

    x = Dense(units=1, activation='sigmoid')(x)

    return x
# Model Code

input = Input(shape = (28,28,1))
x = entry_flow(input)
x = middle_flow(x)
output = exit_flow(x)

model = Model (inputs=input, outputs=output)
model.summary()

my_callbacks = [
    tf.keras.callbacks.EarlyStopping(patience=5,monitor='accuracy'),
    # tf.keras.callbacks.ModelCheckpoint(filepath='model.{epoch:02d}-{val_loss:.2f}.h5'),
]

model.compile(
    loss='binary_crossentropy',
    optimizer='adam',
    metrics=['accuracy', ]
)

history = model.fit(datagen.flow(train_dataset.imgs, train_dataset.labels),
                    epochs=50,
                    callbacks=my_callbacks,
                    validation_data=(val_dataset.imgs, val_dataset.labels),
                    shuffle=True,
                    class_weight=weights)

#saving neural network structure
model_structure = model.to_json()
f = Path("xception.json")
f.write_text(model_structure)

#save neural network's saved weights
model.save_weights("xception.h5")

