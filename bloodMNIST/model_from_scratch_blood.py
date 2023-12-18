from pathlib import Path

from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
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
X_train = train_dataset.imgs/255
X_test = test_dataset.imgs/255
train_labels = to_categorical(train_dataset.labels,num_classes=8)
# train_labels = keras.utils.to_categorical(train_dataset.labels,num_classes=8)
datagen = ImageDataGenerator(rotation_range = 30,brightness_range=[0.4,1.5],horizontal_flip = True)
#model
model = Sequential()

model.add(Conv2D(32, (3, 3), padding='same', input_shape=(28, 28, 3), activation="relu"))
model.add(Conv2D(32, (3, 3), activation="relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(64, (3, 3), padding='same', activation="relu"))
model.add(Conv2D(64, (3, 3), activation="relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(512, activation="relu"))
model.add(Dropout(0.5))
model.add(Dense(8, activation="softmax"))

model.summary()

model.compile(
    loss='categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)

model.fit(datagen.flow(train_dataset.imgs, train_labels, batch_size=28),
                    epochs=20, # one forward/backward pass of training data,
                    steps_per_epoch=train_dataset.imgs.shape[0]//28)
#saving neural network structure
model_structure = model.to_json()
f = Path("model_from_scratch_blood.json")
f.write_text(model_structure)

#save neural network's saved weights
model.save_weights("model_from_scratch_blood.h5")
