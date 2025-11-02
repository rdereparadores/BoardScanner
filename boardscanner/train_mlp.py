import keras
import numpy as np

data = np.loadtxt('../emnist/emnist-byclass-train.csv', delimiter=',', dtype=np.uint8)
y_train = data[:, 0]
x_train = data[:, 1:] / 255.0

def fix_orientation(img_flat):
    img = img_flat.reshape(28, 28)
    img = np.flip(img, axis=0)
    img = np.rot90(img, k=-1)
    return img

x_train = np.array([fix_orientation(x) for x in x_train])
x_train = x_train[..., np.newaxis]

model = keras.Sequential([
    keras.layers.Input((28, 28, 1)),
    keras.layers.Flatten(),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(62, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(x_train, y_train, epochs=20, batch_size=128, validation_split=0.1)
model.save('emnist_sample.keras')