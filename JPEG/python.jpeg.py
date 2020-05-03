# | Created by Ar4ikov
# | Время: 02.05.2020 - 22:12

# | Created by Ar4ikov
# | Время: 21.12.2019 - 23:41

from keras.layers import Dense, Input
from keras.models import Model
from keras.optimizers import Adam
from keras.metrics import accuracy
from typing import List
from numpy import random
import numpy as np
from PIL import Image

# Filter function
x: List[float]
f = lambda x: [sum(x) / 3 for _ in range(3)]

X = random.randint(0, 255, size=[10000, 3]) / 255
Y = np.array(list(map(f, X)))

x = Dense(20, activation="relu")
x = Dense(10, activation="relu")(x)
output_layer = Dense(3, activation="relu")(x)

model = Model(net, output_layer, name="AMD")
model.compile(optimizer=Adam(), loss="mean_squared_error", metrics=[accuracy])

h = model.fit(
    X, Y,
    batch_size=1024,
    epochs=100,
    shuffle=True,
    verbose=0
)


# Testing Solo pixel prediction

solution = model.predict([[233 / 255, 245 / 255, 59 / 255]]) * 255
print(f"Median pixel (RGB) from [233, 245, 59] is 179. Prediction is {solution}")

# Testing Image pixel prediction

img = np.array(Image.open("LizaSU.png")).reshape([551 * 980, 3]) / 255
prediction = model.evaluate(img).reshape([551, 980, 3]) * 255
print("Saving Prediction...")
Image.fromarray(np.array(prediction, dtype="uint8")).save("Liza.jpg")