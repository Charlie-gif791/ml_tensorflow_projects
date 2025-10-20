import numpy as np, tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.losses import BinaryCrossentropy

model = Sequential([
    Dense(4, activation='relu'),
    Dense(1, activation='sigmoid')
])

x_train = (np.linspace(0, 10, 20)).reshape(-1, 1)
y_train = np.ones(20)

model.compile(loss=BinaryCrossentropy())
model.fit(x_train, y_train, epochs=50)
y_pred = np.round(model.predict(x_train), 0)
print(f"Training error: {np.mean(y_pred != y_train)}")


# End