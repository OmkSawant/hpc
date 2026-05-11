
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten

(X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()

X_train = X_train / 255.0
X_test = X_test / 255.0

# Step 4: Build Deep Neural Network
model = Sequential()

# Convert 2D image into 1D vector
model.add(Flatten(input_shape=(28,28)))

# Hidden Layers
model.add(Dense(128, activation='relu'))
model.add(Dense(64, activation='relu'))

# Output Layer
model.add(Dense(10, activation='softmax'))

model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])

model.fit(X_train,y_train,epochs=10,batch_size=32,verbose=1)

loss, accuracy = model.evaluate(X_test, y_test)

print("\nAccuracy:", accuracy)

predictions = model.predict(X_test)

fashion_labels = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot"
]

# Step 9: Display Predictions
print("\nActual Category  |  Predicted Category\n")

for i in range(5):

    actual = fashion_labels[y_test[i]]

    predicted = fashion_labels[np.argmax(predictions[i])]

    print(actual, "   |   ", predicted)

for i in range(5):

    plt.imshow(X_test[i], cmap='gray')

    actual = fashion_labels[y_test[i]]

    predicted = fashion_labels[np.argmax(predictions[i])]

    plt.title(f"Actual: {actual} | Predicted: {predicted}")

    plt.show()
