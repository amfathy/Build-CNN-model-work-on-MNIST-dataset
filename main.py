import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten
from tensorflow.keras.utils import to_categorical


(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

train_images = train_images.reshape((60000, 28, 28, 1)).astype("float32") / 255
test_images = test_images.reshape((10000, 28, 28, 1)).astype("float32") / 255
train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    Flatten(),
    Dense(64, activation='relu'),
    Dense(10, activation='softmax')
])

model.compile(optimizer="adam",
              loss="categorical_crossentropy",
              metrics=["accuracy"])

model.summary()

model.fit(train_images, train_labels, epochs=5, batch_size=128, verbose=1)

train_loss, train_acc = model.evaluate(train_images, train_labels, verbose=0)
print("Training Loss:", train_loss)
print("Training Accuracy:", train_acc)

test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=0)
print("Test Loss:", test_loss)
print("Test Accuracy:", test_acc)

conv1_weights = model.layers[0].get_weights()[0]
plt.figure(figsize=(10, 5))
for i in range(32):
    plt.subplot(4, 8, i+1)
    plt.imshow(conv1_weights[:, :, 0, i], cmap='gray')
    plt.axis('off')
plt.suptitle('CNN Filters - Convolutional Layer 1')
plt.show()

layer_outputs = [layer.output for layer in model.layers]
activation_model = Model(inputs=model.input, outputs=layer_outputs)

image_index = 0
activation = activation_model.predict(test_images[image_index].reshape(1, 28, 28, 1))

plt.figure(figsize=(20, 6))
for i, feature_map in enumerate(activation):
    if len(feature_map.shape) == 4:
        # Plot only feature maps (skip intermediate layers)
        square = int(np.ceil(np.sqrt(feature_map.shape[-1])))
        for j in range(feature_map.shape[-1]):
            plt.subplot(len(activation), square, i * square + j + 1)
            plt.imshow(feature_map[0, :, :, j], cmap='viridis')
            plt.axis('off')
plt.suptitle('Feature Maps - Activation of Each Layer')
plt.show()
