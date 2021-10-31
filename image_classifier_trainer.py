import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
import random
import pickle
from tensorflow import keras
from tensorflow.keras import layers
from time import time

from numpy.random import seed
seed(42)
import tensorflow
tensorflow.random.set_seed(42)


def resize_image(image, new_size):
    """Resize image."""
    return cv2.resize(image, new_size, cv2.INTER_CUBIC)

def normalize(arr, factor):
    """Normalize an array by a given factor."""
    return arr.astype('float32')/float(factor)

def simple_classifier_cnn(input_shape, num_classes):
    """Get model architecture.
    Params:
        input_shape (list): shape of input image
        num_classes (int): number of classes to predict

    Returns:
        2-layer CNN
    """
    return keras.Sequential(
        [
            keras.Input(shape=input_shape),
            layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Flatten(),
            layers.Dropout(0.5),
            layers.Dense(num_classes, activation="softmax"),
        ]
    )

class TimingCallback(keras.callbacks.Callback):
    """Callback class to get training times per epoch."""
    def __init__(self):
        self.logs=[]
    def on_epoch_begin(self, epoch, logs={}):
        self.starttime=time()
    def on_epoch_end(self, epoch, logs={}):
        self.logs.append(time()-self.starttime)

## Load the dataset
root_dir = os.getcwd()
dataset_name = "squares"
dataset_dir = os.path.join(root_dir, dataset_name)
splits = [folder for folder in os.listdir(dataset_dir) if os.path.isdir(os.path.join(dataset_dir, folder))]

dataset_dict = {}
for split in splits:
    split_dir = os.path.join(dataset_dir, split)
    classes = [folder for folder in os.listdir(split_dir) if os.path.isdir(os.path.join(split_dir, folder))]
    dataset_dict[split] = {}
    for class_name in classes:
        class_dir = os.path.join(split_dir, class_name)
        image_names = os.listdir(class_dir)
        images = []
        image_sizes = []
        image_paths = []
        for image_name in image_names:
            image_path = os.path.join(class_dir, image_name)
            image = cv2.imread(image_path)
            if image is not None:
                images.append(image)
                image_sizes.append(image.shape[0])
                image_paths.append(image_path)
        dataset_dict[split][class_name] = {"images": images, "image_paths": image_paths}
        print("split {}, class {}: {} images".format(split, class_name, len(images)))
        print("\tsizes: mean = {}, median = {}, std = {}, min = {}, max = {}".format(np.mean(image_sizes), np.median(image_sizes), np.std(image_sizes), np.min(image_sizes), np.max(image_sizes)))
        
## Prepare the dataset for training
image_size = [128, 128]
image_size_tag = "{}x{}".format(image_size[0], image_size[1])  # use a tag for saved files

class_idxs = {
    "a": 0,
    "b": 1,
    "c": 2,
}
new_image_sizes = []
for split in dataset_dict:
    resized_images = []
    labels = []
    image_paths = []
    for class_name in class_idxs.keys():
        curr_class_images = dataset_dict[split][class_name]["images"]
        curr_class_image_paths = dataset_dict[split][class_name]["image_paths"]
        for image, image_path in zip(curr_class_images, curr_class_image_paths):
            labels.append(class_idxs[class_name])
            resized_image = resize_image(image, image_size)
            resized_images.append(normalize(resized_image, 255))
            image_paths.append(image_path)
            assert resized_image.shape[0] == image_size[0]
            assert resized_image.shape[1] == image_size[1]
    resized_images_array = np.stack(resized_images)
    dataset_dict[split]["data"] = resized_images_array
    labels_onehot = keras.utils.to_categorical(np.asarray(labels))
    dataset_dict[split]["labels"] = labels_onehot
    dataset_dict[split]["image_paths"] = image_paths
    print("{}: data shape = {}, labels shape = {}".format(split, resized_images_array.shape, labels_onehot.shape))

# save the dataset
dataset_filepath = os.path.join(root_dir, "{}_{}.pkl".format(dataset_name, image_size_tag))
with open(dataset_filepath, "wb") as f:
    pickle.dump({"class_idxs": class_idxs, "dataset_dict": dataset_dict}, f)
print("Saved dataset to {}.".format(dataset_filepath))

## Define the model
input_shape = [image_size[0], image_size[1], 3]
num_classes = len(list(class_idxs.keys()))
early_stopping = keras.callbacks.EarlyStopping(monitor="val_loss", patience=15)
model_checkpoint = keras.callbacks.ModelCheckpoint('best_model_{}.h5'.format(image_size_tag), monitor="val_loss", mode="min", save_best_only=True, verbose=1)
timing_callback = TimingCallback()
model = simple_classifier_cnn(input_shape, num_classes)
model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])

## Define the training parameters
train_images = dataset_dict["train"]["data"]
train_labels = dataset_dict["train"]["labels"]
batch_size = 10
epochs = 100
validation_images = dataset_dict["val"]["data"]
validation_labels = dataset_dict["val"]["labels"]
history = model.fit(train_images, train_labels,
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=1,
                    shuffle=True,
                    callbacks=[early_stopping, model_checkpoint, timing_callback],
                    validation_data=(validation_images, validation_labels))

print("Model took {:.2f}m to train for {} epochs. Average epoch time = {}s.".format(sum(timing_callback.logs)/60, len(history.history["loss"]), np.mean(timing_callback.logs)))

fig, axs = plt.subplots(2)
# Plot accuracy over epochs
axs[0].plot(history.history["accuracy"], label="train")
axs[0].plot(history.history["val_accuracy"], label="val")
axs[0].set_title("model_accuracy")
axs[0].set(xlabel="epoch", ylabel="accuracy")
axs[0].legend(loc='upper left')
# Plot loss over epochs
axs[1].plot(history.history["loss"], label="train")
axs[1].plot(history.history["val_loss"], label="val")
axs[1].set_title("model_loss")
axs[1].set(xlabel="epoch", ylabel="loss")
axs[1].legend(loc='upper left')

fig.tight_layout()
fig_filepath = os.path.join(root_dir, "train_{}_epochs_{}.png".format(len(history.history["loss"]), image_size_tag))
fig.savefig(fig_filepath)
print("Saved training plots to {}".format(fig_filepath))

