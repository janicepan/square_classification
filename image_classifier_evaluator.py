import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
import random
import pickle
from tensorflow import keras
from tensorflow.keras import models
from sklearn import metrics


## Define some visualization parameters
font = cv2.FONT_HERSHEY_SIMPLEX
ground_truth_location = (10, 30)
prediction_location = (10, 60)
font_scale = 1
line_type = 2

## Load the dataset dictionary
root_dir = os.getcwd()
dataset_name = "squares"
image_size = [256, 256]
image_size_tag = "{}x{}".format(image_size[0], image_size[1])

visualization_dir = os.path.join(root_dir, "visualize_{}_{}".format(dataset_name, image_size_tag))

if not os.path.isdir(visualization_dir):
	os.mkdir(visualization_dir)

dataset_filepath = os.path.join(root_dir, "{}_{}.pkl".format(dataset_name, image_size_tag))
with open(dataset_filepath, "rb") as f:
    dataset_loaded_dict = pickle.load(f)

print("Loaded dataset from {}.".format(dataset_filepath))

## Visualize the training set
dataset_dict = dataset_loaded_dict["dataset_dict"]
class_idxs = dataset_loaded_dict["class_idxs"]
splits = dataset_dict.keys()
classes = class_idxs.keys()

plot_rows = 20
plot_cols = 25

for split in splits:
	for class_name in classes:
		curr_images = dataset_dict[split][class_name]["images"]
		fig, axs = plt.subplots(nrows=plot_rows, ncols=plot_cols)
		for image_idx, image in enumerate(curr_images):
			image_row_idx = int(np.floor(image_idx / plot_cols))
			image_col_idx = np.mod(image_idx, plot_cols)
			# print("{}: [{}, {}]".format(image_idx, image_row_idx, image_col_idx))
			axs[image_row_idx, image_col_idx].imshow(image)
			axs[image_row_idx, image_col_idx].axis("off")
		fig_filepath = os.path.join(visualization_dir, "{}_{}.png".format(split, class_name))
		fig.tight_layout()
		fig.savefig(fig_filepath)
		print("Wrote image for {} split, class {} to {}".format(split, class_name, fig_filepath))

## Load the model and plot all errors
model_filepath = os.path.join(root_dir, "best_model_{}.h5".format(image_size_tag))
cached_model = models.load_model(model_filepath)

validation_images = dataset_dict["val"]["data"]
validation_labels = dataset_dict["val"]["labels"]
validation_image_paths = dataset_dict["val"]["image_paths"]
predictions = cached_model.predict(validation_images)

## Get indices of errors to visualize
error_idxs = [i for i in range(predictions.shape[0]) if np.argmax(predictions[i,:]) != np.argmax(validation_labels[i,:])]

print("Model misclassified {} out of {} ({:.2f}%)".format(len(error_idxs), predictions.shape[0], 100*len(error_idxs)/predictions.shape[0]))

colors = {
	"a": (255, 0, 0),
	"b": (0, 255, 0),
	"c": (0, 0, 255),
}  # text colors

class_idxs_inv = {}  # dictionary mapping indices to class names; class_idxs dictionary maps class names to indices
for key in class_idxs.keys():
	class_idxs_inv.update({class_idxs[key]: key})

error_dir = os.path.join(visualization_dir, "errors_{}".format(image_size_tag))  # directory to save error images
if not os.path.isdir(error_dir):
	os.mkdir(error_dir)

for idx, error_idx in enumerate(error_idxs):
	label = class_idxs_inv[np.argmax(validation_labels[error_idx,:])]
	prediction = class_idxs_inv[np.argmax(predictions[error_idx,:])]
	image_path = validation_image_paths[error_idx]
	image = cv2.imread(image_path)
	cv2.putText(
		image,
		"GT: {}".format(label),
		ground_truth_location,
		font,
		font_scale,
		colors[label],
		line_type)
	cv2.putText(
		image,
		"Prediction: {}".format(prediction),
		prediction_location,
		font,
		font_scale,
		colors[prediction],
		line_type)
	save_filepath = os.path.join(error_dir, "{}_label{}_pred{}.png".format(error_idx, label, prediction))
	cv2.imwrite(save_filepath, image)

print("Wrote {} errors to {}.".format(len(error_idxs), error_dir))

print(cached_model.summary())

## Plot confusion matrix and print numbers
labels = ["a", "b", "c"]
confusion_matrix = metrics.confusion_matrix(validation_labels.argmax(axis=1), predictions.argmax(axis=1))
print(confusion_matrix)
fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(confusion_matrix)
plt.title('Confusion Matrix')
fig.colorbar(cax)
ax.set_xticklabels([''] + labels)
ax.set_yticklabels([''] + labels)
plt.xlabel('Predicted')
plt.ylabel('Ground Truth')
fig_filepath = os.path.join(visualization_dir, "confusion_matrix.png")
fig.savefig(fig_filepath)

