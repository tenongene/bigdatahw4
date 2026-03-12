import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix
# TODO: You can use other packages if you want, e.g., Numpy, Scikit-learn, etc.




def plot_learning_curves(train_losses, valid_losses, train_accuracies, valid_accuracies):
	# TODO: Make plots for loss curves and accuracy curves.
	# TODO: You do not have to return the plots.
	# TODO: You can save plots as files by codes here or an interactive way according to your preference.
	##==========================================================##
	
	epochs = range(1, len(train_losses) + 1)

	fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

	# --- Loss curves ---
	ax1.plot(epochs, train_losses, label='Training Loss')
	ax1.plot(epochs, valid_losses, label='Validation Loss')
	ax1.set_title('Loss Curve')
	ax1.set_xlabel('epoch')
	ax1.set_ylabel('Loss')
	ax1.legend()

	# --- Accuracy curves ---
	ax2.plot(epochs, train_accuracies, label='Training Accuracy')
	ax2.plot(epochs, valid_accuracies, label='Validation Accuracy')
	ax2.set_title('Accuracy Curve')
	ax2.set_xlabel('epoch')
	ax2.set_ylabel('Accuracy')
	ax2.legend()

	plt.tight_layout()
	plt.savefig('learning_curves.png', dpi=150)
	plt.show()




def plot_confusion_matrix(results, class_names):
	# TODO: Make a confusion matrix plot.
	# TODO: You do not have to return the plots.
	# TODO: You can save plots as files by codes here or an interactive way according to your preference.

	## ----------------------------------------------##

	true_labels = [r[0] for r in results]
	pred_labels = [r[1] for r in results]

	# Compute and normalize confusion matrix (row = true class)
	cm = confusion_matrix(true_labels, pred_labels)
	cm_normalized = cm.astype(np.float32) / cm.sum(axis=1, keepdims=True)

	fig, ax = plt.subplots(figsize=(7, 6))
	im = ax.imshow(cm_normalized, interpolation='nearest', cmap=plt.cm.Blues)
	plt.colorbar(im, ax=ax)

	# Axis labels
	tick_marks = np.arange(len(class_names))
	ax.set_xticks(tick_marks)
	ax.set_xticklabels(class_names, rotation=45, ha='right')
	ax.set_yticks(tick_marks)
	ax.set_yticklabels(class_names)

	# Annotate each cell with its value
	thresh = cm_normalized.max() / 2.0
	for i in range(cm_normalized.shape[0]):
		for j in range(cm_normalized.shape[1]):
			ax.text(j, i, f'{cm_normalized[i, j]:.2f}',
					ha='center', va='center',
					color='white' if cm_normalized[i, j] > thresh else 'black')

	ax.set_title('Normalized Confusion Matrix')
	ax.set_ylabel('True')
	ax.set_xlabel('Predicted')

	plt.tight_layout()
	plt.savefig('confusion_matrix.png', dpi=150)
	plt.show()

