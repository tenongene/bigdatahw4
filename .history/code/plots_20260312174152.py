import matplotlib.pyplot as plt
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

	pass


def plot_confusion_matrix(results, class_names):
	# TODO: Make a confusion matrix plot.
	# TODO: You do not have to return the plots.
	# TODO: You can save plots as files by codes here or an interactive way according to your preference.
	pass
