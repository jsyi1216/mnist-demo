from sklearn.datasets import fetch_mldata
from sklearn.neural_network import MLPClassifier
import pickle
filename = 'prediction_Module/trained_models/nn_trained_digit_classifier.pkl'


def train_nn():
	mnist = fetch_mldata("MNIST original")

	# # rescale the data, use the traditional train/test split
	X, y = mnist.data / 255., mnist.target
	X_train = X[:60000]
	y_train = y[:60000]

	mlp = MLPClassifier(hidden_layer_sizes=(50,), max_iter=10, alpha=1e-4,
	                    solver='sgd', verbose=True, tol=1e-4, random_state=1,
	                    learning_rate_init=.1)

	# Train the model
	mlp.fit(X_train, y_train)

	# Save the model using pickle
	with open(filename, 'wb') as f:
		pickle.dump(mlp, f)

	print("Training set score: %f" % mlp.score(X_train, y_train))