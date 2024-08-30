"""
Implementation of Simple Feed Forward Network.
Get the Module at: https://github.com/thevaishnav/RecurrentTreeNet
Made by Vaishnav Chincholkar
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from source import *
import pickle


def load_data():
    print("Loading Data", end="")
    # Load the mnist training data provided by Google Collab
    train_set = pd.read_csv("mnist_test.csv", header=None)
    X_train = train_set.iloc[:, 1:].values
    Y_train = train_set.iloc[:, 0].values

    # Load the mnist test data provided by Google Collab
    test_set = pd.read_csv("mnist_test.csv", header=None)
    X_test = test_set.iloc[:, 1:].values
    Y_test = test_set.iloc[:, 0].values

    # Prepare the data
    encoder = OneHotEncoder(categories='auto', sparse_output=False)

    Y_train = Y_train.reshape(-1, 1)  # required for OneHotEncoder
    Y_train = encoder.fit_transform(Y_train)

    Y_test = Y_test.reshape(-1, 1)  # required for OneHotEncoder
    Y_test = encoder.transform(Y_test)
    print("- Done")
    return X_train, Y_train, X_test, Y_test


def check_accuracy(epoch, error):
    """
    Checks the accuracy of the model (number of correct predictions / total number of samples) on the test dataset
    This function will be called after each epoch to ensure that the model is not over-fitting
    """
    predictions = model.predict(X_test)
    predicted_labels = np.argmax(predictions, axis=1)
    true_labels = np.argmax(Y_test, axis=1)
    accuracy = np.mean(predicted_labels == true_labels)
    print(f"Accuracy: {accuracy * 100:.2f}%")


def create_model():
    print("Creating Model", end="")
    # Create the Neural Network and Layers
    model = Network(OptimRMPProp())  # Every edge and layer will use This optimizer by default.
    IL = InputLayer(model, 784, title="IL")
    HL1 = HiddenLayer(model, 32, title="HL1", _act_fun=ActLReLU(), optimizer=OptimAdam())
    HL2 = HiddenLayer(model, 32, title="HL2", _act_fun=ActLReLU())
    OL = OutputLayer(model, 10, title="OL", _act_fun=ActSoftmax())

    # Connect the layers
    model.linear_connect(IL, HL1, HL2, OL)
    """
    model.linear_connect(IL, HL1, HL2, OL) is equivalent to connecting these layers in a Linear Feed Forward Neural Network:
    IL -> HL1 -> HL2 -> OL
    """

    # Necessary step, Validates the connections, and
    # Decides execution order for Forward Pass and Backward Pass
    model.compile()
    print("- Done")
    return model


def train_and_save_model():
    # Train the model and save data to a file
    print("Training Model")
    model.fit(X_train, Y_train, 100, 10, epoch_complete_call=check_accuracy)

    print()
    print("Saving Model", end="")
    with open("example_1_trained.bin", 'wb') as f:
        model_data = model.serialize()
        pickle.dump(model_data, f)
    print("- Done. Saved to: \"example_1_trained.bin\" file")


def load_and_test_mode():
    print("Loading Model", end="")
    with open("example_1_trained.bin", 'rb') as f:
        model_data = pickle.load(f)
        model.deserialize(model_data)
        print("- Done")

    print("Testing Model")
    check_accuracy(1, 0)


X_train, Y_train, X_test, Y_test = load_data()
model = create_model()
# train_and_save_model()
load_and_test_mode()