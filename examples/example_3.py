from network import *


def mnist_data():
    from torchvision import datasets

    train_set = datasets.MNIST(r'E:\\Projects\\Pycharm\\AI\\backpropogation\\data', train=True, download=True)
    test_set = datasets.MNIST(r'E:\\Projects\\Pycharm\\AI\\backpropogation\\data', train=False, download=True)

    trainX = train_set.data.numpy().reshape((60000, 784))
    trainY1 = train_set.train_labels.numpy()
    testX = test_set.data.numpy().reshape((10000, 784))
    testY1 = test_set.test_labels.data.numpy()

    vals = {0: [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            1: [0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
            2: [0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
            3: [0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
            4: [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
            5: [0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
            6: [0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
            7: [0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
            8: [0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
            9: [0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
            }
    testY = np.array([vals[val] for val in testY1])
    trainY = np.array([vals[val] for val in trainY1])
    return testX, testY, trainX, trainY


def check_accuracy(epoch, error):
    true_cnt = 0
    first_image = test_imgs
    A3 = net.predict(first_image)
    lfi = len(first_image)
    for i in range(lfi):
        pred = (np.round(A3[i], 3)).tolist()
        original = test_labs[i].tolist().index(1)
        prediction = pred.index(max(pred))
        if prediction == original: true_cnt += 1

    print(f"Epoch {epoch}: Predicted {true_cnt} correctly out of {lfi} ({error} error on validation)")


test_imgs, test_labs, trainX, trainY = mnist_data()
net = Network(OptimAdam())
IL = InputLayer(net, 784, title="IL")
HL1 = HiddenLayer(net, 32, title="HL1")
HL2 = HiddenLayer(net, 32, title="HL2")
HL3 = HiddenLayer(net, 32, title="HL3")
HL4 = HiddenLayer(net, 32, title="HL4")
HL5 = HiddenLayer(net, 32, title="HL5")
OL = OutputLayer(net, 10, title="OL")

net.linear_connect(IL, HL1, HL3, OL)
net.linear_connect(IL, HL2, HL4, HL5, OL)
net.linear_connect(HL1, HL5)
net.connect(HL5, HL1, delay_iterations=2)
net.fit(trainX, trainY, 100, 1, epoch_complete_call=check_accuracy)
