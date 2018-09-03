from data_utils import downloadMNIST,loadMNIST
from residual_block_net import *
import numpy as np

def main():
        # download datasets
    downloadMNIST('train-images-idx3-ubyte.gz')
    downloadMNIST('train-labels-idx1-ubyte.gz')
    downloadMNIST('t10k-images-idx3-ubyte.gz')
    downloadMNIST('t10k-labels-idx1-ubyte.gz')

    # process datasets
    X_train, y_train = loadMNIST(dataset="training", num_image=60000)
    X_test, y_test = loadMNIST(dataset="test", num_image=10000)
    num_train = 50000
    num_val = 10000
    num_test = 10000
    mask,mask1,mask2 = range(num_train, num_train + num_val),range(num_train),range(num_test)
    X_val,y_val = X_train[mask],y_train[mask]
    X_train,y_train = X_train[mask1],y_train[mask1]
    X_test,y_test = X_test[mask2],y_test[mask2]
    # normalize
    mean_image = np.mean(X_train, axis=0)
    X_train -= mean_image
    X_val -= mean_image
    X_test -= mean_image
    # reshape (m,n_row*n_col)
    X_train = np.reshape(X_train, (X_train.shape[0], -1))
    X_val = np.reshape(X_val, (X_val.shape[0], -1))
    X_test = np.reshape(X_test, (X_test.shape[0], -1))

    # train a residual block neural network and predict
    input_size = 28 * 28
    hidden_size = 500
    hidden2_size = 200
    num_classes = 10
    net = ResidualBlockNet(input_size, hidden_size, hidden2_size, num_classes, std = 0.005)

    stats = net.train(X_train, y_train, X_val, y_val,
                    num_iters=1000, batch_size=200,
                    learning_rate=1e-3, learning_rate_decay=0.999,
                    reg=0.5, verbose=True)

    # train neural network without residual block and predict
    net1 = ResidualBlockNet(input_size, hidden_size, hidden2_size, num_classes, std=0.005, use_Res = False)
    stats1 = net1.train(X_train, y_train, X_val, y_val,
                    num_iters=1000, batch_size=200,
                    learning_rate=1e-3, learning_rate_decay=0.999,
                    reg=0.5, verbose=True)
    plot(stats,stats1)

    # Predict on the validation set
    val_acc = (net.predict(X_val) == y_val).mean()
    train_acc = (net.predict(X_train) == y_train).mean()
    print('Validation accuracy with residual block: ', val_acc)
    print('Train accuracy with residual block: ', train_acc)

    val_acc1 = (net1.predict(X_val) == y_val).mean()
    train_acc1 = (net1.predict(X_train) == y_train).mean()
    print('Validation accuracy without residual block: ', val_acc1)
    print('Train accuracy without residual block: ', train_acc1)

if __name__ == '__main__':
    main()