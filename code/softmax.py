import numpy as np
import findMin
from sklearn.utils import shuffle
    
    
class softmaxClassifier:
     # Logistic Regression
    def __init__(self, epoch=30, minibatch=1000, alpha=0.001):
        self.epoch = epoch
        self.minibatch = minibatch
        self.alpha=alpha

    def funObj(self, w, X, y):
        n, d = X.shape
        W = np.reshape(w, (d, self.n_classes))

        Xw = X@W
        Xwy = np.zeros((n,))
        g1 = np.zeros(W.shape)
        g2 = np.zeros(W.shape)

        for i in range(n):
            Xwy[i] = Xw[i, y[i]]

        for i in np.unique(y):
            # XI, I is 1 when yi=c and 0 otherwise
            g1[:, i] = -np.sum(X[y == i], axis=0)
            # p(yi=c|W,xi) 
            den = np.sum(np.exp(Xw), axis=1)
            num = np.exp(Xw[:, i])
            g2[:, i] = (num[:]/den[:])@X

        f = -np.sum(Xwy)/n+np.sum(np.log(np.sum(np.exp(Xw), axis=1)))

        g = (g1 + g2)

        return f, g.flatten()

    def fit(self, X, y, epoch, minibatch, alpha):
        n, d = X.shape
        self.n_classes = np.unique(y).size
        self.w = np.zeros(d*self.n_classes)
        
        error = 1
        
        # SGD with early stopping
        for e in range(epoch):
            X, y = shuffle(X, y)
            Xtrain = X[:n//3*2]
            ytrain = y[:n//3*2]
            Xvalid = X[n//3*2:]
            yvalid = y[n//3*2:]
                                                    
            for i in range(0, n, minibatch):
                self.w, f = findMin.SGD(self.funObj, 
                                        self.w, self.alpha,
                                        Xtrain[i:i+minibatch,:], 
                                        ytrain[i:i+minibatch])

                # check validation error
                self.w = np.reshape(self.w, (d, self.n_classes))
                yhat = np.argmax(Xvalid@self.w, axis=1)
                error_new = np.mean(yvalid!=yhat)
                self.w = np.reshape(self.w, d*self.n_classes)
                    
            if error_new < error:
                error = error_new
                #print("Epoch ", e, ": error is ", error)
            # stop when validation error doesn't improve
            else:
                break

    def predict(self, X):
        self.w = np.reshape(self.w, (X.shape[1], self.n_classes))
        return np.argmax(X@self.w, axis=1)
