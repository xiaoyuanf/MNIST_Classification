import numpy as np
import findMin
from sklearn.utils import shuffle


class SVM:
    # SVM with SGD
    def __init__(self, epoch=100, minibatch=500, lammy=0.001, alpha=0.001):
        self.epoch = epoch
        self.minibatch = minibatch
        self.lammy = lammy
        self.alpha=alpha
        
    def funObj(self, w, X, y):
        n, d = X.shape
        W = np.reshape(w, (d, self.n_classes))
        
        Xw = X@W
        Xwy = np.zeros((n,1))
        g = np.zeros(W.shape)
        
        for i in range(n):
            Xwy[i] = Xw[i, y[i]]
        
        # a n*k matrix of max{0, 1-wyiTxi+wcTxi} of each example
        hinge_loss=np.maximum(0, 1-Xwy+Xw)

        # ignore the right classes where hinge loss is 1
        hinge_loss[np.arange(n),y] = 0
        
        f = np.sum(hinge_loss)+0.5*self.lammy*np.sum(W.T.dot(W))
        
        # gradient wrt each wc!=yi is xi where hinge loss>0
        # gradient wrt wc==yi is -xi where hinge loss>0 
        # scaled by number of cases where hinge loss>0
        hinge_loss[hinge_loss>0]=1
        make_loss = np.sum(hinge_loss, axis=1)
        hinge_loss[np.arange(n),y] -= make_loss # gradiant wrt wc==yi
        g = (X.T).dot(hinge_loss)
        
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
                self.w, f = findMin.SGD(self.funObj, self.w, self.alpha,
                                        Xtrain[i:i+minibatch,:], 
                                        ytrain[i:i+minibatch])

                # check validation error
                self.w = np.reshape(self.w, (d, self.n_classes))
                yhat = np.argmax(Xvalid@self.w, axis=1)
                error_new = np.mean(yvalid!=yhat)
                self.w = np.reshape(self.w, d*self.n_classes)
                    
            if error_new < error:
                error = error_new
                #print(e)
            # stop when validation error doesn't improve
            else:
                break

        
    def predict(self, X):
        self.w = np.reshape(self.w, (X.shape[1], self.n_classes))
        return np.argmax(X@self.w, axis=1)



        
