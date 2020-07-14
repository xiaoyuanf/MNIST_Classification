import os
import pickle
import gzip
import argparse
import numpy as np


from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import KFold

from knn import KNN
import softmax
import utils
import svm
import neural_net


def load_dataset(filename):
    with open(os.path.join('..','data',filename), 'rb') as f:
        return pickle.load(f)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-m','--Model', required=True)

    io_args = parser.parse_args()
    Model = io_args.Model
    
    with gzip.open(os.path.join('..', 'data', 'mnist.pkl.gz'), 'rb') as f:
        train_set, valid_set, test_set = pickle.load(f, encoding="latin1")
    X, y = train_set
    Xvalid, yvalid = valid_set
    Xtest, ytest = test_set

    X = X.astype('float32')
    Xtest = Xtest.astype('float32')

    binarizer = LabelBinarizer()
    Y = binarizer.fit_transform(y)
    
    # create k-folds for cross validation
    kf = KFold(n_splits = 5)
        
    if Model == "knn":

        X = X.astype('float32')
        Xtest = Xtest.astype('float32')
        
        model = KNN(k=3)
        model.fit(X, y)
        yhat = model.predict(X)
        tr_error=np.mean(yhat != y)
        print("k is 3", "training error is ", tr_error)
            
        yhat = model.predict(Xvalid)
        v_error=np.mean(yhat != yvalid)
        print("k is 3", "validation error is ", v_error)
        
    elif Model == "softmax":

        # standardize X
        X, mu, sigma = utils.standardize_cols(X)
        Xvalid, _, _ = utils.standardize_cols(Xvalid, mu, sigma)
        Xtest, _, _ = utils.standardize_cols(Xtest, mu, sigma)     
        
        model = softmax.softmaxClassifier()
        minibatch=[500, 1000, 1500]
        alpha=[0.01, 0.001, 0.0005]
        min_val_error = 1
        best_batch = 0
        best_alpha = 0
        
        for m in range(3):
            for a in range(3):
                val_error = []
                # cross validation
                for train, validate in kf.split(X, y):
                    model.fit(X[train], y[train], epoch=100, 
                          minibatch=minibatch[m], 
                          alpha=alpha[a])

                    # record validation error
                    v_error = utils.classification_error(model.predict(X[validate]), y[validate])
                    val_error.append(v_error)
                    
                avg_val_error = np.average(np.asarray(val_error))
                print("batch size: {0}, alpha: {1}, validation error: {2}".format(minibatch[m], 
                      alpha[a], 
                      avg_val_error))
                
                if avg_val_error < min_val_error:
                    min_val_error = avg_val_error
                    best_batch = minibatch[m]
                    best_alpha = alpha[a]
                    
        print("When batch size is {0}, alpha is {1}, test error is {2}".format(best_batch, 
              best_alpha, 
              utils.classification_error(model.predict(Xtest), ytest)))
        
    #elif Model == 'rbf':
        

    elif Model == 'svm':
        
        # standardize X
        X, mu, sigma = utils.standardize_cols(X)
        Xvalid, _, _ = utils.standardize_cols(Xvalid, mu, sigma)
        Xtest, _, _ = utils.standardize_cols(Xtest, mu, sigma)
        
        # SVM
        model = svm.SVM()
        minibatch=[500, 1000, 1500]
        alpha=[0.01, 0.001, 0.0001]
        min_val_error = 1
        best_batch = 0
        best_alpha = 0
        
        for m in range(3):
            for a in range(3):
                val_error = []
                # cross validation
                for train, validate in kf.split(X, y):
                    model.fit(X[train], y[train], epoch=100, 
                          minibatch=minibatch[m], 
                          alpha=alpha[a])
                
                    # record validation error
                    v_error = utils.classification_error(model.predict(X[validate]), y[validate])
                    val_error.append(v_error)
                    
                avg_val_error = np.average(np.asarray(val_error))
                print("batch size: {0}, alpha: {1}, validation error: {2}".format(minibatch[m], 
                      alpha[a], 
                      avg_val_error))
                
                if avg_val_error < min_val_error:
                    min_val_error = avg_val_error
                    best_batch = minibatch[m]
                    best_alpha = alpha[a]
                
        print("When batch size is {0}, alpha is {1}, test error is {2}".format(best_batch, 
              best_alpha, 
              utils.classification_error(model.predict(Xtest), ytest)))
                        
    elif Model == 'MLP':
        
        #X, mu, sigma = utils.standardize_cols(X)
        #Xtest, _, _ = utils.standardize_cols(Xtest, mu, sigma)
        
        # MLP
        hidden_layer_sizes = [50]
        model = neural_net.NeuralNet(hidden_layer_sizes)

        model.fitWithSGD(X, Y, epoch=100, minibatch=1000, alpha=0.001)

        print("When using SGD, training error %.3f" % utils.classification_error(model.predict(X), y))
        print("When using SGD, validation error %.3f" % utils.classification_error(model.predict(Xtest), ytest))
        
        model.fit(X, Y)
        print("When using GD, training error %.3f" % utils.classification_error(model.predict(X), y))
        print("When using GD, validation error %.3f" % utils.classification_error(model.predict(Xtest), ytest))
       
    else:
        print("Unknown Model: %s" % Model)    
