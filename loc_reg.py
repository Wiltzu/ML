from numpy import *

def loc_reg(X, y):
    """
	Positive/Negative class classification
	"""
    m, n = X.shape
    learn_rate = 0.2
    weights = ones(n + 1)
    X = hstack((ones((m, 1)), X))
    y = array(y)
    print "initial weigts:", weights
    print "training set:", X
    print "training set labels:",y

    predictions = predict(X, weights)
    print "first predictions", predictions

    for i in range(20):
        print 'iteration', i
        weights -= learn_rate * (1.0/m) * dot(predictions - y,X)
        print 'weights =', weights
        predictions = predict(X, weights)
        print 'predictions =', predictions

def predict(X, weights):
    return where((1.0/(1.0 + exp(-dot(X, weights)))) > 0.5, 1, 0);
		

if __name__ == '__main__':
    loc_reg(arange(8).reshape(4,2), [0.0,0.0,1.0,1.0])
