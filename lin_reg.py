from numpy import *

def lin_reg(X, y):
	m, n = X.shape
	learn_rate = 0.000001
	weights = ones(n + 1)
	X = hstack((ones((m, 1)), X))
	print weights
	print X

	predictions = predict(X, weights)
	print predictions
	
	for i in range(20):
		print 'iteration', i
		weights -= learn_rate * (1.0/m) * dot(predictions - y,X)
		print 'weights =', weights
		predictions = predict(X, weights)
		print 'predictions =', predictions

def regularized_lin_reg(X, y):
	m, n = X.shape
	learn_rate = 0.000001
	regularization_param = 5
	weights = ones(n + 1)
	X = hstack((ones((m, 1)), X))
	print weights
	print X

	predictions = predict(X, weights)
	print predictions
	
	for i in range(20):
		print 'iteration', i
		weights -= learn_rate * ((1.0/m) * dot(predictions - y,X) + ((regularization_param/m) * weights)) 
		print 'weights =', weights
		predictions = predict(X, weights)
		print 'predictions =', predictions



def predict(X, weights):
	return dot(X, weights)
		
X = arange(512).reshape(32,16) 
y = [i for i in range(32)]
print "linear regression:"
lin_reg(X, y)
print "regularized linear regression:"
regularized_lin_reg(X,y)
