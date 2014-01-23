from numpy import *

def lin_reg(X, y):
	m, n = X.shape
	learn_rate = 0.001
	weights = ones(n + 1)
	X = hstack((ones((m, 1)), X))
	print weights
	print X

	predictions = dot(X, weights)
	print predictions
	
	for i in range(20):
		print 'iteration', i
		weights -= learn_rate * (1.0/m) * dot(predictions - y,X)
		print 'weights =', weights
		predictions = dot(X, weights)
		print 'predictions =', predictions


		

lin_reg(arange(8).reshape(4,2), [0,2,4,8])
