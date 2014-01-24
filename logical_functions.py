import unittest
from numpy import *
import loc_reg

TEST_DATA = array([[0,0],[0,1],[1,0],[1,1]])

class LogicalFunctionsTest(unittest.TestCase):

	def testAND(self):
		self.assertItemsEqual(array([0,0,0,1]), and_func(TEST_DATA))

	def testOR(self):
		self.assertItemsEqual(array([0,1,1,1]), or_func(TEST_DATA))

	def testXOR(self):
		self.assertItemsEqual(array([1,0,0,1]), xor_func(TEST_DATA))

	def testNEGATION(self):
		self.assertEqual(1, negation_func(0))
		self.assertEqual(0, negation_func(1))

def and_func(data):
	n, m = data.shape
	data = hstack((ones((n, 1)), data))
	theta = array([-30, 20, 20])
	return loc_reg.predict(data, theta)

def or_func(data):
	n, m = data.shape
	data = hstack((ones((n, 1)), data))
	theta = array([-10, 20, 20])
	return loc_reg.predict(data, theta)

def negation_func(bit):
	data = [1, bit]
	theta = array([10, -20])
	return loc_reg.predict(data, theta)

def xor_func(data):
	n, m = data.shape
	hidden_layer21 = and_func(data) 
	negation = array([negation_func(data[i][j]) for i in range(len(data)) for j in data[i]]).reshape(4,2)
	hidden_layer22 = and_func(negation)
	#output layer
	return or_func(array(zip(hidden_layer21, hidden_layer22)))

if __name__ == '__main__':
	unittest.main()
