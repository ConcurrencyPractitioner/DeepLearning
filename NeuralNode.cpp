/*
 * NeuralNode.cpp
 *
 *  Created on: May 27, 2018
 *      Author: richardyu
 */

//Implementation of Kacmarcz's algorithm on single node
#include "NeuralNode.h"
#include "MatrixOperations.h"
#include <vector>
#include <iostream>

using namespace std;

MatrixOperations operations;

NeuralNode::NeuralNode(vector<double> modifiers, double rate)
: weights(modifiers), learningRate(rate) {
	if (learningRate < 0 || learningRate > 2) {
		learningRate = 0.5;
	}
}

NeuralNode::NeuralNode() : learningRate(0.0) {}
NeuralNode::~NeuralNode() {}

//gradient of the following cost function:: 1/2 * (||Ax-b||^2 )
// is transpose(A) * A * x - A * b
//Note that ||x|| means Euclidean norm (x is a vector)
//A -> input matrix
//x -> weights
//b -> outputs
vector<double> NeuralNode::calculateGradient(vector<double> inputs,
		vector<vector<double> > matrix,
		vector<double> outputs) {
	vector<vector<double> > firstPart = operations.multiply(matrix, operations.transpose(operations.boxVector(inputs)));
	vector<double> sum =
			operations.unboxVector(operations.transpose(operations.add(firstPart, operations.transpose(operations.boxVector(operations.reduce(outputs, -1))))));
	return sum;
}

double NeuralNode::computeResult(vector<double> inputs) {
	double sum = 0;
	if (inputs.size() != weights.size()) return -1;
	for (int i = 0; i < weights.size(); i++) {
		sum += inputs.at(i) * weights.at(i);
	}
	return sum;
}

//We want to minimize the error over all pairs of data sets (using L2-algorithm)
void NeuralNode::train(vector<vector<double> > inputs, vector<double> outputs, double epochs) {
	for (int i = 0; i < epochs; i++) {
		for (int j = 0; j < inputs.size(); j++) {
			double result = this -> computeResult(inputs.at(j));
			double error = outputs.at(j) - result;
			double norm = operations.dotProduct(inputs.at(j), inputs.at(j));
			norm = 1 / norm;
			double multiplicand = norm * this -> learningRate * error;
			vector<double> addendum = operations.reduce(inputs.at(j), multiplicand);
			weights = operations.add(weights, addendum);
		}
	}
}

double NeuralNode::determineRate(vector<double> gradient,
		vector<double> conjugateDirection,
		vector<vector<double> > inputs) {
	double numerator = operations.dotProduct(gradient, conjugateDirection);
	vector<vector<double> > transposedDirection = operations.boxVector(conjugateDirection);
	vector<double> firstPart = operations.unboxVector(operations.multiply(transposedDirection, inputs));
	double divisor = operations.dotProduct(firstPart, conjugateDirection);
	double alpha_k = numerator / divisor;
	return -1 * alpha_k;
}

vector<vector<double> > NeuralNode::computeHessian(vector<vector<double> > initHessian,
		vector<double> xDelta,
		vector<double> gradientDelta) {
	vector<vector<double> > result = initHessian;
	vector<vector<double> > transposedXDelta = operations.transpose(operations.boxVector(xDelta));
	vector<vector<double> > boxedXDelta = operations.boxVector(xDelta);
	vector<vector<double> > transposedGradientDelta = operations.transpose(operations.boxVector(gradientDelta));
	vector<vector<double> > boxedGradientDelta = operations.boxVector(gradientDelta);
	vector<vector<double> > product1 = operations.multiply(transposedXDelta, boxedXDelta);
	double factor = operations.dotProduct(xDelta, gradientDelta);
	product1 = operations.matrixReduce(product1, 1 / factor);
	vector<vector<double> > product2 = operations.multiply(initHessian, transposedGradientDelta);
	vector<vector<double> > factor2_1 = operations.multiply(boxedGradientDelta, initHessian);
	product2 = operations.multiply(product2, operations.transpose(product2));
	double factor2 = operations.multiply(factor2_1, operations.transpose(boxedGradientDelta)).at(0).at(0);
	product2 = operations.matrixReduce(product2, -1 * (1 / factor2));
	if (factor != 0) result = operations.matrixAdd(result, product1);
	if (factor2 != 0) result = operations.matrixAdd(result, product2);
	return result;
}

/**
 * A quasi-newton method which approximates the inverse hessian rather
 * than calculate it directly. Avoids a lot of expensive computations. It is an implementation
 * of the DFP algorithm which was first introduced in 1959 by Davidon and subsequently modified by
 * Fletcher and Powell in 1963.
 */
void NeuralNode::newtonTrain(vector<vector<double> > inputs, vector<double> output, int epochs) {
	vector<double> initPoint = weights;
	vector<vector<double> > initHessian = operations.getIdentityMatrix(inputs.at(0).size());
	vector<vector<double> > matrixQ = operations.multiply(operations.transpose(inputs), inputs);
	vector<double> constant = operations.unboxVector(operations.transpose(operations.multiply(operations.transpose(inputs), operations.transpose(operations.boxVector(output)))));
	vector<double> gradient = calculateGradient(initPoint, matrixQ, constant);
	for (int i = 0; i < epochs; i++) {
		vector<double> conjugateDirection = operations.unboxVector(operations.transpose(operations.multiply(initHessian, operations.transpose(operations.boxVector(gradient)))));
		conjugateDirection = operations.reduce(conjugateDirection, -1);
		double rate = determineRate(gradient, conjugateDirection, matrixQ);
		initPoint = operations.add(initPoint, operations.reduce(conjugateDirection, rate));
		vector<double> xDelta = operations.reduce(conjugateDirection, rate);
		vector<double> newGradient = calculateGradient(initPoint, matrixQ, constant);
		vector<double> gradientDelta = operations.add(newGradient, operations.reduce(gradient, -1));
		gradient = newGradient;
		initHessian = computeHessian(initHessian, xDelta, gradientDelta);
	}
	weights = initPoint;
}

void print(NeuralNode& node) {
	cout << "Current weights of the model is: " << endl;
	for (int i = 0; i < node.weights.size(); i++) {
		cout << node.weights.at(i) << " ";
	}
	cout << endl;
}
