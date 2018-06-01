/*
 * NeuralNode.cpp
 *
 *  Created on: May 27, 2018
 *      Author: richardyu
 */

//Implementation of Kacmarcz's algorithm on single node
#include "NeuralNode.h"
#include <vector>
#include <iostream>

using namespace std;

double dotProduct(vector<double>& vec1, vector<double>& vec2) {
	double sum = 0;
	if (vec1.size() != vec2.size()) return -1;
	for (int i = 0; i < vec1.size(); i++) {
		sum += vec1.at(i) * vec2.at(i);
	}
	return sum;
}

vector<double> reduce(vector<double> vec, double factor) {
	vector<double> res = vec;
	for (int i = 0; i < vec.size(); i++) {
		res.at(i) *= factor;
	}
	return res;
}

vector<vector<double> > matrixReduce(vector<vector<double> > matrix, double factor) {
	vector<vector<double> > result = matrix;
	for (int i = 0; i < matrix.size(); i++) {
		for (int j = 0; j < matrix.at(i).size(); j++) {
			result.at(i).at(j) *= factor;
		}
	}
	return result;
}

vector<double> add(vector<double> vec, vector<double> vec2) {
	vector<double> res = vec;
	for (int i = 0; i < vec2.size(); i++) {
		res.at(i) += vec2.at(i);
	}
	return res;
}

vector<vector<double> > add(vector<vector<double> > mat1, vector<vector<double> > mat2) {
	vector<vector<double> > result = mat1;
	for (int i = 0; i < mat1.size(); i++) {
		for (int j = 0; j < mat1.at(i).size(); j++) {
			result.at(i).at(j) += mat2.at(i).at(j);
		}
	}
	return result;
}

vector<vector<double> > multiply(vector<vector<double> > matrix1, vector<vector<double> > matrix2) {
	if (matrix1.at(0).size() != matrix2.size()) {
		return vector<vector<double> >();
	}
	vector<vector<double> > result =
			vector<vector<double> >(matrix1.size(), vector<double>(matrix2.at(0).size(), 0.0));
	for (int i = 0; i < result.size(); i++) {
		for (int j = 0; j < result.at(i).size(); j++) {
			double sum = 0;
			for (int k = 0; k < matrix1.at(0).size(); k++) {
				sum += matrix1.at(i).at(k) * matrix2.at(k).at(j);
			}
			result.at(i).at(j) = sum;
		}
	}
	return result;
}

vector<vector<double> > transpose(vector<vector<double> > matrix) {
	vector<vector<double> > result =
			vector<vector<double> > (matrix.at(0).size(), vector<double>(matrix.size(), 0.0));
	for (int i = 0; i < matrix.size(); i++) {
		for (int j = 0; j < matrix.at(i).size(); j++) {
			result.at(j).at(i) = matrix.at(i).at(j);
		}
	}
	return result;
}

vector<vector<double> > matrixAdd(vector<vector<double> > mat1, vector<vector<double> > mat2) {
	vector<vector<double> > result = mat1;
	for (int i = 0; i < mat2.size(); i++) {
		for (int j = 0; j < mat2.at(i).size(); j++) {
			result.at(i).at(j) += mat2.at(i).at(j);
		}
	}
	return result;
}

vector<vector<double> > boxVector(vector<double> input) {
	vector<vector<double> > res;
	res.push_back(input);
	return res;
}

vector<double> unboxVector(vector<vector<double> > input) {
	return input.at(0);
}

NeuralNode::NeuralNode(vector<double> modifiers, double rate)
: weights(modifiers), learningRate(rate) {
	if (learningRate < 0 || learningRate > 2) {
		learningRate = 0.5;
	}
}

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
	vector<vector<double> > firstPart = multiply(matrix, transpose(boxVector(inputs)));
	vector<double> sum =
			unboxVector(transpose(add(firstPart, transpose(boxVector(reduce(outputs, -1))))));
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
			double norm = dotProduct(inputs.at(j), inputs.at(j));
			norm = 1 / norm;
			double multiplicand = norm * this -> learningRate * error;
			vector<double> addendum = reduce(inputs.at(j), multiplicand);
			weights = add(weights, addendum);
		}
	}
}

vector<vector<double> > getIdentityMatrix(int size) {
	vector<vector<double> > result;
	for (int i = 0; i < size; i++) {
		vector<double> row;
		for (int j = 0; j < size; j++) {
			if (i == j) row.push_back(1);
			else row.push_back(0);
		}
		result.push_back(row);
	}
	return result;
}

double determineRate(vector<double> gradient,
		vector<double> conjugateDirection,
		vector<vector<double> > inputs) {
	double numerator = dotProduct(gradient, conjugateDirection);
	vector<vector<double> > transposedDirection = boxVector(conjugateDirection);
	vector<double> firstPart = unboxVector(multiply(transposedDirection, inputs));
	double divisor = dotProduct(firstPart, conjugateDirection);
	double alpha_k = numerator / divisor;
	return -1 * alpha_k;
}

vector<vector<double> > computeHessian(vector<vector<double> > initHessian,
		vector<double> xDelta,
		vector<double> gradientDelta) {
	vector<vector<double> > result = initHessian;
	vector<vector<double> > transposedXDelta = transpose(boxVector(xDelta));
	vector<vector<double> > boxedXDelta = boxVector(xDelta);
	vector<vector<double> > transposedGradientDelta = transpose(boxVector(gradientDelta));
	vector<vector<double> > boxedGradientDelta = boxVector(gradientDelta);
	vector<vector<double> > product1 = multiply(transposedXDelta, boxedXDelta);
	double factor = dotProduct(xDelta, gradientDelta);
	product1 = matrixReduce(product1, 1 / factor);
	vector<vector<double> > product2 = multiply(initHessian, transposedGradientDelta);
	vector<vector<double> > factor2_1 = multiply(boxedGradientDelta, initHessian);
	product2 = multiply(product2, transpose(product2));
	double factor2 = multiply(factor2_1, transpose(boxedGradientDelta)).at(0).at(0);
	product2 = matrixReduce(product2, -1 * (1 / factor2));
	result = matrixAdd(result, product1);
	result = matrixAdd(result, product2);
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
	vector<vector<double> > initHessian = getIdentityMatrix(inputs.at(0).size());
	vector<vector<double> > matrixQ = multiply(transpose(inputs), inputs);
	vector<double> constant = unboxVector(transpose(multiply(transpose(inputs), transpose(boxVector(output)))));
	vector<double> gradient = calculateGradient(initPoint, matrixQ, constant);
	for (int i = 0; i < epochs; i++) {
		vector<double> conjugateDirection = unboxVector(transpose(multiply(initHessian, transpose(boxVector(gradient)))));
		conjugateDirection = reduce(conjugateDirection, -1);
		double rate = determineRate(gradient, conjugateDirection, matrixQ);
		initPoint = add(initPoint, reduce(conjugateDirection, rate));
		vector<double> xDelta = reduce(conjugateDirection, rate);
		vector<double> newGradient = calculateGradient(initPoint, matrixQ, constant);
		vector<double> gradientDelta = add(newGradient, reduce(gradient, -1));
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

int main() {
	vector<double> weights = vector<double>(2, 0.0);
	NeuralNode node = NeuralNode(weights, 0.2);
	vector<double> trainedWeights;
	trainedWeights.push_back(0.1);
	trainedWeights.push_back(0.9);
	NeuralNode trainedNode = NeuralNode(trainedWeights, 0.2);
	vector<double> input1;
	input1.push_back(0.2);
	input1.push_back(0.8);
	vector<double> input2;
	input2.push_back(0.8);
	input2.push_back(0.2);
	vector<double> input3;
	input3.push_back(0.4);
	input3.push_back(0.5);
	vector<double> outputs;
	outputs.push_back(trainedNode.computeResult(input1));
	outputs.push_back(trainedNode.computeResult(input2));
	outputs.push_back(trainedNode.computeResult(input3));
	vector<vector<double> > inputs;
	inputs.push_back(input1);
	inputs.push_back(input2);
	inputs.push_back(input3);
	node.train(inputs, outputs, 100);
	print(node);

	cout << "Testing -- Second Section: Quasi-Newton method, DPS" << endl;
	vector<double> weights2 = vector<double>(2, 0.0);
	NeuralNode newtonNode = NeuralNode(weights2, 0.01);
	newtonNode.newtonTrain(inputs, outputs, 2);
	print(newtonNode);
}
