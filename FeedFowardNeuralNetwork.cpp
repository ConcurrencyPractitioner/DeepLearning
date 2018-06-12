/*
 * FeedFowardNeuralNetwork.cpp
 *
 *  Created on: Jun 2, 2018
 *      Author: richardyu
 */

#include "FeedFowardNeuralNetwork.h"
#include "NeuralNode.h"
#include "MatrixOperations.h"
#include <math.h>
#include <iostream>

using namespace std;

typedef pair<double, double> DataPair;

MatrixOperations helper;

const double episilon = pow(10, -9);

FeedFowardNeuralNetwork::FeedFowardNeuralNetwork(double size)
: stepSize (size) {}

FeedFowardNeuralNetwork::~FeedFowardNeuralNetwork() {}

void FeedFowardNeuralNetwork::addLayer(int size, int weightCount, vector<vector<double> >  weights) {
	vector<NeuralNode> layer;
	for (int i = 0; i < size; i++) {
		layer.push_back(NeuralNode(weights.at(i), stepSize));
	}
	this -> nodes.push_back(layer);
	this -> memorizedData.push_back(vector<DataPair>(size, DataPair(0.0, 0.0)));
}

void FeedFowardNeuralNetwork::modifyWeight(int layer,
		int index,
		int weightIndex,
		double newWeight) {
	this -> nodes.at(layer).at(index).weights.at(weightIndex) = newWeight;
}

// sigmoid function
double FeedFowardNeuralNetwork::activationFunction(double input) {
	double sum = 1 + exp(-1 * input);
	return 1.0 / sum;
}

double FeedFowardNeuralNetwork::derivativeFunction(double input) {
	return activationFunction(input) * (1 - activationFunction(input));
}

vector<double> FeedFowardNeuralNetwork::fowardPropagate(vector<double> inputs) {
	for (int i = 0; i < nodes.size(); i++) {
		vector<double> next;
		for (int j = 0; j < nodes.at(i).size(); j++) {
			double unboundedInput = nodes.at(i).at(j).computeResult(inputs);
			double boundedInput = activationFunction(unboundedInput);
			memorizedData.at(i).at(j).first = unboundedInput;
			memorizedData.at(i).at(j).second = boundedInput;
			next.push_back(boundedInput);
		}
		inputs = next;
	}
	return inputs;
}

vector<vector<vector<double> > > FeedFowardNeuralNetwork::calculateGradient(vector<double> inputs,
		vector<double> outputs, int stopLayer) {
	vector<vector<vector<double> > > gradients;
	vector<double> results = fowardPropagate(inputs);
	vector<double> negativeResults = helper.reduce(results, -1);
	vector<double> outputErrors = helper.add(outputs, negativeResults);
	int last = nodes.size() - 1;
	vector<vector<double> > prevWeights;
	vector<vector<double> > layerGradient;
	for (int j = 0; j < outputErrors.size(); j++) {
		outputErrors.at(j) *= results.at(j) * (1 - results.at(j));
		vector<double> nodeWeights;
		vector<double> nodeGradient;
		for (int k = 0; k < nodes.at(last).at(j).weights.size(); k++) {
			nodeWeights.push_back(nodes.at(last).at(j).weights.at(k));
			nodeGradient.push_back(outputErrors.at(j) * memorizedData.at(last - 1).at(k).second);
			nodeGradient.at(nodeGradient.size() - 1) *= -1;
		}
		layerGradient.push_back(nodeGradient);
		prevWeights.push_back(nodeWeights);
	}
	gradients.insert(gradients.begin(), layerGradient);
	for (int j = nodes.size() - 2; j >= stopLayer; j--) {
		layerGradient = vector<vector<double> >();
		vector<vector<double> > nextWeights;
		vector<double> newErrors;
		for (int k = 0; k < nodes.at(j).size(); k++) {
			vector<double> nodeWeights;
			double totalError = 0.0;
			for (int count = 0; count < nodes.at(j+1).size(); count++) {
				totalError += outputErrors.at(count) * prevWeights.at(count).at(k);
			}
			double activationGradient = derivativeFunction(memorizedData.at(j).at(k).first);
			double newError = totalError * activationGradient;
			newErrors.push_back(newError);
			vector<double> nodeGradient;
			for (int l = 0; l < nodes.at(j).at(k).weights.size(); l++) {
				nodeWeights.push_back(nodes.at(j).at(k).weights.at(l));
				double input = j-1 == -1 ? inputs.at(l) : memorizedData.at(j-1).at(l).second;
				nodeGradient.push_back(-1 * newError * input);
			}
			nextWeights.push_back(nodeWeights);
			layerGradient.push_back(nodeGradient);
		}
		outputErrors = newErrors;
		prevWeights = nextWeights;
		gradients.insert(gradients.begin(), layerGradient);
	}
	return gradients;
}

/**
 * Performs one episode of back propagation on given inputs and outputs
 */
void FeedFowardNeuralNetwork::backPropagate(vector<vector<double> > inputs,
		vector<vector<double> > outputs) {
	for (int z = 0; z < inputs.size(); z++) {
		vector<vector<vector<double> > > gradients = calculateGradient(inputs.at(z), outputs.at(z));
		for (int i = 0; i < nodes.size(); i++) {
			for (int j = 0; j < nodes.at(i).size(); j++) {
				for (int k = 0; k < nodes.at(i).at(j).weights.size(); k++) {
					nodes.at(i).at(j).weights.at(k) -= stepSize * gradients.at(i).at(j).at(k);
				}
			}
		}
	}
}

/**
 * Uses the secant method to perform a line search to reach target.
 */
double FeedFowardNeuralNetwork::determineRate(vector<vector<vector<double> > > currGradients,
		vector<double> conjugateDirection, int i, int j,
		vector<double> inputs, vector<double>  outputs) {
	double alpha = 0.5;
	double change = 0.5;
	vector<vector<vector<double> > > newGradients;
	while (abs(change) > episilon) {
		nodes.at(i).at(j).weights = helper.add(nodes.at(i).at(j).weights,
				helper.reduce(conjugateDirection, alpha));
		newGradients = calculateGradient(inputs, outputs);
		double prevInput = i-1 == -1 ? inputs.at(0) : memorizedData.at(i-1).at(0).second;
		vector<double> predecessorInputs;
		if (i-1 == -1) predecessorInputs = inputs;
		else {
			for (int k = 0; k < nodes.at(i-1).size(); k++) {
				predecessorInputs.push_back(memorizedData.at(i-1).at(k).second);
			}
		}
		double oldAlphaGradient = currGradients.at(i).at(j).at(0) / prevInput *
				helper.dotProduct(conjugateDirection, predecessorInputs);
		double newAlphaGradient = newGradients.at(i).at(j).at(0) / prevInput *
				helper.dotProduct(conjugateDirection, predecessorInputs);
		nodes.at(i).at(j).weights = helper.add(nodes.at(i).at(j).weights,
				helper.reduce(conjugateDirection, -1 * alpha));
		if (newAlphaGradient - oldAlphaGradient == 0.0) {
			return alpha;
		}
		alpha -= change / (newAlphaGradient - oldAlphaGradient) * newAlphaGradient;
		change = -1 * change / (newAlphaGradient - oldAlphaGradient) * newAlphaGradient;
		currGradients = newGradients;
	}
	return alpha;
}

/**
 * This does not work for all cases, as of yet. The Hessian matrix is not guaranteed to be positive
 * definite. Consequently, it does not converge if the error function is not close to the point.
 * This could be solved if we add a identity matrix multiplied by a factor which would make it so.
 * (This is a implementation of the DFP algorithm, not of the more popular BFGS. So occasionally,
 * might seem to stop at saddle points)
 */
void FeedFowardNeuralNetwork::newtonBackPropagate(vector<vector<double> > inputs,
		vector<vector<double> > outputs, int episodes) {
	for (int z = 0; z < inputs.size(); z++){
		vector<vector<vector<double> > > gradients = calculateGradient(inputs.at(z), outputs.at(z));
		vector<vector<vector<vector<double> > > > hessianMatrices;
		for (int i = 0; i < nodes.size(); i++) {
			vector<vector<vector<double> > > layerHessians;
			for (int j = 0; j < nodes.at(i).size(); j++) {
				vector<vector<double> > nodeHessian =
						helper.getIdentityMatrix(nodes.at(i).at(j).weights.size());
				layerHessians.push_back(nodeHessian);
			}
			hessianMatrices.push_back(layerHessians);
		}
		vector<double> point;
		for (int count = 0; count < episodes; count++) {
			vector<vector<vector<double> > > conjugateDirections;
			vector<vector<double> > rates;
			for (int i = 0; i < nodes.size(); i++) {
				vector<vector<double> > conjugateDirectionList =
						vector<vector<double> >(nodes.at(i).size(), vector<double>(1, 0.0));
				vector<double> layerRates = vector<double>(nodes.at(i).size(), 0.0);
				conjugateDirections.push_back(conjugateDirectionList);
				rates.push_back(layerRates);
			}
			for (int i = 0; i < nodes.size(); i++) {
				for (int j = 0; j < nodes.at(i).size(); j++) {
					point = nodes.at(i).at(j).weights;
					vector<vector<double> > currHessian = hessianMatrices.at(i).at(j);
					vector<double> conjugateDirection =
							helper.unboxVector(helper.transpose(helper.multiply(currHessian,
									helper.transpose(helper.boxVector(gradients.at(i).at(j))))));
					conjugateDirection = helper.reduce(conjugateDirection, -1);
					conjugateDirections.at(i).at(j) = conjugateDirection;
					rates.at(i).at(j) = determineRate(gradients,
							conjugateDirection, i, j, inputs.at(z), outputs.at(z));
					point = helper.add(point, helper.reduce(conjugateDirections.at(i).at(j),
							rates.at(i).at(j)));
					nodes.at(i).at(j).weights = point;
				}
			}
			vector<vector<vector<double> > > newGradients =
					calculateGradient(inputs.at(z), outputs.at(z));
			for (int i = 0; i < nodes.size(); i++) {
				for (int j = 0; j < nodes.at(i).size(); j++) {
					vector<double> xDelta = helper.reduce(conjugateDirections.at(i).at(j),
							rates.at(i).at(j));
					vector<double> gradientDelta =
							helper.add(newGradients.at(i).at(j),
									helper.reduce(gradients.at(i).at(j), -1));
					hessianMatrices.at(i).at(j) =
							nodes.at(i).at(j).computeHessian(hessianMatrices.at(i).at(j),
									xDelta, gradientDelta);
				}
			}
			gradients = newGradients;
		}
	}
}

int main() {
	FeedFowardNeuralNetwork network = FeedFowardNeuralNetwork(21);
	vector<vector<double> > layer1;
	vector<double> node1;
	vector<double> node2;
	node1.push_back(0.1);
	node1.push_back(0.3);
	node2.push_back(0.3);
	node2.push_back(0.4);
	layer1.push_back(node1);
	layer1.push_back(node2);
	network.addLayer(2, 2, layer1);
	vector<vector<double> > layer2;
	vector<double> middleNode;
	middleNode.push_back(0.4);
	middleNode.push_back(0.6);
	layer2.push_back(middleNode);
	network.addLayer(1, 2, layer2);
	vector<double> input;
	input.push_back(0.2);
	input.push_back(0.6);
	vector<double> result = network.fowardPropagate(input);
	cout << result.at(0) << endl;
	vector<double> direction;
	direction.push_back(1);
	direction.push_back(0);
	vector<double> output = vector<double>(1, 0.7);
	vector<vector<vector<double> > > gradients = network.calculateGradient(input, output);
	network.newtonBackPropagate(helper.boxVector(input), helper.boxVector(vector<double>(1, 0.7)), 1);
	result = network.fowardPropagate(input);
	cout << result.at(0) << endl;
}


