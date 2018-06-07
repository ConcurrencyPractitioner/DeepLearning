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
	return 1 / sum;
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

vector<vector<vector<double> > > FeedFowardNeuralNetwork::calculateGradient(
		vector<vector<double> > inputs,
		vector<vector<double> > outputs) {
	vector<vector<vector<double> > > gradients;
	for (int i = 0; i < inputs.size(); i++) {
		vector<double> results = fowardPropagate(inputs.at(i));
		vector<double> negativeResults = helper.reduce(results, -1);
		vector<double> outputErrors = helper.add(outputs.at(i), negativeResults);
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
		for (int j = nodes.size() - 2; j >= 0; j--) {
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
					double input = j-1 == -1 ? inputs.at(i).at(l) : memorizedData.at(j-1).at(l).second;
					nodeGradient.push_back(-1 * newError * input);
				}
				nextWeights.push_back(nodeWeights);
				layerGradient.push_back(nodeGradient);
			}
			outputErrors = newErrors;
			prevWeights = nextWeights;
			gradients.insert(gradients.begin(), layerGradient);
		}
	}
	return gradients;
}

/**
 * Performs one episode of back propagation on given inputs and outputs
 */
void FeedFowardNeuralNetwork::backPropagate(vector<vector<double> > inputs,
		vector<vector<double> > outputs) {
	vector<vector<vector<double> > > gradients = calculateGradient(inputs, outputs);
	for (int i = 0; i < nodes.size(); i++) {
		for (int j = 0; j < nodes.at(i).size(); j++) {
			for (int k = 0; k < nodes.at(i).at(j).weights.size(); k++) {
				nodes.at(i).at(j).weights.at(k) -= stepSize * gradients.at(i).at(j).at(k);
			}
		}
	}
}

int main() {
	FeedFowardNeuralNetwork network = FeedFowardNeuralNetwork(10);
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
	vector<vector<double> > outputNodes;
	vector<double> outputNode1;
	vector<double> outputNode2;
	outputNode1.push_back(0.4);
	outputNode2.push_back(0.9);
	outputNodes.push_back(outputNode1);
	outputNodes.push_back(outputNode2);
	network.addLayer(2, 1, outputNodes);
	vector<double> input;
	input.push_back(0.2);
	input.push_back(0.6);
	vector<double> result = network.fowardPropagate(input);
	cout << result.at(0) << endl;
	for (int i = 0; i < 15; i++) {
		network.backPropagate(helper.boxVector(input), helper.boxVector(vector<double>(2, 0.7)));
	}
	result = network.fowardPropagate(input);
	cout << result.at(0) << endl;
}


