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
#include <cmath>
#include <iostream>

using namespace std;

typedef pair<double, double> DataPair;

MatrixOperations helper;

const double episilon = pow(10, -1000);

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

vector<vector<vector<double> > > scaleToGradients(vector<vector<vector<double> > >& gradients) {
	vector<vector<vector<double> > > emptyMomentVector = vector<vector<vector<double> > >();
	for (int i = 0; i < gradients.size(); i++) {
		emptyMomentVector.push_back(vector<vector<double> >());
		for (int j = 0; j < gradients.at(i).size(); j++) {
			emptyMomentVector.at(i).push_back(vector<double>(gradients.at(i).at(j).size(), 0));
		}
	}
	return emptyMomentVector;
}

void checkForScale(vector<vector<vector<double> > > gradients,
		vector<vector<vector<double> > >& momentVectors1,
		vector<vector<vector<double> > >& momentVectors2) {
	if (momentVectors1.size() == 0) {
		momentVectors1 = scaleToGradients(gradients);
		momentVectors2 = scaleToGradients(gradients);
	}
}

/**
 * ADAM optimizer: implemented based on the following paper in arxiv:
 * https://arxiv.org/pdf/1412.6980.pdf
 */
void FeedFowardNeuralNetwork::adamOptimize(vector<vector<double> > inputs,
		vector<vector<double> > outputs,
		double stepSize = 0.001,
		double decayRate1 = 0.9,
		double decayRate2 = 0.999,
		double epsilon = 0.00000001) {
	vector<vector<vector<double> > > momentVectors1 = vector<vector<vector<double> > >();
	vector<vector<vector<double> > > momentVectors2 = vector<vector<vector<double> > >();
	int timestep = 0;
	for (int z = 0; z < inputs.size(); z++) {
		vector<vector<vector<double> > > gradients = calculateGradient(inputs.at(z), outputs.at(z));
		checkForScale(gradients, momentVectors1, momentVectors2);
		timestep++;
		for (int i = 0; i < gradients.size(); i++) {
			for (int j = 0; j < gradients.at(i).size(); j++) {
				vector<double> moment1 = momentVectors1[i][j];
				vector<double> moment2 = momentVectors2[i][j];
				vector<double> localGradient = gradients[i][j];
				moment1 = helper.add(helper.reduce(moment1, decayRate1),
						helper.reduce(localGradient, 1.0 - decayRate1));
				vector<double> squaredGradient = helper.dotProduct(localGradient, localGradient);
				moment2 = helper.add(helper.reduce(moment2, decayRate2),
						helper.reduce(squaredGradient, decayRate2));
				momentVectors1[i][j] = moment1;
				momentVectors2[i][j] = moment2;
				vector<double> biasCorrectedMoment1 =
						helper.reduce(moment1, 1.0 / (1.0 - pow(decayRate1, timestep)));
				vector<double> biasCorrectedMoment2 =
						helper.reduce(moment2, 1.0 / (1.0 - pow(decayRate2, timestep)));
				vector<double> gagedGradient =
						helper.divide(biasCorrectedMoment1,
								helper.add(helper.sqrt(biasCorrectedMoment2),
										epsilon));
				gagedGradient = helper.reduce(gagedGradient, stepSize);
				for (int k = 0; k < gagedGradient.size(); k++) {
					nodes.at(i).at(j).weights.at(k) -= gagedGradient[k];
				}
			}
		}
	}
}

/**
 * Secant method
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
		if (newAlphaGradient < episilon || newAlphaGradient - oldAlphaGradient == 0.0) {
			return alpha;
		}
		alpha -= change / (newAlphaGradient - oldAlphaGradient) * newAlphaGradient;
		change = -1 * change / (newAlphaGradient - oldAlphaGradient) * newAlphaGradient;
		currGradients = newGradients;
	}
	return alpha;
}

/**
 * (This is a implementation of the DFP algorithm, not of the more popular BFGS. So occasionally,
 * might seem to converge slower than first order gradient descents.)
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
	outputNode1.push_back(0.7);
	outputNode2.push_back(0.6);
	outputNodes.push_back(outputNode1);
	outputNodes.push_back(outputNode2);
	network.addLayer(2, 1, outputNodes);
	vector<double> input;
	input.push_back(0.2);
	input.push_back(0.6);
	vector<double> direction;
	direction.push_back(1);
	direction.push_back(0);
	double goal; cout << "Output: " << endl; cin >> goal;
	vector<double> output = vector<double>(2, goal);
	int iterations; cout << "Iterations? " << endl; cin >> iterations;
	vector<double> result = network.fowardPropagate(input);
	cout << "Before second order back propagate (results): " << endl;
	cout << result.at(0) << endl;
	cout << result.at(1) << endl;
	network.newtonBackPropagate(helper.boxVector(input), helper.boxVector(output), iterations);
	result = network.fowardPropagate(input);
	cout << "After training was complete (" << iterations << " iterations): " << endl;
	cout << result.at(0) << endl;
	cout << result.at(1) << endl;
	FeedFowardNeuralNetwork network2 = FeedFowardNeuralNetwork(10);
	network2.addLayer(2, 2, layer1);
	network2.addLayer(1, 2, layer2);
	network2.addLayer(2, 1, outputNodes);
	result = network2.fowardPropagate(input);
	cout << "Before first order back propagate (results): " << endl;
	cout << result.at(0) << endl;
	cout << result.at(1) << endl;
	for (int i = 0; i < iterations; i++) {
		network2.backPropagate(helper.boxVector(input), helper.boxVector(output));
	}
	result = network2.fowardPropagate(input);
	cout << "After propagate (" << iterations << " iterations): " << endl;
	cout << result.at(0) << endl;
	cout << result.at(1) << endl;
}

