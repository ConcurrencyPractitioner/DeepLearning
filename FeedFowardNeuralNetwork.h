/*
 * FeedFowardNeuralNetwork.h
 *
 *  Created on: Jun 2, 2018
 *      Author: richardyu
 */

#ifndef FEEDFOWARDNEURALNETWORK_H_
#define FEEDFOWARDNEURALNETWORK_H_

#include "NeuralNode.h"
#include <vector>

using namespace std;

typedef pair<double, double> DataPair;

class FeedFowardNeuralNetwork {
private:
	vector<vector<NeuralNode> > nodes;
	vector<vector<DataPair> > memorizedData;
	double stepSize;
public:
	FeedFowardNeuralNetwork(double size);
	virtual ~FeedFowardNeuralNetwork();
	void addLayer(int size, int weightCount, vector<vector<double> > weights);
	vector<double> fowardPropagate(vector<double> inputs);
	vector<vector<vector<double> > > calculateGradient(vector<double> inputs,
													   vector<double> outputs,
													   int stopLayer = 0);
	void backPropagate(vector<vector<double> > inputs, vector<vector<double> > outputs);
	double determineRate(vector<vector<vector<double> > >& currGradients,
						 vector<double> conjugateDirection, int i, int j,
						 vector<double> directInputs);
	void newtonBackPropagate(vector<vector<double> > inputs,
			                 vector<vector<double> > outputs,
							 int episodes);
	double activationFunction(double input);
	double derivativeFunction(double input);
	void modifyWeight(int layer, int index, int weightIndex, double newWeight);
};

#endif /* FEEDFOWARDNEURALNETWORK_H_ */
