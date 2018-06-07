/*
 * NeuralNode.h
 *
 *  Created on: May 27, 2018
 *      Author: richardyu
 */

#ifndef NEURALNODE_H_
#define NEURALNODE_H_

#include <vector>

using namespace std;

class NeuralNode {
private:
	double learningRate;
public:
	vector<double> weights;
	NeuralNode();
	NeuralNode(vector<double> modifiers, double rate);
	virtual ~NeuralNode();
	double computeResult(vector<double> inputs);
	vector<double> calculateGradient(vector<double> inputs,
					 vector<vector<double> > matrix,
					 vector<double> output);
	void train(vector<vector<double> > inputs, vector<double> outputs, double epochs);
	void newtonTrain(vector<vector<double> > inputs, vector<double> output, int epochs);
	static vector<vector<double> > computeHessian(vector<vector<double> > initHessian,
						      vector<double> xDelta,
						      vector<double> gradientDelta);
};

#endif /* NEURALNODE_H_ */
