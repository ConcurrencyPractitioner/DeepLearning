/*
 * KFoldCrossValidation.h
 *
 *  Created on: Jun 10, 2018
 *      Author: richardyu
 */

#ifndef KFOLDCROSSVALIDATION_H_
#define KFOLDCROSSVALIDATION_H_
#include "FeedFowardNeuralNetwork.h"

#include <vector>

using namespace std;

class KFoldCrossValidation {
private:
	int folds, testCases;
	vector<vector<double> > inputs, outputs;
	FeedFowardNeuralNetwork& network;
public:
	KFoldCrossValidation(int num, int cases,
			vector<vector<double> > in,
			vector<vector<double> > out,
			FeedFowardNeuralNetwork& net);
	virtual ~KFoldCrossValidation();
	vector<double> calculateError(int episodes);
};

#endif /* KFOLDCROSSVALIDATION_H_ */
