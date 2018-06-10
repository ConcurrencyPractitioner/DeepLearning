/*
 * KFoldCrossValidation.cpp
 *
 *  Created on: Jun 10, 2018
 *      Author: richardyu
 */

#include "KFoldCrossValidation.h"
#include "FeedFowardNeuralNetwork.h"
#include "MatrixOperations.h"
#include "NeuralNode.h"

#include <math.h>
#include <vector>
#include <functional>

using namespace std;

MatrixOperations asisstant;

KFoldCrossValidation::KFoldCrossValidation(int num, int cases,
		vector<vector<double> > in,
		vector<vector<double> > out,
		FeedFowardNeuralNetwork& net) : folds(num), testCases(cases),
				inputs(in), outputs(out), network (net) {}

KFoldCrossValidation::~KFoldCrossValidation() {}

void excludeTestSet(reference_wrapper<vector<vector<double> > >& copiedInputs,
					reference_wrapper<vector<vector<double> > >& copiedOutputs,
					int first, int last) {
	for (int i = first; i < last; i++) {
		copiedInputs.get().erase(copiedInputs.get().begin() + first);
		copiedOutputs.get().erase(copiedOutputs.get().begin() + first);
	}
}

vector<double> KFoldCrossValidation::calculateError(int episodes) {
	if (folds <= 1) folds = 2;
	vector<int> testIndexes;
	for (int i = 0; i < folds; i++) {
		int nextIndex = floor((double) testCases / (double) folds * i);
		testIndexes.push_back(nextIndex);
	}
	reference_wrapper<FeedFowardNeuralNetwork> networkReference = network;
	reference_wrapper<vector<vector<double> > > inputsReference = inputs;
	reference_wrapper<vector<vector<double> > > outputsReference = outputs;
	vector<double> totalErrors;
	for (int i = 0; i < folds; i++) {
		reference_wrapper<FeedFowardNeuralNetwork> copiedNetwork = networkReference;
		reference_wrapper<vector<vector<double> > > copiedInputs = inputsReference;
		reference_wrapper<vector<vector<double> > > copiedOutputs = outputsReference;
		int first = testIndexes.at(i);
		int second = i == folds - 1 ? testIndexes.size() : testIndexes.at(i+1);
		excludeTestSet(copiedInputs, copiedOutputs, first, second);
		for (int i = 0; i < episodes; i++) {
			copiedNetwork.get().backPropagate(copiedInputs.get(), copiedOutputs.get());
		}
		vector<double> givenOutputs =
				copiedNetwork.get().fowardPropagate(inputsReference.get().at(first));
		vector<double> averageErrors =
				asisstant.add(givenOutputs, asisstant.reduce(outputsReference.get().at(first), -1));
		for (int j = first + 1; j < second; j++) {
			givenOutputs = copiedNetwork.get().fowardPropagate(inputsReference.get().at(first));
			vector<double> nextErrors =
					asisstant.add(givenOutputs, asisstant.reduce(outputsReference.get().at(first), -1));
			averageErrors = asisstant.add(nextErrors, averageErrors);
		}
		asisstant.reduce(averageErrors, 1.0 / (double) (second - first));
		totalErrors = totalErrors.size() == 0 ? averageErrors :
				asisstant.add(averageErrors, totalErrors);
	}
	return asisstant.reduce(totalErrors, 1.0 / (double) folds);
}


