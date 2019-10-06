/*
 * MatrixOperations.cpp
 *
 *  Created on: Jun 4, 2018
 *      Author: richardyu
 */

#include "MatrixOperations.h"
#include <vector>
#include <cmath>

using namespace std;

MatrixOperations::MatrixOperations() {}

MatrixOperations::~MatrixOperations() {}

double MatrixOperations::dotProduct(vector<double>& vec1, vector<double>& vec2) {
	double sum = 0;
	if (vec1.size() != vec2.size()) return -1;
	for (int i = 0; i < vec1.size(); i++) {
		sum += vec1.at(i) * vec2.at(i);
	}
	return sum;
}

vector<double> MatrixOperations::add(vector<double> vec, double increment) {
	vector<double> res = vector<double>(res.size(), 0);
	for (int i = 0; i < vec.size(); i++) res[i] = vec[i] + increment;
	return res;
}

vector<double> MatrixOperations::divide(vector<double> vec1, vector<double> vec2) {
	vector<double> res = vector<double>(vec1.size(), 0);
	for (int i = 0; i < res.size(); i++) {
		res[i] = vec1[i] / vec2[i];
	}
	return res;
}

vector<double> MatrixOperations::sqrt(vector<double> vec) {
	vector<double> res = vector<double>(vec.size(), 0);
	for (int i = 0; i < vec.size(); i++) {
		res[i] = sqrt(vec[i]);
	}
	return res;
}

vector<double> MatrixOperations::reduce(vector<double> vec, double factor) {
	vector<double> res = vec;
	for (int i = 0; i < vec.size(); i++) {
		res.at(i) *= factor;
	}
	return res;
}

vector<vector<double> > MatrixOperations::matrixReduce(vector<vector<double> > matrix, double factor) {
	vector<vector<double> > result = matrix;
	for (int i = 0; i < matrix.size(); i++) {
		for (int j = 0; j < matrix.at(i).size(); j++) {
			result.at(i).at(j) *= factor;
		}
	}
	return result;
}

vector<double> MatrixOperations::add(vector<double> vec, vector<double> vec2) {
	vector<double> res = vec;
	for (int i = 0; i < vec2.size(); i++) {
		res.at(i) += vec2.at(i);
	}
	return res;
}

vector<vector<double> > MatrixOperations::add(vector<vector<double> > mat1, vector<vector<double> > mat2) {
	vector<vector<double> > result = mat1;
	for (int i = 0; i < mat1.size(); i++) {
		for (int j = 0; j < mat1.at(i).size(); j++) {
			result.at(i).at(j) += mat2.at(i).at(j);
		}
	}
	return result;
}

vector<vector<double> > MatrixOperations::multiply(vector<vector<double> > matrix1, vector<vector<double> > matrix2) {
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

vector<vector<double> > MatrixOperations::transpose(vector<vector<double> > matrix) {
	vector<vector<double> > result =
			vector<vector<double> > (matrix.at(0).size(), vector<double>(matrix.size(), 0.0));
	for (int i = 0; i < matrix.size(); i++) {
		for (int j = 0; j < matrix.at(i).size(); j++) {
			result.at(j).at(i) = matrix.at(i).at(j);
		}
	}
	return result;
}

vector<vector<double> > MatrixOperations::matrixAdd(vector<vector<double> > mat1, vector<vector<double> > mat2) {
	vector<vector<double> > result = mat1;
	for (int i = 0; i < mat2.size(); i++) {
		for (int j = 0; j < mat2.at(i).size(); j++) {
			result.at(i).at(j) += mat2.at(i).at(j);
		}
	}
	return result;
}

vector<vector<double> > MatrixOperations::boxVector(vector<double> input) {
	vector<vector<double> > res;
	res.push_back(input);
	return res;
}

vector<double> MatrixOperations::unboxVector(vector<vector<double> > input) {
	return input.at(0);
}

vector<vector<double> > MatrixOperations::getIdentityMatrix(int size) {
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

