/*
 * MatrixOperations.h
 *
 *  Created on: Jun 4, 2018
 *      Author: richardyu
 */

#ifndef MATRIXOPERATIONS_H_
#define MATRIXOPERATIONS_H_
#include <vector>

using namespace std;

class MatrixOperations {
public:
	MatrixOperations();
	virtual ~MatrixOperations();
	static double dotProduct(vector<double>& vec1, vector<double>& vec2);
	static vector<double> reduce(vector<double> vec, double factor);
	static vector<double> divide(vector<double> vec1, vector<double> vec2);
	static vector<vector<double> > matrixReduce(vector<vector<double> > matrix, double factor);
	static vector<double> sqrt(vector<double> vec);
	static vector<double> add(vector<double> vec, vector<double> vec2);
	static vector<vector<double> > add(vector<vector<double> > mat1, vector<vector<double> > mat2);
	static vector<vector<double> > multiply(vector<vector<double> > matrix1, vector<vector<double> > matrix2);
	static vector<vector<double> > transpose(vector<vector<double> > matrix);
	static vector<vector<double> > matrixAdd(vector<vector<double> > mat1, vector<vector<double> > mat2);
	static vector<vector<double> > boxVector(vector<double> input);
	static vector<double> unboxVector(vector<vector<double> > input);
	static vector<vector<double> > getIdentityMatrix(int size);
	static vector<double> add(vector<double> vec, double increment);
};

#endif /* MATRIXOPERATIONS_H_ */
