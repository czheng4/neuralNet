/*
	2019 chaohui zheng
	a example of suing matrix class
*/
#include <iostream>
#include "mat.h"
using namespace std;
using namespace zzh;


int main()
{

	vector <int> v;
	v.push_back(1);
	v.push_back(2);
	v.push_back(3);

	vector <vector <double> > v1;
	v1.resize(2);
	v1[0].push_back(1);
	v1[0].push_back(2);
	v1[0].push_back(3);
	v1[1].push_back(4);
	v1[1].push_back(5);
	v1[1].push_back(6);
	/* three ways to initialize matrix*/
	

	Mat<int> m(v); // initialize with one dimentional vector. Note: the data type of vector should match up to matrix
	cout << m; cout << "---------------------------" << endl;

	Mat<double> m1(v1); // initialize with two dimentional vector
	cout << m1; cout << "---------------------------" << endl;

	Mat<int> m2(3,2); // initialize with the shape of matrix. 3 rows and 2 cols
	m2 << 1 << 2 << 3 << 4 << 5 << 6; // getting number 
	cout << m2; cout << "---------------------------" << endl;

	/* cast data type by calling "cast" member function */
	// Mat<double> m3 = m;  error
	Mat<double> m3 = m.cast<double>();
	cout << m3; cout << "---------------------------" << endl;


	/* some simple calculation */
	cout << m1 + 2; cout << "---------------------------" << endl;
	cout << m1 * 2; cout << "---------------------------" << endl;
	cout << m1.dot(m2.cast<double>()); cout << "---------------------------" << endl; // matrix multiplication

	cout << m1 + m2.reshape(2,3).cast<double>(); cout << "---------------------------" << endl; // reshape matrix

	cout << zzh::sum(m1) << endl; cout << "---------------------------" << endl;




}
