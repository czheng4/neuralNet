/*	Libraries for simple N.N.
	2019 chaohui zheng */

#ifndef NET_H
#define NET_H
#include <iostream>
#include <string>
#include <vector>
#include <fstream>
#include <chrono>
#include <random>
#include <cmath>
#include "mat.h"
#include <cstdlib>
#include <cstdio>
using namespace zzh;
using namespace std;

namespace zzh
{

	// sigmoid function
	inline Mat<double> sigmoid(Mat<double> mat);

	// the derivate of sigmoid
	inline Mat<double> sigmoid_prime(Mat<double> &mat);

	/* normal distribution generator */
	inline Mat<double> gen_random_number(class shape s);


	// shape class 
	class shape
	{
		public:
			shape();
			shape(int rows, int cols);
			int rows;
			int cols;
	};

	

	// store the data info (input and correct result)
	class pair
	{
		public:
			Mat<double> input;
			Mat<double> mat_result;
			int result;
	};


	enum CostFunction
	{
		QuadraticCost,
		CrossEntropyCost
	};


	class QuadraticCost
	{
		public:
			static double fn(Mat<double> &a, Mat<double> &y);
			static Mat<double> delta(Mat<double> &z, Mat<double> &a,  Mat<double> &y);
	};

	class CrossEntropyCost
	{
		public:

			static double fn(Mat<double> &a, Mat<double> &y);
			static Mat<double> delta(Mat<double> &z, Mat<double> &a, Mat<double> &y);
	};

	// data class
	class data
	{
		public:
			data(const string &input_name, const shape &input, const string &result_name, const int &output_num);
			data(){};
			vector <zzh::pair*> pairs;
			void read_input(const string &file, const shape &input);
			void read_result(const string &file, const int &output_num);
			int sizes;
			void shuffle();

	};

	

	// the layer of neural network
	class Layer
	{
		public:
			Layer(const shape &s);
			Layer();
			void cleanDelta();
			class shape s;
			Mat<double> weights;
			Mat<double> biases;
			Mat<double> delta_w;
			Mat<double> delta_b;
			Mat<double> z;
			Mat<double> activation;
	};

	// network class
	class Net
	{
		public:
			int num_layers;
			vector <Layer*> layers;
			CostFunction cost_function;
			Net(vector <int> sizes, CostFunction cost = CostFunction::CrossEntropyCost);
			Net(string load);
			vector <int> sizes;

			Mat<double> feedforward(Mat <double> a); // 'a' means activation
			void SGD(data &training_data, int epochs, int mini_batch_size, double eta, double lambda, const data &test_data,\
					bool monitor_test_cost = false, bool monitor_test_accuracy = false,\
					bool monitor_training_cost = false, bool monitor_training_accuracy = false);

			void SGD(data &training_data, int epochs, int mini_batch_size, double eta, double lambda, \
					bool monitor_training_cost = false, bool monitor_training_accuracy = false);

			void backprob(Mat<double> inputs, Mat<double> results);


			void save(const string &output);
			void update_mini_batch(const data &mini_batch, double eta, double lambda, int num_data);
			int evaluate(const data &test_data);
			double total_cost(const data &data, double lambda);
			void print();

	};

}

#endif

