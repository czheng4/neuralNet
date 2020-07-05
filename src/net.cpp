/*	Libraries for simple N.N.
	2019 chaohui zheng */

#include <iostream>
#include <string>
#include <vector>
#include <fstream>
#include <chrono>
#include <random>
#include <cmath>
#include "net.h"
#include "mat.h"
#include <cstdlib>
#include <cstdio>
using namespace zzh;
using namespace std;

namespace zzh
{

	shape::shape(){};
	shape::shape(int r, int c) : rows(r), cols(c) {};


	double QuadraticCost::fn(Mat<double> &a, Mat<double> &y)
	{
		double rv = zzh::norm(a-y);
		return 0.5 * rv * rv;
	}

	Mat<double> QuadraticCost::delta(Mat<double> &z, Mat<double> &a,  Mat<double> &y)
	{
		return (a-y)*sigmoid_prime(z);
	}

	double CrossEntropyCost::fn(Mat<double> &a, Mat<double> &y)
	{
		return zzh::sum(-y * zzh::log(a) - (1.0-y) * zzh::log(1.0-a));
	}

	Mat<double> CrossEntropyCost::delta(Mat<double> &z, Mat<double> &a, Mat<double> &y)
	{
		return (a-y);
	}

	//shuffle the data
	void data::shuffle()
	{
		zzh::pair *tmp;
		int rn;
		for(int i = pairs.size() - 1; i >= 0; i--)
		{
			rn = lrand48()%(i+1);

			tmp = pairs[i]; 
			pairs[i] = pairs[rn]; 
			pairs[rn] = tmp; 	
		}
	}
	// read input file
	void data::read_input(const string &file, const shape &s)
	{
		fstream myfile(file);
		double d;
		Mat<double> m(1,s.cols);
		zzh::pair *p;

		for(int i = 0; i < s.rows; i++)
		{
			p = pairs[i];
			for(int j = 0; j < s.cols; j++)
			{
				myfile >> d;
				m.set(0,j,d);
				assert(!myfile.eof());
			}
			p->input = m;

		}
		myfile.close();
	}

	// read result file
	void data::read_result(const string &file, const int &output_num)
	{
		fstream myfile(file);
		vector <int> results;
		int a;
		int count = 0;
		zzh::pair *p;
		for(int i = 0; i < sizes; i++)
		{
			myfile >> a;
			p = pairs[count];
			p->mat_result = Mat<double>::vectorize(output_num,a);

			p->result = a;
			count++;

		}
		assert(count == sizes);
		myfile.close();
	}
	data::data(const string &input_name, const shape &input, const string &result_name, const int &output_num)
	{
		sizes = input.rows;
		zzh::pair *p;
		for(int i = 0; i < sizes; i++)
		{
			p = new zzh::pair();
			pairs.push_back(p);
		}
		read_result(result_name, output_num);
		read_input(input_name, input);

	}

	Layer::Layer(){}
	void Layer::cleanDelta()
	{
		delta_w = Mat<double>::zeros(s.rows,s.cols);
		delta_b = Mat<double>::zeros(1,s.cols);
	}
	Layer::Layer(const shape &s)
	{
		this->s = s;
		weights = gen_random_number(s);
		biases = gen_random_number(shape(1,s.cols));
		delta_w = Mat<double>::zeros(s.rows,s.cols);
		delta_b = Mat<double>::zeros(1,s.cols);
	}

	void Net::print()
	{
		for(int i = 0; i < layers.size(); i++)
		{
			cout << "weights" << endl;
			cout << layers[i]->weights;
			cout << "biases" << endl;
			cout << layers[i]->biases;
		}
	}

	double Net::total_cost(const data &data, double lambda)
	{
		double size = data.pairs.size();
		double cost = 0, weights_cost = 0;
		double temp;
		for(int i = 0; i < size; i++)
		{
			Mat<double> a = feedforward(data.pairs[i]->input);
			if(cost_function == CostFunction::CrossEntropyCost) cost += CrossEntropyCost::fn(a, data.pairs[i]->mat_result);
			if(cost_function == CostFunction::QuadraticCost) cost += QuadraticCost::fn(a, data.pairs[i]->mat_result);
		}
		cost /= size;

		for(int i = 0; i < layers.size(); i++)
		{
			temp = zzh::norm(layers[i]->weights);
			weights_cost += temp * temp;
		}
		cost += 0.5 * (lambda / size) * weights_cost;
		return cost;
	}

	void Net::save(const string &output)
	{
		fstream myfile(output,fstream::out);
		for(int i = 0; i < sizes.size(); i++) myfile << sizes[i] << " ";
		myfile << endl;
		for(int i = 0; i < layers.size(); i++)
		{
			myfile << "weights" << endl;
			myfile << layers[i]->weights;
			myfile << "biases" << endl;
			myfile << layers[i]->biases;
		}
		myfile << "cost" << endl;
		if(cost_function == CostFunction::CrossEntropyCost) myfile << "CrossEntropyCost" << endl;
		if(cost_function == CostFunction::QuadraticCost) myfile << "QuadraticCost" << endl;
		myfile.close();
	}

	int Net::evaluate(const data &test_data)
	{
		int size = test_data.pairs.size();
		int correct = 0;

		for(int i = 0; i < size; i++)
		{
			Mat<double> results = feedforward(test_data.pairs[i]->input);
			if(results.argmax() == test_data.pairs[i]->result) correct++;
		}
		return correct;
	}

	void Net::SGD(data &training_data, int epochs, int mini_batch_size, double eta, double lambda,bool monitor_training_cost,bool monitor_training_accuracy)
	{
		cout << "start training" << endl;
		data mini_batch;
		mini_batch.sizes = mini_batch_size;
		mini_batch.pairs.resize(mini_batch_size);
		int num_batch = training_data.sizes/mini_batch_size;
		int count;
		for(int i = 0; i < epochs; i++)
		{
			training_data.shuffle();
			// partition the mini batch

			for(int j = 0; j < num_batch; j++)
			{
				count = 0;
				for(int k = j * mini_batch_size; k < (j + 1) * mini_batch_size; k++)
				{
					mini_batch.pairs[count] = training_data.pairs[k];
					count++;
				}

				update_mini_batch(mini_batch, eta,lambda,(int)training_data.sizes);


			}
			printf("Epoch %d training complete\n",i+1);
			if(monitor_training_accuracy)
				printf("Accuracy on training data %d / %d\n",evaluate(training_data),training_data.sizes);


			if(monitor_training_cost)
				printf("Cost on training data %f \n",total_cost(training_data, lambda));

			printf("\n");
		}
		cout << "end training" << endl;
	}


	void  Net::SGD(data &training_data, int epochs, int mini_batch_size, double eta,double lambda, const data &test_data,\
			bool monitor_test_cost, bool monitor_test_accuracy, bool monitor_training_cost, bool monitor_training_accuracy)
	{

		cout << "start training" << endl;
		data mini_batch;
		mini_batch.sizes = mini_batch_size;
		mini_batch.pairs.resize(mini_batch_size);
		int num_batch = training_data.sizes/mini_batch_size;
		int count;
		for(int i = 0; i < epochs; i++)
		{
			training_data.shuffle();
			// partition the mini batch

			for(int j = 0; j < num_batch; j++)
			{
				count = 0;
				for(int k = j * mini_batch_size; k < (j + 1) * mini_batch_size; k++)
				{
					mini_batch.pairs[count] = training_data.pairs[k];
					count++;
				}
				update_mini_batch(mini_batch, eta,lambda,(int)training_data.sizes);
			}
			printf("Epoch %d training complete\n",i+1);

			if(monitor_training_accuracy)
				printf("Accuracy on training data %d / %d\n",evaluate(training_data),training_data.sizes);


			if(monitor_training_cost)
				printf("Cost on training data %f\n",total_cost(training_data, lambda));


			if(monitor_test_accuracy)
				printf("Accuracy on test data %d / %d\n",evaluate(test_data),test_data.sizes);


			if(monitor_test_cost)
				printf("Cost on test data %f \n",total_cost(test_data, lambda));

			printf("\n");
		}
		cout << "end training" << endl;

	}
	//update mini batch
	void Net::update_mini_batch(const data &mini_batch, double eta, double lambda, int num_data)
	{
		int size = mini_batch.sizes;
		for(int i = 0; i < size; i++)
		{

			Mat<double> &inputs = mini_batch.pairs[i]->input;
			Mat<double> &results = mini_batch.pairs[i]->mat_result;
			backprob(inputs,results);
		}

		for(int i = 0; i < layers.size(); i++)
		{
			Layer *layer = layers[i];
			layer->biases -=  (eta * layer->delta_b / (double)size);
			layer->weights = (1 - eta*lambda/num_data)*layer->weights - (eta * layer->delta_w / (double)size);
			layer->cleanDelta();
		}
	}

	/* update the delta_b and delta_w*/
	void Net::backprob(Mat<double> inputs, Mat<double> results)
	{

		//feedforward
		Layer *layer;
		Layer *lastLayer = layers[layers.size() - 1];
		Mat<double> delta;
		for(int i = 0; i < layers.size(); i++)
		{

			layer = layers[i];
			if(i == 0) layer->z = inputs.dot(layer->weights) + layer->biases;
			else layer->z = layers[i-1]->activation.dot(layer->weights) + layer->biases;
			layer->activation = sigmoid(layer->z);	
		}

		/* get the derivitive of weights and biases for the last later*/
		if(cost_function == CostFunction::QuadraticCost) delta = QuadraticCost::delta(lastLayer->z,lastLayer->activation,results);
		if(cost_function == CostFunction::CrossEntropyCost) delta = CrossEntropyCost::delta(lastLayer->z,lastLayer->activation,results);
		lastLayer->delta_b += delta;
		lastLayer->delta_w += Mat<double>::transpose(layers[layers.size() - 2]->activation).dot(delta);

		// get the derivitive of weights and biases for the hidden layers
		for(int i = layers.size() - 2; i >= 0; i--)
		{
			delta = delta.dot(Mat<double>::transpose(layers[i+1]->weights)) * sigmoid_prime(layers[i]->z);
			layers[i]->delta_b += delta;
			if(i == 0) layers[i]->delta_w +=  Mat<double>::transpose(inputs).dot(delta);
			else layers[i]->delta_w += Mat<double>::transpose(layers[i-1]->activation).dot(delta);
		}

	}
	// feedforward
	Mat<double> Net::feedforward(Mat <double> a)
	{
		Layer *layer;
		for(int i = 0; i < layers.size(); i++)
		{
			layer = layers[i];
			a = sigmoid(a.dot(layer->weights) + layer->biases);
		}
		return a;
	}
	// loading N.N
	Net::Net(string load)
	{
		fstream myfile(load);
		int size;
		string s;
		double d;
		Layer *layer;
		int pointer = 1;
		cout << "loading neural network" << endl;
		if(myfile.is_open())
		{
			while(myfile >> s)
			{
				if(s == "weights")
				{
					layer = new Layer();
					layers.push_back(layer);

					layer->delta_w = Mat<double>::zeros(sizes[pointer-1],sizes[pointer]);
					layer->delta_b = Mat<double>::zeros(1,sizes[pointer]);
					layer->weights = Mat<double>(sizes[pointer-1],sizes[pointer]);
					layer->biases =  Mat<double>(1,sizes[pointer]);
					layer->s = shape(sizes[pointer-1],sizes[pointer]);

					for(int i = 0; i < sizes[pointer-1]; i++)
					{
						for(int j = 0; j < sizes[pointer]; j++)
						{

							if(myfile >> d) layer->weights.set(i,j,d);
							else
							{
								cerr << "loading file failed" << endl;
								exit(1);
							}

						}
					}
				}
				else if(s == "biases")
				{

					for(int i = 0; i < sizes[pointer]; i++)
					{
						if(myfile >> d) layer->biases.set(0,i,d);
						else
						{
							cerr << "loading file failed" << endl;
							exit(1);
						}
					}
					pointer++;
				}
				else if(s == "cost")
				{
					myfile >> s;
					if(s == "CrossEntropyCost") cost_function = CostFunction::CrossEntropyCost;
					else if(s == "QuadraticCost") cost_function = CostFunction::QuadraticCost;
					else
					{
						cerr << "couldn't load cost function" << endl;
						exit(1);
					}
				}
				else sizes.push_back(stoi(s));
			}

		}
		else 
		{
			cout << "cannot open file to load neural netword" << endl;
			exit(1);
		}
		cout << "finish loading" << endl;

	}
	Net::Net(vector <int> sizes, CostFunction cost)
	{
		this->cost_function = cost;
		this->sizes = sizes;
		Layer *layer;
		for(int i = 1; i < sizes.size(); i++)
		{
			layer = new Layer(shape(sizes[i-1],sizes[i]));
			layers.push_back(layer);
		}
	}



	// sigmoid function
	inline Mat<double> sigmoid(Mat<double> mat)
	{	
		return 1.0 / (1.0 + exp(-mat));	
	}


	// the derivate of sigmoid
	inline Mat<double> sigmoid_prime(Mat<double> &mat)
	{
		return sigmoid(mat) * (1.0-sigmoid(mat));
	}

	/* normal distribution generator */
	inline Mat<double> gen_random_number(shape s)
	{
		unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
		std::default_random_engine generator (seed);
		std::normal_distribution<double> distribution (0.0,1.0);
		Mat<double> rv(s.rows, s.cols);
		for (int i = 0; i < s.rows ; i++)
		{
			for(int j = 0; j < s.cols; j++)
			{	
				rv.set(i, j, distribution(generator));
			}
		}
		if(s.cols != 1) rv /= sqrt(s.rows); // minimize the weights and improve the neural network to learn weights
		return rv;
	}
}


