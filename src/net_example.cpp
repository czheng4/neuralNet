/*
	2019 chaohui zheng
	A easy example of using N.N on handwriting digits recognition.
*/
#include <iostream>
#include <string>
#include <vector>
#include <fstream>
#include <chrono>
#include <random>
#include <cmath>
#include "mat.h"
#include "net.h"
#include <cstdlib>
#include <cstdio>
using namespace zzh;
using namespace std;



int main()
{
	
	vector <int> v;
	v.push_back(784);
	v.push_back(20);
	v.push_back(10);
	v.push_back(10);
	
	
	Net net(v);
	data train("mnist_inputs.txt",shape(10000,784),"mnist_results.txt",10);
	net.SGD(train,10,30,0.5,4, true,true);
	net.save("n1.txt");
}
