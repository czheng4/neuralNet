# neuralNet
The neuralNet was implemented in c++. It allows you to train feedforward neural network. neuralNet is composed of two parts:
1. Matrix: This is in `include/mat.h`. It was implemented by using c++ template. It's a pare-down version of numpy
2. Net: This is in `src/net.cpp`. The backpropagation algorithm is from http://neuralnetworksanddeeplearning.com/chap1.html.


## usage
The `bin/net_example` loads dataset "mnist_inputs.txt"(x) and "mnist_results.txt"(y) and trains for 10 epochs. You can find the MNIST dataset at [here](http://yann.lecun.com/exdb/mnist/).
```
UNIX> pwd
/yourpath/neuralNet
UNIX> make
g++ --std=c++11 -O3 -c -o obj/matrix_example.o src/matrix_example.cpp -I./include/
g++ --std=c++11 -O3 -o bin/matrix_example obj/matrix_example.o -I./include/
g++ --std=c++11 -O3 -c -o obj/net_example.o src/net_example.cpp -I./include/
g++ --std=c++11 -O3 -c -o obj/net.o src/net.cpp -I./include/
g++ --std=c++11 -O3 -o bin/net_example obj/net_example.o obj/net.o -I./include/
UNIX> bin/net_example 
```

