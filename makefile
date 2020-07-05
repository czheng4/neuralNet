

all: bin/matrix_example bin/net_example

INCLUDE = -I./include/
FLAGS = --std=c++11 -O3

obj/matrix_example.o: src/matrix_example.cpp
	g++ $(FLAGS) -c -o $@ $< $(INCLUDE)

obj/net_example.o: src/net_example.cpp
	g++ $(FLAGS) -c -o $@ $< $(INCLUDE)

obj/net.o: src/net.cpp
	g++ $(FLAGS) -c -o $@ $< $(INCLUDE)

bin/matrix_example:  obj/matrix_example.o 
	g++ $(FLAGS) -o $@ $< $(INCLUDE)

bin/net_example: obj/net_example.o obj/net.o
	g++ $(FLAGS) -o $@ $^ $(INCLUDE)

clean: 
	rm obj/* bin/*



