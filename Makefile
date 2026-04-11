all: $(wildcard *.cpp *.hpp) 
	g++ main.cpp -o nn 
