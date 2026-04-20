all: $(wildcard *.cpp *.hpp) 
	g++ -g -Wall -Wextra main.cpp -o nn 
