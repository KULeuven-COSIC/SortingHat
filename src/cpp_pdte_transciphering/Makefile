CCX = g++
CCXFLAGS = -O3 -funroll-loops -march=native -std=c++17 -pthread -I. -I.final/include
DEPS = -lntl -lgmp -lfftw3 -lm

all: clean test_filip test_node test_homomorphic_filip test_pdte_transciphering
 
clean:
	$(RM) final/test test test_node test_filip test_homomorphic_filip test_pdte_transciphering *.o

test_pdte_transciphering: test_pdte_transciphering.cpp libfinal.a
	$(CCX) $(CCXFLAGS) -o test_pdte_transciphering test_pdte_transciphering.cpp node.o ./final/libfinal.a $(DEPS)

libfinal.a: 
	$(MAKE) -C ./final

test_filip: filip.o test_filip.cpp libfinal.a
	$(CCX) $(CCXFLAGS) -o test_filip test_filip.cpp filip.o utils.o ./final/libfinal.a $(DEPS)

filip.o: filip.h filip.cpp utils.o
	$(CCX) $(CCXFLAGS) -c filip.cpp -o filip.o

utils.o: utils.h utils.cpp
	$(CCX) $(CCXFLAGS) -c utils.cpp -o utils.o

node.o: node.h node.cpp
	$(CCX) $(CCXFLAGS) -c node.cpp -o node.o

test_node: node.o test_node.cpp libfinal.a
	$(CCX) $(CCXFLAGS) node.o test_node.cpp -o test_node ./final/libfinal.a $(DEPS)

test_homomorphic_filip: filip.o utils.o homomorphic_filip.o test_homomorphic_filip.cpp libfinal.a
	$(CCX) $(CCXFLAGS) -o test_homomorphic_filip test_homomorphic_filip.cpp homomorphic_filip.o filip.o utils.o ./final/libfinal.a $(DEPS)

homomorphic_filip.o: filip.o utils.o homomorphic_filip.h homomorphic_filip.cpp
	$(CCX) $(CCXFLAGS) -c homomorphic_filip.cpp -o homomorphic_filip.o


