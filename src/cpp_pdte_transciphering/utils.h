#ifndef __MY_UTILS_FUNCS__
#define __MY_UTILS_FUNCS__

#include <vector>
#include <iostream> 
#include <random>

void swap(std::vector<int>& v, int i, int j);

void shuffle(std::vector<int>& v,
              std::default_random_engine& rand_engine,
              std::uniform_int_distribution<int>& uni_sampler);


template <typename T>
std::ostream& operator<< (std::ostream &out, const std::vector<T> & u) {
    if (0 == u.size())
        return out << "[ ]";
        
    std::cout << "[";
    for (long i = 0; i < u.size()-1; i++)
        out << u[i] << ", ";
    out << u[u.size()-1] << "]";
    return out;
}

#endif
