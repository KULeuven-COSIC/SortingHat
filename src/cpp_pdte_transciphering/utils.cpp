#include "utils.h"

using namespace std;


void swap(vector<int>& v, int i, int j)
{
    int tmp = v[i];
    v[i] = v[j];
    v[j] = tmp;
}

void shuffle(std::vector<int>& v,
              std::default_random_engine& rand_engine,
              std::uniform_int_distribution<int>& uni_sampler)
{
    for(int i = 0; i < v.size() / 2; i++){
        int j = uni_sampler(rand_engine) % v.size();
        swap(v, i, j);
    }
}


