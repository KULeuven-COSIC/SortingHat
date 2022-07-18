#include "filip.h"
#include <cassert>
#include <iostream>
//#include "vectorutils.hpp"

using namespace std;


template <typename ELEMENT>
std::ostream& operator<<(std::ostream& os, const vector<ELEMENT>& u){
	unsigned int lastPosition = u.size() - 1;
	for (unsigned int i = 0; i < lastPosition; i++){
		os << u[i] << ", ";
	}
	os << u[lastPosition];
	return os;
}

int main()
{

    int N = 1 << 10; // length of the secret key
    int n = 144;    // size of subset used to encrypt each bit
    int k = 63;     // number of bits added in the threshold function
    int d = 35;          // threshold limit
    FiLIP filip(N, n, k, d);

    cout << filip << endl;

    long int iv = 1234567;

    // create a random message
    srand(time(NULL));
    int size_msg = 8;
    vector<int> m(size_msg);
    for(int i = 0; i < size_msg; i++)
        m[i] = rand() % 2; 

    vector<int> c = filip.enc(iv, m);
    vector<int> dec_m = filip.dec(iv, c);

    cout << "    m  = " << m << endl;
    cout << "    c  = " << c << endl;
    cout << "dec(c) = " << dec_m << endl;

    assert(m == dec_m);

    return 0;
}
