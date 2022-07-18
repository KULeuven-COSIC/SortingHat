#include "filip.h"
#include "utils.h"
#include "final/FINAL.h"
//#include "vectorutils.hpp"

using namespace std;

FiLIP::FiLIP(int len_sk, int size_subset, int size_domain_thr, int threshold_limit)
    : len_sk(len_sk), size_subset(size_subset), size_domain_thr(size_domain_thr),
        threshold_limit(threshold_limit), 
        num_bits_xored(size_subset - size_domain_thr),
        uni_sampler(0, len_sk)
{

    sk = vector<int>(len_sk);
    Sampler::get_binary_vector(sk);

    whitening = vector<int>(size_subset);
}

vector<int> FiLIP::subset_permut_whiten()
{
    vector<int> v(sk); // copy secret key
    vector<int> res(size_subset);

    // sample whitening mask
    for(int i = 0; i < size_subset; i++)
        whitening[i] = uni_sampler(rand_engine) % 2;

    // shuffle
    shuffle(v, rand_engine, uni_sampler);

    // take subset and apply whitening
    for(int i = 0; i < res.size(); i++){
        res[i] = v[i] ^ whitening[i];
    }
     
    return res;
}

int FiLIP::compute_xor_thr(vector<int> perm_subset_sk)
{
    int _xor = 0;
    int i;
    // xor the first bits of perm_subset_sk
    for(i = 0; i < num_bits_xored; i++)
        _xor = (_xor + perm_subset_sk[i]) % 2;

    // sum the last bits of perm_subset_sk
    int sum = 0;
    for(; i < size_subset; i++)
        sum += perm_subset_sk[i];
    int T_d_n = (sum < threshold_limit ? 0 : 1);
    return _xor ^ T_d_n; // XOR(x_1, ..., x_k) xor T_{d,n}(y_1, ..., y_n) 
}

int FiLIP::enc_bit(int b)
{

    assert(0 == b || 1 == b);

    vector<int> permuted_subset = subset_permut_whiten();

    int f_x_y = compute_xor_thr(permuted_subset);

    return (b + f_x_y) % 2;
}

/**
*  Uses the initialization vector iv to encrypt each entry of msg, 
* which is supposed to be a binary vector.
*/
std::vector<int> FiLIP::enc(long int iv, std::vector<int> msg)
{
    rand_engine = default_random_engine(iv);// reset the seed using the given iv

    vector<int> ctxt(msg.size());

    for(int i = 0; i < msg.size(); i++)
        ctxt[i] = enc_bit(msg[i]);
    return ctxt;
}

std::vector<int> FiLIP::dec(long int iv, std::vector<int> c)
{
    return enc(iv, c); // decryption and encryption are actually the same
}

std::ostream& operator<<(std::ostream& os, const FiLIP& u){
    os << "FiLIP: {" 
        << "len_sk: " << u.len_sk
        << ", size_subset: " << u.size_subset
        << ", size_domain_threshold: " << u.size_domain_thr
        << ", threshold_limit: " << u.threshold_limit 
        << "}";
        return os;
}

