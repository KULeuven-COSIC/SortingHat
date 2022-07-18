/**
 *  Implements the FiLIP stream cipher with f = XOR-THR, as described
 * in the paper 'Transciphering, using FiLIP and TFHE for an efficient
 * delegation of computation' (https://eprint.iacr.org/2020/1373.pdf).
 */

#ifndef __FiLIP_XOT_THR__
#define __FiLIP_XOT_THR__

#include <vector>
#include <random>

class FiLIP
{
    public:

        int len_sk; // number of bits of secret key
        std::vector<int> sk; // secret key

        int size_subset; // number of bits of sk used to encrypt each bit

        int size_domain_thr; // number of bits added in the threshold function
        int threshold_limit; // added bits are compared to this limit

        int num_bits_xored; // number of bits that are xored with the threshold.
                            // We have: size_subset = size_domain_thr + num_bits_xored

        std::vector<int> whitening; // random bits used to mask permuted subset


        std::default_random_engine rand_engine;
        std::uniform_int_distribution<int> uni_sampler;


        FiLIP(int len_sk, int size_subset, int size_domain_thr, int threshold_limit);

        /** 
         *  Permutes the secret key, then takes a subset of it and XOR with the
         * whitening vector. Returns the result in a vector of length size_subset.
         */
        std::vector<int> subset_permut_whiten();


        /**
         *  Uses the initialization vector iv to encrypt each entry of msg, 
         * which is supposed to be a binary vector.
         */
        std::vector<int> enc(long int iv, std::vector<int> msg);


        /**
         *  Uses the initialization vector iv to decrypt each entry of c, 
         * which is supposed to be a binary vector.
         */
        std::vector<int> dec(long int iv, std::vector<int> c);


        /**
         *  Auxiliar function used to encrypt and decrypt. 
         *  It receives a permutation of a subset of the secret key,
         * denoted by x_1, x_2, ..., x_(n-k), y_1, ..., y_k, and outputs
         *      x_1 XOR ... XOR x_(n-k) XOR THR(y_1, ..., y_k)
         * where THR(y_1, ..., y_k) is 0 if sum y_i < threshold_limit and 1 otherwise.
         */
        int compute_xor_thr(std::vector<int> perm_subset_sk);


        int enc_bit(int b);
};


std::ostream& operator<<(std::ostream& os, const FiLIP& u);

#endif
