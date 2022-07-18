/**
 *  Implements a homomorphic version of the FiLIP stream cipher 
 * with f = XOR-THR, which described in the paper
 * 'Transciphering, using FiLIP and TFHE for an efficient delegation of computation' 
 * (https://eprint.iacr.org/2020/1373.pdf).
 */

#ifndef __HOM_FiLIP_XOT_THR__
#define __HOM_FiLIP_XOT_THR__

#include "filip.h"

#include "final/FINAL.h"

class HomomorphicFiLIP
{
    public:

        SchemeLWE fhe;

        std::vector<Ctxt_LWE> enc_sk;
        std::vector<Ctxt_LWE> enc_not_sk;
        std::vector<NGSFFTctxt> enc_X_to_sk;
        std::vector<NGSFFTctxt> enc_X_to_not_sk;

        std::vector<Ctxt_LWE> subset_enc_sk;
        std::vector<NGSFFTctxt> subset_enc_X_to_sk;

        int len_sk; // number of bits of secret key
        int size_subset; // number of bits of sk used to encrypt each bit

        int size_domain_thr; // number of bits added in the threshold function
        int threshold_limit; // added bits are compared to this limit

        int num_bits_xored; // number of bits that are xored with the threshold.
                            // We have: size_subset = size_domain_thr + num_bits_xored

        int B_ngs; // decomposition base used in the vector ciphertexts of NGS. 
        int l_ngs; // dimension of vector ciphertexts of NGS. l_ngs = log_{B_ngs}(Q)
        int log_B_ngs; // log(B_ngs) in base 2


        std::default_random_engine rand_engine;
        std::uniform_int_distribution<int> uni_sampler;

        std::vector<int> whitening; // random bits used to mask permuted subset

        HomomorphicFiLIP(const FiLIP& filip);

        /**
         *  Encrypts the secret key of the FiLIP stream cipher into FHE ciphertexts.
         */
        void encrypt_sk(const vector<int>& sk);


        /**
         *  Select a random subset of the encrypted bits of the secret key
         * and apply a whitening mask to it.
         *  Then, update the variables subset_enc_sk and subset_enc_X_to_sk
         * to store pointers this permuted and masked subset.
         */
        void subset_permut_whiten();


        /**
         *  Receives the initialization vector iv that was previously used by 
         * FiLIP to encrypt a binary vector m into the ciphertext c under a
         * secret key sk.
         *  Uses iv and the encryption of sk to homomorphically evaluate the 
         * decryption function of FiLIP on c, generating a vector of LWE 
         * ciphertexts encrypting m.
         */
        std::vector<Ctxt_LWE> transform(long int iv, const std::vector<int>& c);


        /**
         *  Receives a vector of LWE ciphertexts and decrypt them.
         */
        std::vector<int> dec(std::vector<Ctxt_LWE> c);

        /**
         *  Receives a vector of LWE ciphertexts and the messages they are supposed
         * to encrypt. Computes the log of noise of each of ciphertext 
         * and returns the maximum of these logs.
         */
        double noise(const std::vector<Ctxt_LWE>& c, const std::vector<int>& m);


        /**
         *  Auxiliar function used to encrypt and decrypt. 
         *  It receives a permutation of a subset of the secret key,
         * denoted by x_1, x_2, ..., x_(n-k), y_1, ..., y_k, and outputs
         *      x_1 XOR ... XOR x_(n-k) XOR THR(y_1, ..., y_k)
         * where THR(y_1, ..., y_k) is 0 if sum y_i < threshold_limit and 1 otherwise.
         */
        Ctxt_LWE compute_xor_thr();


        /**
         *  Apply the FiLIP encryption homomorphically to the bit ci given as input.
         *  This function is used by the HomomorphicFiLIP::transform.
         */
        Ctxt_LWE enc_bit(int ci);
};

std::ostream& operator<<(std::ostream& os, const HomomorphicFiLIP& u);

#endif
