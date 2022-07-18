#include "filip.h"
#include "utils.h"
#include "homomorphic_filip.h"
#include "final/FINAL.h"

using namespace std;

HomomorphicFiLIP::HomomorphicFiLIP(const FiLIP& f)
    : len_sk(f.len_sk), size_subset(f.size_subset), size_domain_thr(f.size_domain_thr),
        threshold_limit(f.threshold_limit), 
        num_bits_xored(f.size_subset - f.size_domain_thr),
        uni_sampler(0, f.len_sk)
{
    log_B_ngs = 7; 
    B_ngs = 1 << log_B_ngs;
    l_ngs = ceil(log(q_boot) / log(B_ngs));

    auto start = clock();
    encrypt_sk(f.sk);
    cout << "Time to encrypt FiLIP's sk with FHE: " << float(clock()-start)/CLOCKS_PER_SEC << endl;

    // initialize vectors that will store the permuted and masked subsets used 
    // during homomorphic evaluation of FiLIP::decryption
    subset_enc_sk = std::vector<Ctxt_LWE>(size_subset);
    subset_enc_X_to_sk = std::vector<NGSFFTctxt>(size_subset);

    whitening = vector<int>(size_subset);
}

void HomomorphicFiLIP::encrypt_sk(const vector<int>& sk){
    int q = parLWE.q_base;
    // encrypt each bit of sk into an LWE ciphertext with Delta = q/2,
    // that is, (a, b) \in Z_q^(n+1) where
    // b = a*s + e + (q/2) * sk[i]
    enc_sk = std::vector<Ctxt_LWE>(len_sk);
    for(int i = 0; i < len_sk; i++){
        fhe.encrypt(enc_sk[i], 0);
        enc_sk[i].b += (q/2) * sk[i];
        parLWE.mod_q_base(enc_sk[i].b);
    }

    // precompute LWE::enc(NOT(sk[i]))
    enc_not_sk = std::vector<Ctxt_LWE>(len_sk);
    for(int i = 0; i < len_sk; i++){
        enc_not_sk[i].a = enc_sk[i].a;
        enc_not_sk[i].b = enc_sk[i].b + (q/2);
        parLWE.mod_q_base(enc_not_sk[i].b);
    }
    
    enc_X_to_sk = std::vector<NGSFFTctxt>(len_sk);
    enc_X_to_not_sk = std::vector<NGSFFTctxt>(len_sk);
    for(int i = 0; i < len_sk; i++){
        ModQPoly msg(Param::N, 0L);
        ModQPoly not_msg(Param::N, 0L);
        if(1 == sk[i]){
            msg[1] = 1; // msg = X^sk[i]
            not_msg[0] = 1; // not_msg = X^(not sk[i])
        }else{
            msg[0] = 1; // msg = X^sk[i]
            not_msg[1] = 1; // not_msg = X^(not sk[i])
        }
        enc_ngs(enc_X_to_sk[i], msg, l_ngs, B_ngs, fhe.sk_boot);
        enc_ngs(enc_X_to_not_sk[i], not_msg, l_ngs, B_ngs, fhe.sk_boot);
    }
}

void HomomorphicFiLIP::subset_permut_whiten()
{
    // sample whitening mask
    for(int i = 0; i < size_subset; i++)
        whitening[i] = uni_sampler(rand_engine) % 2;

    vector<int> indexes(len_sk);
    for(int i = 0; i < len_sk; i++)
        indexes[i] = i;
    shuffle(indexes, rand_engine, uni_sampler);

    // now, indexes represents the random permutation, namely, if indexes[i] = j,
    // then the i-th element of the permuted subset is the j-th element of the
    // original set

    // copy permuted subset of encryptions of sk and apply whitening
    int q = parLWE.q_base;
    for(int i = 0; i < size_subset; i++){
        int j = indexes[i];
        int wi = whitening[i];

        // copy LWE.enc( XOR(sk[j], wi) ) and NGS.enc( X^XOR(sk[j], wi) )
        if (0 == wi){
            subset_enc_sk[i] = enc_sk[j];
            subset_enc_X_to_sk[i] = enc_X_to_sk[j];
        }else{
            subset_enc_sk[i] = enc_not_sk[j];
            subset_enc_X_to_sk[i] = enc_X_to_not_sk[j];
        }
    }
}

ModQPoly get_test_vector_for_threshold(int N, int d, int Q)
{
    ModQPoly t(N); // t(X) = sum_{i=0}^{d-1} 0 * X^{2N - i}+sum_{i=d}^{N-1} 1 * X^{2N - i}
                   //      = sum_{i=0}^{d-1} 0 * X^{N - i} - sum_{i=d}^{N-1} X^{N - i}
                   //      = - X^1 - X^2 - ... - X^{N-d}
    t[0] = 0;
    for(int i = N-d+1; i < N; i++)
        t[i] = 0;
    for(int i = 1; i <= N-d; i++)
        t[i] = -(Q/2); // we are actually defining (Q/2) * t(X) instead of t(X)

    return t;
}

Ctxt_LWE HomomorphicFiLIP::compute_xor_thr()
{
    Ctxt_LWE _xor = subset_enc_sk[0]; // copying encryption of first permuted bit
    int i;
    // xor the first bits of permuted subset of sk
    for(i = 1; i < num_bits_xored; i++)
        _xor = (_xor + subset_enc_sk[i]); // XXX: Implement += in FINAL and replace this line

    // sum the last bits of permuted subset of sk
    int N = Param::N;
    int Q = q_boot; // Q used by the NGS scheme (accumulator)

    ModQPoly acc = get_test_vector_for_threshold(N, threshold_limit, Q);


    vector<long> tmp_acc_long(N);
    for(; i < size_subset; i++){
        // tmp_acc_long = decompose(acc) * subset_enc_X_to_sk[i]
        external_product(tmp_acc_long, acc, subset_enc_X_to_sk[i], B_ngs, log_B_ngs, l_ngs);
        // acc = tmp_acc_long mod Q
        mod_q_boot(acc, tmp_acc_long);
    }
    // Now acc encrypts t(X) * X^(sk[num_bits_xored] + ... + sk[size_subset-1])
    // where t(X) is the test polynomial corresponding to the threshold function.

    // mod switch from Q to q_base
    modulo_switch_to_base_lwe(acc);
    
    // key switch
    Ctxt_LWE ct;
    fhe.key_switch(ct, acc);
    // now ct is an LWE ciphertext encrypting the threshold T_{d, n}(y_1, ..., y_n)
    // with delta equals to q/2.

    return _xor + ct; // enc( XOR(x_1, ..., x_k) xor T_{d,n}(y_1, ..., y_n) )
}

Ctxt_LWE HomomorphicFiLIP::enc_bit(int ci)
{
    assert(0 == ci || 1 == ci);

    this->subset_permut_whiten();

    // compute encryption of f(x, y) = XOR(x) + THR(y) % 2
    Ctxt_LWE f_x_y = compute_xor_thr();
    
    // XOR it with the encrypted bit c_i to get an LWE encryption of m_i
    int q = parLWE.q_base;
    f_x_y.b += ci * (q/2);
    parLWE.mod_q_base(f_x_y.b);

    return f_x_y;
}

std::vector<Ctxt_LWE> HomomorphicFiLIP::transform(long int iv, const std::vector<int>& c)
{
    rand_engine = default_random_engine(iv);// reset the seed using the given iv

    vector<Ctxt_LWE> ctxt(c.size());

    for(int i = 0; i < c.size(); i++){
        ctxt[i] = enc_bit(c[i]);
    }
    return ctxt;
}

std::vector<int> HomomorphicFiLIP::dec(std::vector<Ctxt_LWE> c)
{
    vector<int> m(c.size()); 
    for(int i = 0; i < m.size(); i++)
        // We cannot simply return SchemeLWE::decrypt(c[i]) because SchemeLWE 
        // assumes that the encrypted message is multiplied by q/4,
        // but in our case, it is multiplied by q/2, so, in SchemeLWE::decrypt,
        // when we multiply by 4/q, we get (4/q) * (q/2) * m = 2*m instead of m.
        // Thus, we fix it here by mapping 2*m two a binary message again.
        m[i] = (fhe.decrypt(c[i]) == 0 ? 0 : 1);
    return m;
}

double HomomorphicFiLIP::noise(const std::vector<Ctxt_LWE>& c, const std::vector<int>& m)
{
    int q = parLWE.q_base;
    double noise = 0;
    for(int i = 0; i < c.size(); i++){
        Ctxt_LWE ct = c[i];
        ct.b -= (q/2) * m[i];
        ct.b = parLWE.mod_q_base(ct.b); // now ct encrypts zero
                                        // this is necessary because fhe.noise 
                                        // expects that m is multipliplied by 
                                        // q/4, but here c uses q/2
        double noise_i = fhe.noise(ct, 0);
        if (noise_i > noise)
            noise = noise_i;
    }
    return noise;
}


std::ostream& operator<<(std::ostream& os, const HomomorphicFiLIP& u){
    os << "HomomorphicFiLIP: {" 
        << "len_sk: " << u.len_sk
        << ", size_subset: " << u.size_subset
        << ", size_domain_threshold: " << u.size_domain_thr
        << ", threshold_limit: " << u.threshold_limit 
        << ", FINAL: {"
        << " l_ngs: " << u.l_ngs
        << ", log_B_ngs: " << u.log_B_ngs
        << ", N: " << parLWE.N
        << ", logQ: " << ceil(log(q_boot) / log(2))
        << ", n: " << parLWE.n
        << ", logq: " << ceil(log(parLWE.q_base) / log(2))
        << "} }";
        return os;
}

