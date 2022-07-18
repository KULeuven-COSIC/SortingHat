/*********************************************************************************
*   Implementation of PDTE plus transciphering using the FINAL library.
*   All the trees are stored as JSON files in the directory named data.
*   
*   First, we run the homomorphic comparisons (using our recursive grouped comparison
* algorithm) to compute the encrypted control bit of each internal node of the
* tree. Then, we run our tree traversal using FINAL::AND gates.
********************************************************************************/


#include "final/FINAL.h"
#include "node.h"
#include "utils.h"
#include <algorithm>
#include <fstream>
#include <iostream>
#include <assert.h>


using namespace std;

void print(const vector<Node*>& u){
	for (unsigned int i = 0; i < u.size(); i++){
		print_node(*(u[i]));
	}
}

// Receives a ciphertext c encrypting m.
// Returns a ciphertext encrypting the result of the comparison m > v
Ctxt_LWE greater_than(const SchemeLWE& fhe, const vector<Ctxt_LWE>& c, const vector<int>& v)
{
    int num_bits = c.size();
    vector<int> not_v(num_bits);
    for(int i = 0; i < num_bits; i++)
        not_v[i] = 1 - v[i];

    Ctxt_LWE res, tmp, x;
    fhe.encrypt(x, 1);
    if (1 == not_v[num_bits-1]){
        res = c[num_bits-1];
    }else{
        fhe.encrypt(res, 0);
    }

    for(int i = num_bits-2; i >= 0; i--){
        // compute tmp = x_{i+1} := v[i+1]*c[i+1] + not(v[i+1]) * not(c[i+1]) \in {0,1}
        if (v[i+1])
            tmp = c[i+1];
        else
            fhe.not_gate(tmp, c[i+1]);
        // update x = x_{num_bits} * x_{num_bits-1} * ... * x_i
        fhe.and_gate(x, x, tmp);
        if (not_v[i]){
            fhe.and_gate(tmp, x, c[i]); // tmp = x * c[i]
            res = (res + tmp);
        }
    }
    return res;
}

Ctxt_LWE less_or_equal(const SchemeLWE& fhe, const vector<Ctxt_LWE>& c, const vector<int>& v)
{
    Ctxt_LWE res = greater_than(fhe, c, v);
    fhe.not_gate(res, res);
    return res;
}

// Assumes that both a and b are bits
const Ctxt_LWE& xnor(const Ctxt_LWE& a, const Ctxt_LWE& not_a, int b){
    if (b)
        return a;
    return not_a;
}



std::map<int, Ctxt_LWE> R1;
std::map<int, Ctxt_LWE> X1;

/**
 * Compare the integer m with all the integers v[0], v[1], ... 
 * Assume that all integers are between 0 and 2^nbits - 1, where nbits = bits_m.size().
 * The result is stored in R1, i.e., after calling this function, 
 * R1[v[i]] stores (m > v[i]).
 * Returns the number of homomorphic gates used in the evaluation.
*/
int grouped_comp(const SchemeLWE& fhe, const vector<Ctxt_LWE>& bits_m, const vector<int>& v)
{
    int nbits = bits_m.size();
    Ctxt_LWE zero;
    fhe.encrypt(zero, 0);
    Ctxt_LWE tmp;
    vector<Ctxt_LWE> bits_not_m(nbits);
    for(int i = 0; i < nbits; i++)
        fhe.not_gate(bits_not_m[i], bits_m[i]);

    R1 = {
            {0, bits_m[nbits-1] },
            {1, zero }
        };
    X1 = {
            {0, bits_not_m[nbits-1] },
            {1, bits_m[nbits-1] }    
        };


    int num_and_gates = 0;
    int k = nbits-2;
    while (k >= 0){
        std::map<int, Ctxt_LWE> R;
        std::map<int, Ctxt_LWE> X;
        for(int i = 0; i < v.size(); i++){
            int vk = v[i] >> k; // nbits-k most significant bits of v[i]
            int vk1 = v[i] >> (k+1); // nbits-k+1 most significant bits of v[i]
            int kth_bit = vk % 2; // k-th bit of v[i]
            if (0 == X.count(vk)){
                fhe.and_gate(X[vk], xnor(bits_m[k], bits_not_m[k], kth_bit), X1[vk1]); // X[vk] = XNOR() * X1[vk1]
                // now compute R[vk] = R1[vk1] + X1[vk1] * (1-kth_bit) * bits_m[k] 
                if (kth_bit){
                    R[vk] = R1[vk1];
                    num_and_gates += 1;
                }else{
                    fhe.and_gate(tmp, X1[vk1], bits_m[k]);
                    R[vk] = tmp + R1[vk1];
                    num_and_gates += 2;
                }
            }
        }
        X1 = X;
        R1 = R;
        k -= 1;
    }
    return num_and_gates;
}


int BOUND_RECURSION = 4;

int rec_split_grouped_comp(const SchemeLWE& fhe, const vector<Ctxt_LWE>& bits_m, const vector<int>& v, bool compX = true)
{

    int nbits = bits_m.size();
    if (nbits <= log(v.size())/log(2))
        return grouped_comp(fhe, bits_m, v);
    
    int k = floor(nbits / 2);
    int two_to_k = 1 << k;
    // XXX TODO use pointers to avoid copying ciphertexts many times
    vector<Ctxt_LWE> lsb_m(k);
    vector<Ctxt_LWE> msb_m(nbits - k);
    for(int i = 0; i < k; i++)
        lsb_m[i] = bits_m[i];
    for(int i = 0; i < nbits-k; i++)
        msb_m[i] = bits_m[k + i];

    vector<int> lsb_v(v.size());
    vector<int> msb_v(v.size());
    for(int i = 0; i < v.size(); i++)
        lsb_v[i] = v[i] % two_to_k;
    for(int i = 0; i < v.size(); i++)
        msb_v[i] = v[i] >> k;

//    for(int i = 0; i < v.size(); i++){
//        assert(v[i] == lsb_v[i] + two_to_k * msb_v[i]);
//    }

    int msb_num_ands = rec_split_grouped_comp(fhe, msb_m, msb_v);
    auto msbR = R1;
    auto msbX = X1;
    int lsb_num_ands = rec_split_grouped_comp(fhe, lsb_m, lsb_v);
    auto lsbR = R1;
    auto lsbX = X1;

    std::map<int, Ctxt_LWE> R;
    std::map<int, Ctxt_LWE> X;
    Ctxt_LWE tmp;

    int num_ands_X = 0;
    for(int i = 0; i < v.size(); i++){
        int vi = v[i];
        int msb_vi = msb_v[i];
        int lsb_vi = lsb_v[i];
        // R[vi] = (msbR[ msb_vi ]) OR (msbX[ msb_vi ] and lsbR[ lsb_vi ])
        // but we can actually add instead of using OR gate because only one
        // of the clauses can be true
        if (0 == X.count(vi)){
            // R[vi] = msbR[ msb_vi ] + msbX[ msb_vi ] and lsbR[ lsb_vi ]
            fhe.and_gate(tmp, msbX[ msb_vi ], lsbR[ lsb_vi ]);
            R[vi] = msbR[ msb_vi ] + tmp;
            num_ands_X += 1;
            if(compX){
                //X[vi] = msbX[ msb_vi ] and lsbX[ lsb_vi ]
                fhe.and_gate(X[vi], msbX[ msb_vi ], lsbX[ lsb_vi ]);
                num_ands_X += 1;
            }
        }
    }
    int total_gates = msb_num_ands + lsb_num_ands + num_ands_X;
    R1 = R;
    X1 = X;
    return total_gates;
}


int bits_to_int(const vector<int>& bits)
{
    int pow2 = 1;
    int res = 0;
    for (int i = 0; i < bits.size(); i++){
        res += bits[i] * pow2;
        pow2 *= 2;
    }
    return res;
}

vector<int> int_to_bits(int m, int num_bits)
{
    vector<int> bits(num_bits);
    for(int i = 0; i < num_bits; i++){
        bits[i] = m % 2;
        m /= 2;
    }
    return bits;
}


void print_R1(){
    for (auto it = R1.begin(); it != R1.end(); ++it)
           std::cout << it->first << " => " << it->second.b << endl;
}

void test_comparisons() {
    int num_bits = 10; // bit length of the values that will be compared
    int upper_bound = 1 << num_bits; // maximum value
    int n = 50;    // number of plaintexts that will be compared to the ciphertext
        
    SchemeLWE fhe;

    int N_TESTS = 5;
    double avg_time = 0.0;
    double avg_gates = 0.0;
    for(int _i = 0; _i < N_TESTS; _i++){
        cout << "# test " << _i << endl;
        // we want to compare enc(m) with v
        int m = rand() % upper_bound;
        vector<int> bits_m(num_bits); // m[0] is the least significant bit of m
        bits_m = int_to_bits(m, num_bits);
        vector<Ctxt_LWE> c(num_bits);
        for(int i = 0; i < num_bits; i++){
            fhe.encrypt(c[i], bits_m[i]);
        }
//        int m = bits_to_int(bits_m);
//        int v = bits_to_int(bits_v);
//        cout << "bits_m = " << bits_m << endl;
//        cout << "m = " << m << endl;


        vector<int> v(n); // values that will be compared to m
        for(int i = 0; i < n; i++){
            v[i] = rand() % upper_bound;
        }
//        sort(v.begin(), v.end());
//        cout << "v = " << v << endl;

        auto start = clock();
//        int num_and_gates = grouped_comp(fhe, c, v);
        cout << "int num_and_gates = rec_split_grouped_comp(fhe, c, v);" << endl;
        int num_and_gates = rec_split_grouped_comp(fhe, c, v);

        avg_time += float(clock()-start)/CLOCKS_PER_SEC;

        avg_gates += num_and_gates;

        for (int i = 0; i < v.size(); i++){
            const Ctxt_LWE& res = R1[v[i]];
            int dec_comp = fhe.decrypt(res);

//            cout << "v[" << i << "] = " << v[i] << endl;
//            cout << "R1[v[i]].count() = " << R1.count(v[i]) << endl;
//            cout << "dec_comp = " << dec_comp << endl;
//            cout << endl;
            assert(dec_comp == (m > v[i]));
        }

    }

    cout << "Homomorphic comparison of m with " << n 
         << " integers of " << num_bits << " bits." << endl;
    cout << "Avg. time to compare homomorphically: " 
         << avg_time/N_TESTS << " s" << endl;
    cout << "Avg. number of homomorphic gates: " << avg_gates / N_TESTS << endl;
}

// Assume that the control bit of each internal node is already set
void traverse_rec(Ctxt_LWE& out, Node& node, const SchemeLWE& fhe){
    Ctxt_LWE& parent = node.value;
    if (node.is_leaf()) {
        if (node.class_leaf == 1)
//            out += node.class_leaf * parent;
//            out += parent;
            out = (out + parent);
    }else{
        // right.value = parent * node.control_bit
        fhe.and_gate(node.right->value, parent, node.control_bit);
        // left.value = parent*(1 - node.control_bit)
        node.left->value = parent - node.right->value;
        traverse_rec(out, *(node.left), fhe);
        traverse_rec(out, *(node.right), fhe);
    }

}

// num_bits is the bit length of the values that will be compared
void test_json_tree(string filename, int num_bits, bool verbose) {
    int upper_bound = 1 << num_bits; // maximum value
        
    SchemeLWE fhe;

//    std::ifstream ifs("data/phoneme_10bits/model.json");
    std::ifstream ifs(filename);
    json j = json::parse(ifs);
    auto root = Node(j);
    if (verbose)
        print_tree(root);

    vector< vector<Node*> > nodes_by_feat = nodes_by_feature(root);

    int num_features = nodes_by_feat.size();
    std::vector< std::vector< int > > thrs_by_feat = thresholds_by_feature(nodes_by_feat);

    int N_TESTS = 3;
    double avg_time = 0.0;
    for(int _i = 0; _i < N_TESTS; _i++){
        cout << "# test " << _i << endl;
        // we want to compare enc(m) with v
        vector<int> plain_features(num_features);
        vector<vector<int> > bits_m(num_features);
        vector<vector<Ctxt_LWE> > enc_bits(num_features); // enc_bits[i] == encryption of bits of i-th feature
        for(int _j = 0; _j < num_features; _j++){
            plain_features[_j] = rand() % upper_bound;
            bits_m[_j] = vector<int>(num_bits); // m[0] is the least significant bit of m
            bits_m[_j] = int_to_bits(plain_features[_j], num_bits);
            vector<Ctxt_LWE> c(num_bits);
            for(int i = 0; i < num_bits; i++){
                fhe.encrypt(c[i], bits_m[_j][i]);
            }
            enc_bits[_j] = c;
        }

        // Now, compare each feature
        for (int i = 0; i < num_features; i++){
            int m = plain_features[i];
            vector<int>& v = thrs_by_feat[i];
            int n = v.size();    // number of plaintexts that will be compared to the ciphertext
            vector<Ctxt_LWE>& c = enc_bits[i];

            if (0 == n)
                continue;
            auto start = clock();
            float run_time;
            vector<Node*>& nodes = nodes_by_feat[i];
            if (n == 1){
                Ctxt_LWE enc_cmp = greater_than(fhe, c, int_to_bits(v[0], num_bits));
                nodes[0]->control_bit = enc_cmp;
                run_time = float(clock()-start)/CLOCKS_PER_SEC;
            }else{
                int num_and_gates = rec_split_grouped_comp(fhe, c, v);
                run_time = float(clock()-start)/CLOCKS_PER_SEC;
                avg_time += run_time;

                for (int j = 0; j < v.size(); j++){
                    const Ctxt_LWE& res = R1[v[j]];
                    int dec_comp = fhe.decrypt(res);

                    assert(dec_comp == (m > v[j]));
                }

                // now update the control bits with the result of the comparisons
                vector<Node*>& nodes = nodes_by_feat[i];
                for(int j = 0; j < n; j++){
                    nodes[j]->control_bit = R1[v[j]];
                }
            }

            cout << "Homomorphic comparison of feature["<<i<<"] with " << n    
            << " integers of " << num_bits << " bits." << endl;
            cout << "run_time = " << run_time << endl;


        }
        // Now that every feature was compared to the corresponding nodes, we traverse the tree
        Ctxt_LWE out;
        auto start = clock();
        traverse_rec(out, root, fhe);
        float run_time = float(clock()-start)/CLOCKS_PER_SEC;
        cout << "Time to traverse homomorphically: " << run_time << endl;
    }
}


int main()
{
    srand(time(NULL));

    bool verbose = false;

//    test_json_tree("data/phoneme_10bits/model.json", 10, verbose);
    test_json_tree("data/electricity_10bits/model.json", 10, verbose);
//    test_json_tree("data/fake_art_11bits/model.json", 11, verbose);
//    test_json_tree("data/fake_art_16bits/model.json", 16, verbose);
//    test_json_tree("data/fake_hou_11bits/model.json", 11, verbose);
//    test_json_tree("data/fake_hou_16bits/model.json", 16, verbose);
//    test_json_tree("data/spam_11bits/model.json", 11, verbose);

    return 0;
}
