#include "node.h"
#include <iostream>

using namespace std;

void print_node(Node& a, string tabs){
    if (a.is_leaf()){
       cout << tabs << "(class: " << a.class_leaf << ")" << endl; 
    }else{
        cout << tabs
            << "(f: " << a.feature_index 
            << ", t: " << a.threshold
            << ")" << endl; 
    }
}

void print(std::shared_ptr<Node> a, string tabs) {
    if (a->is_leaf()){
       print_node(*a, tabs);
    }else{
        print(a->right, tabs + "        ");
        print_node(*a, tabs);
        print(a->left, tabs + "        ");
    }
}

void print_tree(Node& a) {
    std::shared_ptr<Node> tmp_a = std::make_shared<Node>(a);
    print(tmp_a, " ");
}

void build_tree_from_json(json::reference ref, Node& node) {
    if (!ref["leaf"].is_null()) {
        node.class_leaf = ref["leaf"].get<Leaf>();
        node.left = nullptr;
        node.right = nullptr;
    } else {
        node.threshold = ref["internal"]["threshold"].get<unsigned>();
        node.feature_index = ref["internal"]["feature"].get<unsigned>();
        node.op = ref["internal"]["op"].get<std::string>();
        node.class_leaf = -1;

        node.left = std::make_shared<Node>();
        node.right = std::make_shared<Node>();
        build_tree_from_json(ref["internal"]["left"], *node.left);
        build_tree_from_json(ref["internal"]["right"], *node.right);
    }
}

Node::Node(json::reference j) {
    build_tree_from_json(j, *this);
}

bool Node::is_leaf() const {
    return this->left == nullptr && this->right == nullptr;
}

void Node::gen_with_depth(int d) {
    if (d == 0) {
        this->class_leaf = 0;
        this->left = nullptr;
        this->right = nullptr;
        return;
    }
    this->class_leaf = -1;
    (*this->left).gen_with_depth(d-1);
    (*this->right).gen_with_depth(d-1);
}

unsigned Node::get_depth() {
    if (this->is_leaf()) {
        return 0;
    } else {
        auto l = this->left->get_depth();
        auto r = this->right->get_depth();
        if (l > r) {
            return l + 1;
        } else {
            return r + 1;
        }
    }
}

void eval_rec(unsigned &out, const Node& node, const std::vector<unsigned int> &features, unsigned parent) {
    if (node.is_leaf()) {
        out += node.class_leaf * parent;
    }else{
        if (node.op == "leq") {
            if (features[node.feature_index] <= node.threshold) {
                eval_rec(out, *node.left, features, parent);
                eval_rec(out, *node.right, features, parent*(1-parent));
            } else {
                eval_rec(out, *node.left, features, parent*(1-parent));
                eval_rec(out, *node.right, features, parent);
            }
        } else {
            // unimplemented
            assert(false);
        }
    }
}

unsigned Node::eval(const std::vector<unsigned int> &features) {
    unsigned out = 0;
    unsigned parent = 1;
    eval_rec(out, *this, features, parent);
    return out;
}




void filter_by_feature_index(Node& a, int feature_index, vector<Node*>& vec){
    if (!a.is_leaf()){
        if (a.feature_index == feature_index){
            vec.push_back(&a);
        }
        filter_by_feature_index(*(a.left), feature_index, vec);
        filter_by_feature_index(*(a.right), feature_index, vec);
    }
}

std::vector<Node*> filter_by_feature_index(Node& a, int feature_index){
    vector<Node*> vec;
    filter_by_feature_index(a, feature_index, vec);
    return vec;
}


int max_feature_index(Node& a){
    if (a.is_leaf())
        return -1;
    int f_l = max_feature_index(*(a.left));
    int f_r = max_feature_index(*(a.right));
    
    int max_f = (f_l > f_r ? f_l : f_r);
    if (a.feature_index > max_f)
        max_f = a.feature_index;
    return max_f;
}

std::vector< std::vector< Node* > > nodes_by_feature(Node& node){
    int max_feat_index = max_feature_index(node);
    vector< vector<Node*> > nodes_by_feature(max_feat_index + 1);
    for(int i = 0; i <= max_feat_index; i++){
        nodes_by_feature[i] = filter_by_feature_index(node, i);
    }

    return nodes_by_feature;
}

std::vector< std::vector< int > > thresholds_by_feature(std::vector<std::vector<Node*> > nodes_by_feat){
    int n = nodes_by_feat.size();
    vector< vector< int > > thrs_by_feature(n);

    for(int i = 0; i < n; i++){
        int _n = nodes_by_feat[i].size();
        thrs_by_feature[i] = vector<int>(_n);
        for(int j = 0; j < _n; j++){
            thrs_by_feature[i][j] = nodes_by_feat[i][j]->threshold;
            cout << thrs_by_feature[i][j] << ", ";
        }
        cout << endl;
    }
    return thrs_by_feature;
}

