
#ifndef FILIP_SRC_NODE_H
#define FILIP_SRC_NODE_H

#include <memory>
#include <string>
#include <variant>
#include "json.hpp"

#include "final/FINAL.h"

using json = nlohmann::json;

typedef unsigned Leaf;


class Node {
public:
    std::shared_ptr<Node> left;
    std::shared_ptr<Node> right;
    int threshold;
    int feature_index;
    int class_leaf; // class assigned to the feature. Only leaves have this field set
    std::string op;

    Ctxt_LWE control_bit; // result of comparison of encrypted feature with the threshold value
    Ctxt_LWE value;

    Node() = default;;
    explicit Node(json::reference j);
    void gen_with_depth(int d);
    unsigned get_depth();
    bool is_leaf() const;
    unsigned eval(const std::vector<unsigned> &features);
};

void print_node(Node& a, std::string tabs = "");
void print_tree(Node& a);

std::vector<Node*> filter_by_feature_index(Node& a, int feature_index);

int max_feature_index(Node& a);

std::vector< std::vector< Node* > > nodes_by_feature(Node& node);

std::vector< std::vector< int > > thresholds_by_feature(std::vector<std::vector<Node*> > nodes_by_feat);

#endif //FILIP_SRC_NODE_H
