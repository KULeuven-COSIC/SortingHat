#include "node.h"
#include <fstream>
#include <iostream>
#include <assert.h>

using namespace std;


void print(const vector<Node*>& u){
	for (unsigned int i = 0; i < u.size(); i++){
		print_node(*(u[i]));
	}
}



int main() {
    {
        auto node = Node();
        node.gen_with_depth(0);
        assert(node.is_leaf());
    }
    {
        auto tree_str = "{\"internal\":{\"threshold\":1,\"feature\":2,\"index\":4,\"op\":\"leq\",\"left\":{\"internal\":{\"threshold\":11,\"feature\":0,\"index\":44,\"op\":\"leq\",\"left\":{\"leaf\":9},\"right\":{\"leaf\":2}}},\"right\":{\"leaf\":3}}}";
        auto j = json::parse(tree_str);
        auto node = Node(j);

        print_tree(node);

        assert(!node.is_leaf());

        assert(node.threshold == 1);
        assert(node.feature_index == 2);

        assert(node.right->is_leaf());
        assert(node.right->class_leaf == 3);

        assert(!node.left->is_leaf());
        assert(node.left->threshold == 11);
        assert(node.left->feature_index == 0);

        auto features = std::vector<unsigned>{1, 2};
    }
    {
        std::ifstream ifs("data/electricity_10bits/model.json");
        json j = json::parse(ifs);
        auto node = Node(j);
        print_tree(node);
        auto feat_3 = filter_by_feature_index(node, 3);
        assert(!node.is_leaf());
//        assert(node.threshold == 138);
        assert(node.feature_index == 3);
        assert(node.get_depth() == 10);


        print(feat_3);

        int max_feat_index = max_feature_index(node);

        vector< vector<Node*> > nodes_by_feature(max_feat_index + 1);

        for(int i = 0; i <= max_feat_index; i++){
            nodes_by_feature[i] = filter_by_feature_index(node, i);
        }
        
        for(int i = 0; i <= max_feat_index; i++){
            cout << "feature index = " << i << endl;
            print(nodes_by_feature[i]);
        }

    }
    {
        std::ifstream ifs("data/phoneme_10bits/model.json");
        json j = json::parse(ifs);
        auto node = Node(j);
        print_tree(node);

        int max_feat_index = max_feature_index(node);

        vector< vector<Node*> > nodes_by_feat = nodes_by_feature(node);

        for(int i = 0; i <= max_feat_index; i++){
            cout << "feature index = " << i << endl;
            print(nodes_by_feat[i]);
        }

        std::vector< std::vector< int > > thrs_by_feat = thresholds_by_feature(nodes_by_feat);
    }

    cout << "ok" << endl;
}
