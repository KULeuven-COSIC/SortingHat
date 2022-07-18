#!/usr/bin/env python3

import sys
import numpy
import json
import random
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.metrics import average_precision_score
from concrete.ml.sklearn import DecisionTreeClassifier as ConcreteDecisionTreeClassifier


class Leaf(object):
    def __init__(self, v):
        self.leaf = int(v)


class Internal(object):
    def __init__(self, threshold, feature, op, left, right):
        self.internal = _Internal(threshold, feature, op, left, right).__dict__


class _Internal(object):
    def __init__(self, threshold, feature, op, left, right):
        self.threshold = int(threshold)
        self.feature = int(feature)
        self.index = 0
        self.op = op
        self.left = left.__dict__
        self.right = right.__dict__


def build_tree(tree):
    node_count = [0]
    return (build_tree_rec(tree, 0, None, 0, node_count), node_count[0])


def build_tree_rec(tree, node_id, parent, depth, node_count):

    left_child = tree.tree_.children_left[node_id]
    right_child = tree.tree_.children_right[node_id]
    is_split_node = left_child != right_child

    if is_split_node:
        left = build_tree_rec(tree, left_child, node_id, depth+1, node_count)
        right = build_tree_rec(tree, right_child, node_id, depth+1, node_count)
        # NOTE: the sklearn decision tree should always use the less or equal comparison
        node_count[0] += 1
        t = Internal(tree.tree_.threshold[node_id], tree.tree_.feature[node_id], "leq", left, right)
    else:
        max_index = tree.tree_.value[node_id].argmax()
        t = Leaf(max_index)

    return t


def tweak_feature(node, feature_count):
    if "leaf" in node:
        return
    node["internal"]["feature"] = random.randrange(feature_count)
    tweak_feature(node["internal"]["left"], feature_count)
    tweak_feature(node["internal"]["right"], feature_count)


identity = lambda x: x
SPAM_CONFIG = {"id": 44, "max_leaf_nodes": 58, "max_depth": 17, "tweak_label": identity}
SPAM2_CONFIG = {"id": 44, "max_leaf_nodes": 40, "max_depth": 12, "tweak_label": identity}
HEART_CONFIG = {"id": 1565,  "max_leaf_nodes": 5, "max_depth": 3, "tweak_label": lambda x: 0 if int(x) == 1 else 1}
STEEL_CONFIG = {"id": 1504,  "max_leaf_nodes": None, "max_depth": 5, "tweak_label": lambda x: int(x)-1}
BREAST_CONFIG = {"id": 1510,  "max_leaf_nodes": None, "max_depth": 10, "tweak_label": lambda x: int(x)-1}
PHONEME_CONFIG = {"id": 1489,  "max_leaf_nodes": None, "max_depth": 10, "tweak_label": lambda x: int(x)-1}
MOZILLA_CONFIG = {"id": 1046,  "max_leaf_nodes": None, "max_depth": 10, "tweak_label": identity} # not UCI
FAKE_ART_CONFIG = {"id": 151,  "max_leaf_nodes": 500, "max_depth": 10, "tweak_label": lambda x: 0 if x == "UP" else 1, "tweak_feat": 16}
FAKE_HOU_CONFIG = {"id": 151,  "max_leaf_nodes": 92, "max_depth": 13, "tweak_label": lambda x: 0 if x == "UP" else 1, "tweak_feat": 13}

if __name__ == "__main__":
    config = SPAM_CONFIG
    if len(sys.argv) != 2:
        print("expected one argument", file=sys.stderr)
        sys.exit(1)
    elif sys.argv[1].lower() == "spam":
        config = SPAM_CONFIG
    elif sys.argv[1].lower() == "spam2":
        config = SPAM2_CONFIG
    elif sys.argv[1].lower() == "heart":
        config = HEART_CONFIG
    elif sys.argv[1].lower() == "steel":
        config = STEEL_CONFIG
    elif sys.argv[1].lower() == "breast":
        config = BREAST_CONFIG
    elif sys.argv[1].lower() == "phoneme":
        config = PHONEME_CONFIG
    elif sys.argv[1].lower() == "electricity":
        config = ELECTRICITY_CONFIG
    elif sys.argv[1].lower() == "mozilla":
        config = MOZILLA_CONFIG
    elif sys.argv[1].lower() == "fake_art":
        config = FAKE_ART_CONFIG
    elif sys.argv[1].lower() == "fake_hou":
        config = FAKE_HOU_CONFIG
    else:
        print("invalid argument: " + sys.argv[1], file=sys.stderr)
        sys.exit(1)

    features, classes = fetch_openml(data_id=config["id"], as_frame=False, cache=True, return_X_y=True)
    classes = numpy.array(list(map(config["tweak_label"], classes)))
    classes = classes.astype(numpy.int64)


    x_train, x_test, y_train, y_test = train_test_split(
        features,
        classes,
        test_size=0.15,
        random_state=42,
    )

    model = ConcreteDecisionTreeClassifier(
        random_state=42,
        max_leaf_nodes=config["max_leaf_nodes"],
        max_depth=config["max_depth"],
        n_bits=11,
    )

    model, sklearn_model = model.fit_benchmark(x_train, y_train)
    y_pred_concrete = model.predict_proba(x_test)[:, 1]
    y_pred_sklearn = sklearn_model.predict_proba(x_test)[:, 1]
    concrete_average_precision = average_precision_score(y_test, y_pred_concrete)
    sklearn_average_precision = average_precision_score(y_test, y_pred_sklearn)
    print(f"Sklearn average precision score: {sklearn_average_precision:0.2f}")
    print(f"Concrete average precision score: {concrete_average_precision:0.2f}")

    (t, count) = build_tree(model)
    print(f"Number of internal nodes: {count}")
    print(f"Depth: {model.get_depth()}")

    if "tweak_feat" in config:
        tweak_feature(t.__dict__, config["tweak_feat"])
        with open("model.json", "w") as f:
            f.write(json.dumps(t.__dict__))
    else:
        with open("model.json", "w") as f:
            f.write(json.dumps(t.__dict__))
        numpy.savetxt("x_test.csv", model.quantize_input(x_test), delimiter=",", fmt="%d")
        numpy.savetxt("y_test.csv", y_test, delimiter=",", fmt="%d")


