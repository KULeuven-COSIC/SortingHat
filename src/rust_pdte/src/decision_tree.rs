use std::time::Duration;
use serde::{Deserialize, Serialize};
use rayon::prelude::*;
use bitvec::prelude::*;
use crate::*;

#[derive(Clone, Copy, Debug, Deserialize, Serialize)]
#[serde(rename_all = "lowercase")]
/// Comparison operation in the decision node.
pub enum Op {
    LEQ,
    GT,
}

#[derive(Debug, Deserialize, Serialize)]
#[serde(rename_all = "lowercase")]
/// A node in the tree.
pub enum Node {
    Internal(Box<Internal>),
    Leaf(usize),
}

impl Node {
    /// Turn the node to an Internal, panic if it's a leaf.
    pub fn unwrap(self) -> Internal {
        match self {
            Node::Internal(x) => *x,
            Node::Leaf(_) => panic!("this is a leaf"),
        }
    }

    /// Create a tree with depth 1 (one internal node)
    pub fn new() -> Node {
        let mut processed_one_leaf = false;
        gen_full_tree(1, &mut processed_one_leaf)
    }

    /// Create a tree with depth d
    pub fn new_with_depth(d: usize) -> Node {
        let mut processed_one_leaf = false;
        gen_full_tree(d, &mut processed_one_leaf)
    }

    /// Assign a unique index to every node in DFS order.
    pub fn fix_index(&mut self) -> usize {
        match self {
            Node::Internal(internal) => fix_index(internal, 0),
            Node::Leaf(_) => panic!("this is a leaf")
        }
    }

    /// Return the flattened version of the tree.
    /// If fix_index is called prior, then the index should be ordered.
    pub fn flatten(&self) -> Vec<Internal> {
        match self {
            Node::Internal(internal) => {
                let mut out = Vec::new();
                // TODO reserve memory
                flatten_tree(&mut out, internal);
                out
            }
            Node::Leaf(_) => vec![],
        }
    }

    /// Evaluate the decision tree with a feature vector and output the final class.
    pub fn eval(&self, features: &Vec<usize>) -> usize {
        let mut out = 0;
        eval_node(&mut out, self, features, 1);
        out
    }

    /// Count the number of leaves.
    pub fn count_leaf(&self) -> usize {
        match self {
            Node::Internal(internal) => {
                internal.left.count_leaf() + internal.right.count_leaf()
            }
            Node::Leaf(_) => 1
        }
    }

    /// Count the number of internal nodes.
    pub fn count_internal(&self) -> usize {
        match self {
            Node::Internal(internal) => {
                1 + internal.left.count_internal() + internal.right.count_internal()
            }
            Node::Leaf(_) => 0
        }
    }

    /// Count the maximum depth.
    pub fn count_depth(&self) -> usize {
        match self {
            Node::Internal(internal) => {
                let l = internal.left.count_depth();
                let r = internal.right.count_depth();
                if l > r {
                    l + 1
                } else {
                    r + 1
                }
            }
            Node::Leaf(_) => 0,
        }
    }

    /// Find the maximum feature index in the tree.
    pub fn max_feature_index(&self) -> usize {
        match self {
            Node::Internal(internal) => {
                let i = internal.feature;
                i.max(internal.left.max_feature_index()).max(internal.right.max_feature_index())
            }
            Node::Leaf(_) => 0
        }
    }
}

#[derive(Debug, Deserialize, Serialize)]
#[serde(rename_all = "lowercase")]
/// An internal node in a decision tree.
pub struct Internal {
    pub threshold: usize,
    pub feature: usize,
    pub index: usize,
    pub op: Op,
    pub left: Node,
    pub right: Node,
}

fn fix_index(node: &mut Internal, i: usize) -> usize {
    node.index = i;
    let j = match &mut node.left {
        Node::Leaf(_) => i,
        Node::Internal(left) => fix_index(left, i + 1)
    };
    match &mut node.right {
        Node::Leaf(_) => j,
        Node::Internal(right) => fix_index(right, j + 1)
    }
}

fn flatten_tree(out: &mut Vec<Internal>, node: &Internal) {
    out.push(Internal {
        threshold: node.threshold,
        feature: node.feature,
        index: node.index,
        op: node.op,
        left: Node::Leaf(0),
        right: Node::Leaf(0),
    });
    match &node.left {
        Node::Leaf(_) => (),
        Node::Internal(left) => flatten_tree(out, left),
    }
    match &node.right {
        Node::Leaf(_) => (),
        Node::Internal(right) => flatten_tree(out, right),
    }
}

fn eval_node(out: &mut usize, node: &Node, features: &Vec<usize>, b: usize) {
    match node {
        Node::Leaf(x) => {
            *out += *x * b;
        }
        Node::Internal(node) => {
            match node.op {
                Op::LEQ => {
                    if features[node.feature] <= node.threshold {
                        eval_node(out, &node.left, features, b);
                        eval_node(out, &node.right, features, b * (1 - b));
                    } else {
                        eval_node(out, &node.left, features, b * (1 - b));
                        eval_node(out, &node.right, features, b);
                    }
                }
                Op::GT => todo!(),
            }
        }
    }
}


fn gen_full_tree(d: usize, processed_one_leaf: &mut bool) -> Node {
    if d == 0 {
        if *processed_one_leaf {
            Node::Leaf(0)
        } else {
            *processed_one_leaf = true;
            Node::Leaf(1)
        }
    } else {
        Node::Internal(Box::new(Internal {
            threshold: 0,
            feature: 0,
            index: 0,
            op: Op::LEQ,
            left: gen_full_tree(d - 1, processed_one_leaf),
            right: gen_full_tree(d - 1, processed_one_leaf),
        }))
    }
}

/// Perform the comparison operation between (RLWE) encrypted features and the flattened plaintext decision tree
/// and then output an iterator of (RGSW) encrypted choice bits.
pub fn compare_expand<'a>(flat_nodes: &'a Vec<Internal>,
                          client_cts: &'a Vec<Vec<RLWECiphertext>>,
                          neg_sk_ct: &'a RGSWCiphertext,
                          ksk_map: &'a HashMap<usize, FourierRLWEKeyswitchKey>,
                          ctx: &'a Context,
                          buffers: &'a mut FourierBuffers<Scalar>) -> impl Iterator<Item=RGSWCiphertext> + 'a {
    flat_nodes.iter().map(|node| {
        let cts = client_cts[node.feature].iter().map(|c| {
            let mut ct = RLWECiphertext::allocate(ctx.poly_size);
            ct.fill_with_copy(c);
            match node.op {
                Op::LEQ => ct.less_eq_than(node.threshold, buffers),
                Op::GT => todo!(),
            }
            ct
        }).collect();
        expand_fourier(&cts, ksk_map, neg_sk_ct, ctx, buffers)
    })
}

/// An encrypted node.
pub enum EncNode {
    Internal(Box<EncInternal>),
    Leaf(usize),
}

impl EncNode {
    /// Create a new root from  a plaintext root and encrypted choice bits.
    pub fn new(clear_root: &Node, rgsw_cts: &mut impl Iterator<Item=RGSWCiphertext>) -> EncNode {
        let ct = rgsw_cts.next().unwrap();
        let mut out = EncInternal {
            ct,
            left: EncNode::Leaf(0),
            right: EncNode::Leaf(0),
        };
        match clear_root {
            Node::Internal(inner) => new_enc_node(&mut out, inner, rgsw_cts),
            Node::Leaf(_) => panic!("this is a leaf"),
        }
        EncNode::Internal(Box::new(out))
    }

    /// Evaluate the tree.
    pub fn eval(&self, ctx: &Context, buffers: &mut FourierBuffers<Scalar>) -> Vec<RLWECiphertext> {
        let max_leaf_bits = ((self.max_leaf() + 1) as f64).log2().ceil() as usize;
        let mut out = vec![RLWECiphertext::allocate(ctx.poly_size); max_leaf_bits];
        let mut c = RLWECiphertext::allocate(ctx.poly_size);
        *c.get_mut_body().as_mut_tensor().first_mut() = Scalar::one();
        binary_encode(c.get_mut_body().as_mut_tensor().first_mut());
        eval_enc_node(&mut out, self, c, ctx, buffers);
        out
    }

    pub fn max_leaf(&self) -> usize {
        match self {
            EncNode::Internal(internal) => {
                let l = internal.left.max_leaf();
                let r = internal.right.max_leaf();
                if l > r {
                    l
                } else {
                    r
                }
            }
            EncNode::Leaf(x) => *x
        }
    }
}

/// An encrypted internal node where the ciphertext is the choice bit.
pub struct EncInternal {
    pub ct: RGSWCiphertext,
    pub left: EncNode,
    pub right: EncNode,
}

fn new_enc_node(enc_node: &mut EncInternal, clear_node: &Internal, rgsw_cts: &mut impl Iterator<Item=RGSWCiphertext>) {
    match &clear_node.left {
        Node::Leaf(x) => enc_node.left = EncNode::Leaf(*x),
        Node::Internal(left) => match rgsw_cts.next() {
            None => panic!("missing RGSW ciphertext"),
            Some(ct) => {
                let mut new_node = EncInternal {
                    ct,
                    left: EncNode::Leaf(0),
                    right: EncNode::Leaf(0),
                };
                new_enc_node(&mut new_node, left, rgsw_cts);
                enc_node.left = EncNode::Internal(Box::new(new_node));
            }
        },
    }
    match &clear_node.right {
        Node::Leaf(x) => enc_node.right = EncNode::Leaf(*x),
        Node::Internal(right) => match rgsw_cts.next() {
            None => panic!("missing RGSW ciphertext"),
            Some(ct) => {
                let mut new_node = EncInternal {
                    ct,
                    left: EncNode::Leaf(0),
                    right: EncNode::Leaf(0),
                };
                new_enc_node(&mut new_node, right, rgsw_cts);
                enc_node.right = EncNode::Internal(Box::new(new_node));
            }
        },
    }
}

fn eval_enc_node(out: &mut Vec<RLWECiphertext>, node: &EncNode, b: RLWECiphertext, ctx: &Context, buffers: &mut FourierBuffers<Scalar>) {
    match node {
        EncNode::Leaf(x) => {
            for (bit, ct) in (*x).view_bits::<Lsb0>().iter().zip(out.iter_mut()) {
                if *bit {
                    ct.update_with_add(&b);
                }
            }
        }
        EncNode::Internal(node) => {
            let mut left = RLWECiphertext::allocate(ctx.poly_size);
            node.ct.external_product_with_buf(&mut left, &b, buffers);
            let mut right = b;
            right.update_with_sub(&left);

            eval_enc_node(out, &node.left, left, ctx, buffers);
            eval_enc_node(out, &node.right, right, ctx, buffers);
        }
    }
}

/// Every feature v is encrypted as RLWE(1/(B^j n) X^v) for j in 1...\ell
pub fn encrypt_feature_vector(sk: &RLWESecretKey, vs: &Vec<usize>, ctx: &mut Context) -> Vec<Vec<RLWECiphertext>> {
    let mut pt = PlaintextList::allocate(Scalar::zero(), ctx.plaintext_count());
    let logn = log2(ctx.poly_size.0);
    let mut out = Vec::new(); // TODO preallocate
    for v in vs {
        let mut tmp = Vec::new();
        for level in 1..=ctx.level_count.0 {
            assert!(*v < ctx.poly_size.0);
            let shift: usize = (Scalar::BITS as usize) - ctx.base_log.0 * level - logn;
            pt.as_mut_tensor().fill_with_element(Scalar::zero());
            *pt.as_mut_polynomial().get_mut_monomial(MonomialDegree(*v)).get_mut_coefficient() = Scalar::one() << shift;

            let mut ct = RLWECiphertext::allocate(ctx.poly_size);
            sk.encrypt_rlwe(&mut ct, &pt, ctx.std, &mut ctx.encryption_generator);
            tmp.push(ct);
        }
        out.push(tmp);
    }
    out
}

pub struct SimulationResult {
    pub input_count: usize,
    pub setup_duration: Duration,
    pub server_duration: Duration,
    pub predictions: Vec<Scalar>,
    pub std: LogStandardDev,
    pub poly_size: PolynomialSize,
    pub base_log: DecompositionBaseLog,
    pub level_count: DecompositionLevelCount,
    pub ks_base_log: DecompositionBaseLog,
    pub ks_level_count: DecompositionLevelCount,
    pub negs_base_log: DecompositionBaseLog,
    pub negs_level_count: DecompositionLevelCount,
}

impl SimulationResult {
    pub fn new(input_count: usize, setup_duration: Duration, server_duration: Duration, predictions: Vec<Scalar>, ctx: &Context) -> SimulationResult {
        SimulationResult {
            input_count,
            setup_duration,
            server_duration,
            predictions,
            std: ctx.std,
            poly_size: ctx.poly_size,
            base_log: ctx.base_log,
            level_count: ctx.level_count,
            ks_base_log: ctx.ks_base_log,
            ks_level_count: ctx.ks_level_count,
            negs_base_log: ctx.negs_base_log,
            negs_level_count: ctx.negs_level_count,
        }
    }
}

impl fmt::Display for SimulationResult {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "input_count={}, setup_duration={:?}, server_duration={:?}, q={:?}, \
            poly_size={:?}, log_std={:?}, default_decomp=({:?},{:?}), ks_decomp=({:?},{:?}), negs_decomp=({:?},{:?})",
               self.input_count, self.setup_duration, self.server_duration,
               Scalar::BITS, self.poly_size.0, self.std.get_log_standard_dev(),
               self.base_log.0, self.level_count.0, self.ks_base_log.0, self.ks_level_count.0, self.negs_base_log.0, self.negs_level_count.0)
    }
}

fn decrypt_and_recompose(sk: &RLWESecretKey, cts: &Vec<RLWECiphertext>, ctx: &Context) -> Scalar {
    let mut bv: BitVec<Scalar, Lsb0> = BitVec::new();
    let mut pt = PlaintextList::allocate(Scalar::zero(), ctx.plaintext_count());
    for ct in cts {
        sk.binary_decrypt_rlwe(&mut pt, ct);
        match pt.as_tensor().first() {
            0 => bv.push(false),
            1 => bv.push(true),
            _ => panic!("expected binary plaintext"),
        }
    }
    bv.load::<Scalar>()
}

/// Simulate PDTE evaluations by specifying a `model`, a set of `features`.
/// If `parallel` is set to true then different features are evaluated in parallel.
/// See the rayon documentation for how to control the number of threads.
/// Finally, return a `SimulationResult` which mainly consists of the timing
/// information and the evaluation result.
pub fn simulate(model: &Node, features: &Vec<Vec<usize>>, parallel: bool) -> SimulationResult {
    // Client side
    let setup_instant = Instant::now();
    let mut ctx = Context::default();
    let sk = ctx.gen_rlwe_sk();
    let neg_sk_ct = sk.neg_gsw(&mut ctx);
    let mut buffers = ctx.gen_fourier_buffers();
    let ksk_map = gen_all_subs_ksk_fourier(&sk, &mut ctx, &mut buffers);
    let client_cts: Vec<Vec<Vec<RLWECiphertext>>> = features.iter().map(|f| encrypt_feature_vector(&sk, &f, &mut ctx)).collect();
    let flat_nodes = model.flatten();
    let setup_duration = setup_instant.elapsed();

    // Server side
    let server_f = |ct, buffers: &mut FourierBuffers<Scalar>| {
        let enc_root = {
            let mut rgsw_cts = compare_expand(&flat_nodes, ct, &neg_sk_ct, &ksk_map, &ctx, buffers);
            EncNode::new(&model, &mut rgsw_cts)
        };
        let final_label_ct = enc_root.eval(&ctx, buffers);
        final_label_ct
    };
    let server_instant = Instant::now();
    let output_cts: Vec<Vec<RLWECiphertext>> = if parallel {
            client_cts.par_iter().map(|ct| {
                // NOTE: we need to create new buffers for every operation since it's not thread safe
                let mut buffers = ctx.gen_fourier_buffers();
                server_f(ct, &mut buffers)
            }).collect()
        } else {
            client_cts.iter().map(|ct| {
                server_f(ct, &mut buffers)
            }).collect()
    };
    let server_duration = server_instant.elapsed();

    // Check correctness by doing the evaluation on plaintext model
    let mut predictions = vec![];
    for (ct, feature) in output_cts.iter().zip(features.iter()) {
        let actual_scalar = decrypt_and_recompose(&sk, ct, &ctx);
        let expected_scalar = model.eval(feature) as Scalar;
        assert_eq!(expected_scalar, actual_scalar);
        predictions.push(expected_scalar);
    }

    let input_count = features.len();
    SimulationResult::new(input_count,
                          setup_duration,
                          server_duration,
                          predictions,
                          &ctx)
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_json() {
        let root = Node::Internal(Box::new(Internal {
            threshold: 1,
            feature: 2,
            index: 4,
            op: Op::LEQ,
            left: Node::Internal(Box::new(Internal {
                threshold: 11,
                feature: 22,
                index: 44,
                op: Op::GT,
                left: Node::Leaf(1),
                right: Node::Leaf(2),
            })),
            right: Node::Leaf(3),
        }));
        assert_eq!(
            r#"{"internal":{"threshold":1,"feature":2,"index":4,"op":"leq","left":{"internal":{"threshold":11,"feature":22,"index":44,"op":"gt","left":{"leaf":1},"right":{"leaf":2}}},"right":{"leaf":3}}}"#,
            serde_json::to_string(&root).unwrap());
    }

    #[test]
    fn test_clear_node() {
        let mut root = Node::Internal(Box::new(Internal {
            threshold: 1,
            feature: 2,
            index: 4,
            op: Op::LEQ,
            left: Node::Internal(Box::new(Internal {
                threshold: 11,
                feature: 22,
                index: 44,
                op: Op::GT,
                left: Node::Leaf(0),
                right: Node::new(),
            })),
            right: Node::Leaf(0),
        }));

        assert_eq!(root.fix_index(), 2);
        for (i, x) in root.flatten().iter().enumerate() {
            assert_eq!(x.index, i);
        }

        let internal = root.unwrap();
        assert_eq!(internal.index, 0);
        let left = internal.left.unwrap();
        assert_eq!(left.index, 1);
        let right = left.right.unwrap();
        assert_eq!(right.index, 2);
    }

    #[test]
    fn test_traversal_1() {
        // In this example we consider 2 features, 2 labels and and 3 nodes as shown below
        //        f_0           f_0 <= 2
        //       /   \
        //      f_1   l_0       f_1 <= 2
        //     /  \
        //   l_0   f_1          f_1 <= 3
        //        /  \
        //     l_10  l_0

        let root = {
            let mut tmp = Node::Internal(Box::new(Internal {
                threshold: 2,
                feature: 0,
                index: 0,
                op: Op::LEQ,
                left: Node::Internal(Box::new(Internal {
                    threshold: 2,
                    feature: 1,
                    index: 0,
                    op: Op::LEQ,
                    left: Node::Leaf(0),
                    right: Node::Internal(Box::new(Internal {
                        threshold: 3,
                        feature: 1,
                        index: 0,
                        op: Op::LEQ,
                        left: Node::Leaf(10),
                        right: Node::Leaf(0),
                    })),
                })),
                right: Node::Leaf(0),
            }));
            assert_eq!(tmp.fix_index(), 2);
            tmp
        };
        assert_eq!(root.count_leaf(), 4);
        assert_eq!(root.count_internal(), 3);
        assert_eq!(root.max_feature_index(), 1);

        let features = vec![2, 3]; // f_0 = 2, f_1 = 3
        assert_eq!(10, root.eval(&features));

        simulate(&root, &vec![features], false);
    }

    #[test]
    fn test_traversal_2() {
        // In this example we consider 2 features, 2 labels and and 3 nodes as shown below
        //        f_0           f_0 <= 2
        //       /   \
        //      f_1   l_1       f_1 <= 2
        //     /  \
        //   l_1   f_1          f_1 <= 1
        //        /  \
        //     l_1  l_0

        let root = {
            let mut tmp = Node::Internal(Box::new(Internal {
                threshold: 2,
                feature: 0,
                index: 0,
                op: Op::LEQ,
                left: Node::Internal(Box::new(Internal {
                    threshold: 2,
                    feature: 1,
                    index: 0,
                    op: Op::LEQ,
                    left: Node::Leaf(1),
                    right: Node::Internal(Box::new(Internal {
                        threshold: 1,
                        feature: 1,
                        index: 0,
                        op: Op::LEQ,
                        left: Node::Leaf(1),
                        right: Node::Leaf(0),
                    })),
                })),
                right: Node::Leaf(1),
            }));
            assert_eq!(tmp.fix_index(), 2);
            tmp
        };
        assert_eq!(root.count_leaf(), 4);
        assert_eq!(root.count_internal(), 3);
        assert_eq!(root.max_feature_index(), 1);

        let features = vec![2, 3]; // f_0 = 2, f_1 = 3
        assert_eq!(0, root.eval(&features));

        simulate(&root, &vec![features], false);
    }

    #[test]
    fn test_traversal_long() {
        const TH: usize = 10;
        const D: usize = 10;
        fn gen_line(d: usize) -> Node {
            if d == 0 {
                Node::Leaf(1)
            } else {
                let node = Node::Internal(Box::new(Internal {
                    threshold: TH,
                    feature: 0,
                    index: 0,
                    op: Op::LEQ,
                    left: gen_line(d - 1),
                    right: Node::Leaf(0),
                }));
                node
            }
        }
        let root = {
            let mut out = gen_line(D);
            assert_eq!(out.fix_index(), D - 1);
            out
        };
        {
            let features = vec![1usize];
            assert_eq!(1, root.eval(&features));
            simulate(&root, &vec![features], false);
        }
        {
            let features = vec![11usize];
            assert_eq!(0, root.eval(&features));
            simulate(&root, &vec![features], false);
        }
    }

    #[test]
    fn test_depth() {
        assert_eq!(Node::new_with_depth(0).count_depth(), 0);
        assert_eq!(Node::new_with_depth(1).count_depth(), 1);
        assert_eq!(Node::new_with_depth(3).count_depth(), 3);
    }

    #[test]
    fn test_bitvec() {
        // test conversion
        let mut bv_one: BitVec<usize, Lsb0> = BitVec::new();
        bv_one.push(true);
        let mut bv_two: BitVec<usize, Lsb0> = BitVec::new();
        bv_two.push(false);
        bv_two.push(true);

        assert_eq!(bv_one.load::<usize>(), 1);
        assert_eq!(bv_two.load::<usize>(), 2);

        // test decomposition
        let v = 10usize; // 1010
        let v_bits = v.view_bits::<Lsb0>().to_bitvec();
        assert_eq!(v_bits[0], false);
        assert_eq!(v_bits[1], true);
        assert_eq!(v_bits[2], false);
        assert_eq!(v_bits[3], true);
    }
}
