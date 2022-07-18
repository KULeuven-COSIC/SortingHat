use std::fs;
use std::path::Path;
use clap::Parser;
use rand::Rng;

use homdte::decision_tree::*;
use homdte::Scalar;

#[derive(Parser, Debug)]
#[clap(author, version, about="Privacy preserving decision tree from TFHE.", long_about = None)]
struct Cli {
    #[clap(long, help="directory for the model data")]
    dir: Option<String>,

    #[clap(long, default_value_t=1, help="number of testing data to use, 0 means use all")]
    input_size: usize,

    #[clap(long, help="running the evaluation in parallel or not")]
    parallel: bool,

    #[clap(long, help="when this is true, do not read x_test.csv")]
    artificial: bool,

    #[clap(long, help="print more information")]
    verbose: bool,
}

fn parse_csv(path: &Path) -> Vec<Vec<usize>> {
    let x_test_f = fs::File::open(path).expect("csv file not found, consider using --artificial");

    let mut x_test: Vec<Vec<usize>> = vec![];
    let mut x_train_rdr = csv::Reader::from_reader(x_test_f);
    for res in x_train_rdr.records() {
        let record = res.unwrap();
        let row = record.iter().map(|s| {
            s.parse().unwrap()
        }).collect();
        x_test.push(row);
    }

    x_test
}

fn main() {
    let cli = Cli::parse();
    match cli.dir {
        None => {
            for depth in 6..=11 {
                let root = Node::new_with_depth(depth);
                let features = vec![1usize];
                let res = simulate(&root, &vec![features], cli.parallel);
                println!("{}", res);
            }
        }
        Some(dir) => {
            let base_path = Path::new(&dir);
            let model_path = base_path.join("model.json");
            let x_test_path = base_path.join("x_test.csv");
            let y_test_path = base_path.join("y_test.csv");

            let input_size = cli.input_size;
            let model_f = fs::File::open(model_path).unwrap();
            let root: Node = serde_json::from_reader(model_f).expect("cannot parse json");

            // set the features, generate them if necessary
            let x_test = {
                if cli.artificial {
                    let mut rng = rand::thread_rng();
                    let feature_count = root.max_feature_index() + 1;
                    let mut out = vec![];
                    while out.len() < input_size {
                        out.push((0..feature_count).map(|_| {
                            // TODO get 2048 from Context
                            rng.gen_range(0..2048)
                        }).collect());
                    }
                    out
                } else {
                    parse_csv(&x_test_path)
                }
            };

            // set the classification/label
            let y_test = {
                if cli.artificial {
                    vec![]
                } else {
                    parse_csv(&y_test_path)
                }
            };
            let class_count = x_test[0].len();

            let sim_results = if input_size == 0 {
                simulate(&root, &x_test, cli.parallel)
            } else {
                simulate(&root, &x_test.into_iter().take(input_size).collect(), cli.parallel)
            };

            if cli.verbose {
                println!("{}", sim_results);
                if !cli.artificial {
                    // test whether the PDTE results match the expected result from `y_test`
                    let error_count: Scalar = y_test.iter().zip(sim_results.predictions).map(|(y1, y2)| {
                        let expected = y1[0] as Scalar;
                        if expected != y2 {
                            1
                        } else {
                            0
                        }
                    }).sum();
                    println!("{} errors out of {}", error_count, input_size);
                }
            } else {
                // dataset,depth,leaf_count,internal_count,class_count,duration
                println!("{},{},{},{},{},{:.2}",
                         base_path.to_str().unwrap(),
                         root.count_depth(),
                         root.count_leaf(),
                         root.count_internal(),
                         class_count,
                         sim_results.server_duration.as_micros() as f64 / sim_results.input_count as f64 / 1000f64)
            }
        }
    }
}
