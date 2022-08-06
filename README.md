# Private decision tree evaluation via Homomorphic Encryption and Transciphering 

This is an implementation of the private decision tree evaluation (PDTE) algorithm
from the paper [SortingHat: Efficient Private Decision Tree Evaluation via Homomorphic Encryption and Transciphering](https://eprint.iacr.org/2022/757), 
by [Kelong Cong](https://www.esat.kuleuven.be/cosic/people/kelong-cong/),
[Debajyoti Das](https://dedas111.github.io/),
[Jeongeun Park](https://sites.google.com/view/jeongeunpark/), 
and [Hilder Vitor Lima Pereira](https://hilder-vitor.github.io/), 
which was published in ACM CCS 2022.

**WARNING**:

This is proof-of-concept implementation.
It may contain bugs and security issues.
Please do not use in production systems.

## Installation

The key dependencies of this project are [Concrete](https://github.com/zama-ai/concrete)
version 1.0.0-beta and [FINAL](https://github.com/KULeuven-COSIC/FINAL).
For Concrete, `cargo` will take care of most of the installation except the FFTW dependency (see below). 
Also, we already include FINAL's code here.
However, the user still need to manually install the dependencies of 
Concrete and FINAL, which are:
- [FFTW](https://www.fftw.org/) -- See their [installation instructions](https://github.com/zama-ai/concrete#installation) for details.
- [GNU GMP](https://gmplib.org/).
- [NTL](https://libntl.org/).

For convinience, we prepared Bash scripts to download and install these three dependencies to `/usr/local/bin`. 
So, if you want to use them, you can simply run

`./install_third_party_libs.sh`

## Running our PDTE

- `cd src/rust_pdte` 
- Run tests: `RUSTFLAGS="-C target-cpu=native" cargo test --release`
- Run micro benchmark: `RUSTFLAGS="-C target-cpu=native" cargo bench`

### Using the CLI

The CLI can be build using 
`RUSTFLAGS="-C target-cpu=native" cargo build --release`.
`cargo` usually puts the executable in `target/release/homdte`.

Without any command line arguments, the CLI will simulate decision tree evaluation
using complete binary trees of various depths.
When given a command line argument of a data directory such as `data/heart` or `data/spam`,
the CLI will evaluation a real model
trained using the training script located in `script/train.py`.
Some models are included in the `data` directory.
The detailed options are available from `./homdte --help`.

For convenience, a script is given under `script/run_all_datasets.sh`
to run private decision tree evaluation on all datasets.

### Optional

A script exists under `script/train.py` which perform the training
using [Concrete ML](https://github.com/zama-ai/concrete-ml) version 0.2.0.
This is only necessary if new models need to be trained.
Running the script is not necessary to use the CLI (described above).

## Running our PDTE with transciphering

- `cd src/cpp_pdte_transciphering`
- `make`
- `./test_pdte_transciphering`
