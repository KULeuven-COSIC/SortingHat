#!/bin/bash

set -e
set -u

mkdir gmp_install

cd gmp_install

wget https://gmplib.org/download/gmp/gmp-6.2.1.tar.xz

tar xf gmp-6.2.1.tar.xz

mv gmp-6.2.1/* .

./configure --enable-cxx

make -j2
make check
sudo make install

rm gmp-6.2.1.tar.xz
rm -rf gmp_install

