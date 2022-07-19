#!/bin/bash

set -e
set -u

mkdir fftw_install

cd fftw_install

wget https://fftw.org/fftw-3.3.10.tar.gz

tar xf fftw-3.3.10.tar.gz

mv fftw-3.3.10/* .


./configure #--enable-avx -enable-sse2

make -j2

sudo make install

cd ..

# clean up
rm -rf fftw_install 
