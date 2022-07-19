#!/bin/bash

set -e
set -u

mkdir ntl_install

cd ntl_install

wget https://libntl.org/ntl-11.5.1.tar.gz

tar xf ntl-11.5.1.tar.gz

mv ntl-11.5.1/* .

cd src

./configure NTL_STD_CXX11=on

make -j2

make check

sudo make install

cd ..

rm -rf ntl_install
