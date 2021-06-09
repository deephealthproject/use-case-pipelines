#!/usr/bin/env bash

ROOT_DIR=$(pwd)

wget https://github.com/yaml/libyaml/releases/download/0.2.5/yaml-0.2.5.tar.gz
tar -xzf yaml-0.2.5.tar.gz && rm yaml-0.2.5.tar.gz
cd yaml-0.2.5
./bootstrap
./configure --prefix=$(pwd)/install
make -j7
make install

LIBYAML_INSTALL_DIR=$(pwd)/install
cd $ROOT_DIR

wget https://github.com/yaml/pyyaml/archive/5.4.1.tar.gz
tar -xzf 5.4.1.tar.gz && rm 5.4.1.tar.gz
cd pyyaml-5.4.1/
python setup.py --with-libyaml build_ext --library-dirs=$LIBYAML_INSTALL_DIR/lib --include-dirs=$LIBYAML_INSTALL_DIR/include
python setup.py install --user