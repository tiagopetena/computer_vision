#!/bin/sh 

if [ ! -f input/BSDS300-images.tgz ]; then
    wget https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/segbench/BSDS300-images.tgz -P ./input/
    wget https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/segbench/BSDS300-human.tgz -P ./input/

    tar xvf ./input/BSDS300-images.tgz -C ./input/
	tar xvf ./input/BSDS300-human.tgz -C ./input/
fi