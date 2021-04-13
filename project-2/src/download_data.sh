#!/bin/bash

pip3 install gdown
pip3 install pandas
pip install scikit-image
pip install scipy

zebraFish_1="https://drive.google.com/uc?id=1TXvAkCxznNh9UeSqA1_hVo70rwMPevU9"
zebraFish_2="https://drive.google.com/u/1/uc?id=1_6Kx4hBlvMgQvsv95FVrIJlxiUFknc3-&export=download"
zebraFish_3="https://drive.google.com/u/1/uc?id=1FPX60evv6nJCvM2NbD9ql6xVzzZVZ5vr&export=download"
zebraFish_4="https://drive.google.com/u/1/uc?id=1ZQW2BU2b85wcOgbv-Jq4kE6pnr9ICQZa&export=download"
zebraFish_5="https://drive.google.com/u/1/uc?id=1hOn5_33NZUVMbKN3LenhG-47gyfk3Nvc&export=download"
zebraFish_6="https://drive.google.com/u/1/uc?id=1QUvkC4EWoPdzq1ieEfPt5PVSqLK-NRGn&export=download"
zebraFish_7="https://drive.google.com/u/1/uc?id=1AAkkGhnbJWsottH5plzjtp57hjixj5qF&export=download"
zebraFish_8="https://drive.google.com/u/1/uc?id=1Y-KxYmphqQGd-L0QUYYfUyhdfJma5aGZ&export=download"

gdown $zebraFish_2 -O ./input/zf2.zip
unzip input/zf2.zip -d input/zf2
rm input/zf2.zip

gdown $zebraFish_6 -O ./input/zf6.zip 
unzip input/zf6.zip -d input/zf6
rm input/zf6.zip
