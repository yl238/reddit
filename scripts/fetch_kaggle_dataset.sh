#!/usr/bin/env bash

kaggle competitions download -c house-prices-advanced-regression-techniques -p packages/random_forest_model/random_forest_model/datasets/ 
unzip -o packages/random_forest_model/random_forest_model/datasets/*.zip 
mv test.csv train.csv packages/random_forest_model/random_forest_model/datasets/