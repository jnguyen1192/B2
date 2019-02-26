#!/bin/bash

cd Test
read -p "Put the image in format jpg in Crystal Project # 2/Test/data/dataset"
python3 build_descriptors.py
cd ..
cp -f Test/data/dataset/* app/static/dataset_jpg/
cp -f Test/data/descriptor/* app/static/descriptor/
cd app
python3 app.py

