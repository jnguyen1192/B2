#!/bin/bash

# stop container if it already run
docker stop image-retrieval-deep & docker rm image-retrieval-deep
# build
docker build --rm -t image-retrieval-deep .
# run container
docker run --rm -p 80:5000 image-retrieval-deep
#
#echo $(pwd)/data
#docker run --name image-retrieval-postgres -v $(pwd)/data:/var/lib/postgresql/data -d postgres