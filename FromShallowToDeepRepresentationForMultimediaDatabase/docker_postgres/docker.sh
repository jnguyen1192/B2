#!/bin/bash

#echo $USER
#newgrp docker &
#sudo apt-get install docker.io

docker pull postgres
# stop container if it already run
docker stop image-retrieval-postgres & docker rm image-retrieval-postgres
# run container
docker run --rm --name image-retrieval-postgres -p 5432:5432 -v $(pwd)/pgdata:/var/lib/postgresql -e POSTGRES_PASSWORD=postgres -d postgres
#
#echo $(pwd)/data
#docker run --name image-retrieval-postgres -v $(pwd)/data:/var/lib/postgresql/data -d postgres