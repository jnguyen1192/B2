#!/bin/bash

#sudo apt-get install docker.io
sudo docker pull postgres
sudo docker run --name image-retrieval-postgres -v $(pwd)/data:/var/lib/postgresql/data -e POSTGRES_PASSWORD=postgres -d postgres
