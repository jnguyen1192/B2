#!/usr/bin/env bash

sudo addgroup --system docker

sudo adduser $USER docker

newgrp docker