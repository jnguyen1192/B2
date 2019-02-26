# First Project : Image Retrieval 

Implementation of the image retrieval using Deep descriptor (as RMAC).

* Flask code from : https://www.pyimagesearch.com/2014/12/08/adding-web-interface-image-search-engine-flask/
* RMAC code from : https://github.com/noagarcia/keras_rmac

## Prerequisites 
- [Python][1] (3.6)

## Local descriptor

### Explanation
In this part, we need to retrieve the duplicate image from a dataset.

We follow the rules and use the RMAC algorithm to describe a feature.

The RMAC algorithm was created by Noa Garcia, we only have to use the build of an RMAC descriptor.


### Algorithm
The main algorithm consists in create a Deep Neural Network to describe a feature.
We choose to create a class to manipulate the RMAC descriptor.
We also choose the cosinus similarity to know the score between two images.
### Context
We need to add our image in the directory dataset_jpg on the static directory and on dataset directory from Test directory
### Deep descriptor - Results
To obtain the descriptor directory, we need almost 5 hours for 812 pictures.
To build one Rmac descriptor, we need 20 seconds.
To find duplicate images, we need 30 seconds.
### Usage
First we need to install the libraries using requirement.txt file.
pip install -r requirements.txt
## Deep

To create a descriptor directory.
You need to launch the unittest with some pictures on the dataset directory then duplicate them on the dataset_jpg directory.

To launch the flask application.
Launch those commands :
* cd "Crystal Project #2"
* chmod +x app.sh
* ./app.sh

And follow the instructions.
If it won't work you can contact me at johnny.nguyen@etu.unice.fr

## References
- Frederic, P from Shallow to Deep Representation for multimedia database. Lectures 2019.
- Adrian, R from pyimagesearch.com. Website 2014.
- Noa, G from Github. 2018

``` 

[1]: https://www.python.org/download/releases/3.6/

