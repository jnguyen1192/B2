# First Project : Image Retrieval 

Implementation of the image retrieval using Local (as SIFT) and Global descriptor (as BOVW).

* Local code from :  https://docs.opencv.org
* Bonus code from : https://www.pyimagesearch.com/2014/12/08/adding-web-interface-image-search-engine-flask/
* Global code from : https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_ml/py_kmeans/py_kmeans_opencv/py_kmeans_opencv.html#kmeans-opencv 

## Prerequisites 
This code requires OpenCV version 3.3.0.10 and .OpenCV-contrib version 3.3.0.10
- [Python][1] (3.6)

## Local descriptor

### Explanation
In this part, we need to retrieve the duplicate image from a dataset.

We follow the rules and use the SIFT algorithm from the OpenCV library to get the descriptor of each images.

We choose to use the FLANN matcher to retrieve the duplicate image.
The FLANN matcher is a function that take two descriptors of two different images.
We obtain an array that representing the keypoints that matches.
We add a counter on this FLANN matcher to count the number of keypoints that match between two images. Then we compare this counter with a threshold that we fixed to 200 in our experience.

### Algorithm
The main algorithm consists in create the first cluster with the first image.
We need to store the descriptor in this cluster.
Then we browse to the second image and we compare its descriptor to the first cluster.
In the case, the two previous descriptor match, we will add the image on the first cluster.
In the other case, we will add the second image and its descriptor into a new cluster.
We iterate this process until there was no available image.

At the end of this process, we should obtains directories representing each clusters.
In each directories, we should see one descriptor with the duplicates images.

### Context
We need to add the image we want to analysis in the directory dataset_jpg with the jpg format.

### Implementation
To implement those algorithms, we choose to use a class called ImageRetrieveLocal.
This class contains all the method we will use to implement the local descriptor.
The methods use are :
- incr_nb_cluster : it permits to increment the number of cluster to create a new directory,
- holiday_images : this method give us the image that we will analysis,
- extract_descriptor_from_path_img : it allows us to extract a descriptor using the path of an image,
- save_des_with_img_into_new_cluster : it will create the descriptor into a new directory representing a cluster,
- get_des_from_path_des : it will give us the descriptor using the path of the correct cluster,
- compare_path_des_and_des : it will use the FLANN matcher to compare two descriptors,
- add_img_on_cluster : it will add the currebt image on the cluster we target,
- exec : it will use the algorithm previously defined.

### Local descriptor - Results
When we launch the execution using 100 images from the dataset, it is really slow.
It will take more than thirty minutes.
The fact that we need to compare each descriptor for every image prove us that more the dataset is big,
more the time of execution grow up.

We also try to integrate a postgres database to add each cluster and descriptor on it using Docker.
We could compare the speed between the database and the file storage.


## Bonus
In this part, we will use the image search engine created by Adrian R.
To use this search engine we need to do some code transformation.

### Modification
In the file "app.py" we need to add the directory where we store the images.
This directory in under the directory called "static".
Then we must change the file called "main.js" to put this directory in the "url" variable.
We also need to use the "index.py" file to generate the file "index.csv" containing the histograms.
The command will be :
python3 index.py --dataset static/dataset_jpg --index index.csv
Finnaly, we need to launch the flask application with the following line :
python3 app.py

### Bonus - Results
After lauching the flask application, we obtain a web site with four queries.
When we select an image, the search engine will be activated.
This give us the duplicate image with the associated score.
We understand that Adrian R. choose to use the Color Descriptor to match the images.

The search engine with the Color Descriptor and the histogram is more fast than our local method.


## Global descriptor

In this section, we will focus on the Bag of Visual Word method.

1) Extract keypoints and descriptor

We will extract the descriptor in SIFT using the function build_descriptor_directory_using_images.
This function permit us to extract the descriptor of the image located in dataset_jpg into a descriptor directory.

2) Clustering the features

To do that part, we need to store the descriptors in the list using the method get_list_using_descriptors_on_directory.
This list will be used with the K-means method from the OpenCV library.

3) The visual content of an image

In this last part, we need to create the Bag of Visual Words. We use the labels we found previously.
Then we need to count for each image the number of appears.
We divide this value by the number of cluster that this image was to normalize it.

### Implementation
To implement those algorithms, we choose to use a class called ImageRetrieveGlobal.
This class contains all the method we will use to implement the global descriptor.
The methods use are :
- holiday_images : this method give us the image that we will analysis,
- extract_descriptor_from_path_img : it allows us to extract a descriptor using the path of an image,
- build_path_des : we need this method to build the path for the descriptor,
- get_list_using_descriptors_on_directory : it permits to get the list of descriptors for the K-means,
- k_means : the wrapper using the K-means algorithm from OpenCV library with the prefill parameters.


### Global descriptor - Results

When we use the K-means algorithm, we need to choose K with a value between 100 and 500.
This way seems slow during K-means execution.
We found a new way to have a very fast execution, we need to specify the number of descriptors by image during the SIFT method.


### Usage

First we need to install the libraries using requirement.txt file.

pip install -r requirements.txt

## Local
To launch the script.
In the Local directory :

python3 ImageRetrieveLocal.py

## Bonus
To launch the app.
In Local/flask-image-search-master/app:
python3 app.py
Then in your browser 0.0.0.0:5000

## Global

To launch the script.
You will find every test on Global/Test :

python3 test_image_retrieval_global.py

## References
- Frederic, P from Shallow to Deep Representation for multimedia database. Lectures 2019.
- Adrian, R from pyimagesearch.com. Website 2014.

``` 

[1]: https://www.python.org/download/releases/3.6/
