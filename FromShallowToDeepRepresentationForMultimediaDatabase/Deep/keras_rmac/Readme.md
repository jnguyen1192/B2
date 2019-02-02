# Keras RMAC

Re-implementation of Regional Maximum Activations of Convolutions (RMAC) feature extractor for Keras, based on (Tolias et al. 2016), (Gordo et al. 2016) and (Noa et al. 2018). The architecture of the model is as in the image below:

![rmac](https://github.com/noagarcia/keras_rmac/blob/master/data/model.png?raw=true)

RMAC image from: https://github.com/noagarcia/keras_rmac

RoiPooling code from: https://github.com/yhenon/keras-spp

## Prerequisites 
This code requires Keras version 2.0 or greater.
- [Python][1] (3.6)
- [Keras][2] (2.2.4)
- [Theano][3] (0.9.0)
- [VGG16 imagenet][4]

## References

- Noa , G., & George ,V., Asymmetric Spatio-Temporal Embeddings for Large-Scale Image-to-Video Retrieval. Github 2018. 

- Tolias, G., Sicre, R., & Jégou, H. Particular object retrieval with integral max-pooling of CNN activations. ICLR 2016.

- Gordo, A., Almazán, J., Revaud, J., & Larlus, D. Deep image retrieval: Learning global representations for image search. ECCV 2016. 


## Citation

This code is a re-implementation of RMAC for Keras. 

If using this code, please cite the paper where the re-implementation is used and the original RMAC paper:

```
@article{garcia2018asymmetric,
   author    = {Noa Garcia and George Vogiatzis},
   title     = {Asymmetric Spatio-Temporal Embeddings for Large-Scale Image-to-Video Retrieval},
   booktitle = {Proceedings of the British Machine Vision Conference},
   year      = {2018},
}
``` 
```
@article{tolias2016particular,
   author    = {Tolias, Giorgos and Sicre, Ronan and J{\'e}gou, Herv{\'e}},
   title     = {Particular object retrieval with integral max-pooling of CNN activations},
   booktitle = {Proceedings of the International Conference on Learning Representations},
   year      = {2016},
}
``` 

[1]: https://www.python.org/download/releases/3.6/
[2]: https://keras.io/
[3]: http://deeplearning.net/software/theano_versions/0.9.X/
[4]: https://keras.io/applications/
