# Factorized diffusion autoencoder
Factorized Diffusion Autoencoder for Unsupervised Disentangled Representation Learning (AAAI 2024)  
[[paper](paper/FDAE.pdf)]
[[supplementary material](paper/supplemental.pdf)]

This repository contains the code for training and evaluation on
[Cars3d](http://www.scottreed.info/files/nips2015.pdf),
[Shapes3d](https://github.com/google-deepmind/3d-shapes)
and
[mpi3d_real_complex](https://github.com/rr-learning/disentanglement_dataset). 

## Dependencies

The implementation is based on pytorch 1.13.1+cu117 and 1 NVIDIA RTX 3090.
The required packages are listed in `requirements.txt`.
```sh
pip install -r requirements.txt
```

## Prepare data
* Cars3d   
Download [nips2015-analogy-data.tar.gz](http://www.scottreed.info/files/nips2015-analogy-data.tar.gz), extract `cars` folder and put it in `./data/cars`.  
Download [cars3d.zip](https://drive.google.com/file/d/1aCo9wD4kbY4V0cu7qFXSHF4rmJlEb_E-/view?usp=sharing) and unzip it in `./datasets/cars3d`.

* Shapes3d  
Download `3dshapes.h5` from https://console.cloud.google.com/storage/browser/3d-shapes and put it in `./data/3dshapes.h5`.  
Extract images from `3dshapes.h5` and put them in `./datasets/shapes3d`.

* mpi3d_real_complex  
Download [real3d_complicated_shapes_ordered.npz](https://drive.google.com/file/d/1Tp8eTdHxgUMtsZv5uAoYAbJR1BOa_OQm/view) and put it in `./data/real3d_complicated_shapes_ordered.npz`.  
Extract images from `real3d_complicated_shapes_ordered.npz` and put them in `./datasets/mpi3d_real_complex`. 

## Model training and testing

We provide examples of training in `train.sh`.
After training, both model checkpoint and results of evaluation metrics DCI, FactorVAE and MIG are saved in the `./logs` folder.
If computing mean and std of multiple results is required, run script `collect_result.py` (with `model_list` variable modified) to output results in `./exp_results`. 

## Visualization

`image_sample.sh` provides examples of visualizing masks and generating images by swapping content codes and mask codes.  

## Acknowledgement
We based our codes on [openai/consistency_models](https://github.com/openai/consistency_models)
and evaluation codes of [xrenaa/DisCo](https://https://github.com/xrenaa/DisCo). 

## Citation

If you find this method or code useful, please cite

```bibtex
@inproceedings{wu2024fdae,
  title={Factorized Factorized Diffusion Autoencoder for Unsupervised Disentangled Representation Learning},
  author={Wu, Ancong and Zheng, Wei-Shi},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  year={2024},
}
```