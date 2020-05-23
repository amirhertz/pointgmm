### PointGMM

This is the official Pytorch implementation of the CVPR2020 paper [PointGMM: a Neural GMM Network for Point Clouds](https://arxiv.org/pdf/2003.13326.pdf).

– Download the [ShapeNetCore.v2](https://www.shapenet.org/) dataset. <br>
&nbsp;&nbsp;  Redirect ```constants.DATASET``` to the dataset directory.

– Pre-trained models are available [here](https://drive.google.com/drive/folders/1NZT8uJYcck_CaB8v2h6HO9E9XlF3gXhT?usp=sharing) (optional). <br>
&nbsp;&nbsp;  Redirect ```constants.CHECKPOINTS_ROOT``` to the models directory.

– Train a VAE model: ```python train.py -d 0 -c airplane```. <br>
&nbsp;&nbsp; where ```d``` specify the GPU id and ```c``` specify one of the [ShapeNetCore categories](process_data/categories.txt).

– Train a registration model: ```python train.py -d 1 -c chair -r```. <br>

– Play with a pre-trained  model via ```eval_ae.py```.
