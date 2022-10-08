## Distance Matters in Human-Object Interaction Detection
This repo contains the implementation of our ACM MM 2022 paper titled "Distance Matters in Human-Object Interaction Detection"

## Prerequist
We adopted the lightweight library [Pocket][https://github.com/fredzzhang/pocket], where pytorch 1.9.1 with cuda 11.1 are adopted. Please install this library according to the requirements.
Then, please prepare the datasets according to the following links:
* HICO-DET: https://github.com/fredzzhang/hicodet
* V-COCO: https://github.com/fredzzhang/vcoco

And then download the two datasets in the respective foloders.

Please also clone the DETR codes into this repository
https://github.com/fredzzhang/detr


## Training
Our codes run on 4 GPUs in parallel.

To train on the HICO-DET dataset, please run
```
./scripts/train_hicodet.sh
```
The evaluation results will be automatically printed after training.

To train on V-COCO, please run
```
./scripts/train_vcoco.sh
```

## Acknowledgement
Our code is built based on [UPT](https://github.com/fredzzhang/upt), and [DETR](https://github.com/facebookresearch/detr). 
Thanks them for their great work!
