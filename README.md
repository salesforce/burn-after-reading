# CroDoBo

## Introduction

- Pytorch Implementation of  [Burn After Reading: Online Adaptation for Cross-domain Streaming Data](https://arxiv.org/pdf/2112.04345.pdf)

## Environment

- Code is developed with cuda 11.4, python 3.8.5

## Requirements

- torch 1.8.1
- wilds 1.1.0
- pillow

## Dataset Preparation
- [VisDA-C](http://csr.bu.edu/ftp/visda17/clf/) </br>
- [COVID-DA](https://github.com/qiuzhen8484/COVID-DA)</br>
- [WILDS](https://github.com/p-lambda/wilds)</br>
- [Fashion-MNIST](https://github.com/zalandoresearch/fashion-mnist)</br>
- [DeepFashion](http://mmlab.ie.cuhk.edu.hk/projects/DeepFashion/AttributePrediction.html)</br>
Prepare the randomly perturbated data by `python gen_random_perm.py`


## Training
`python main_crodobo.py --backbone FEATURENETWORK --dataset DATASETOPTION --data_root PATH/TO/YOUR/DATASET`

## Contact
- Please email `loyo@umd.edu` or `cramaiah@salesforce.com` if you have any questions.

## Citation
If you find this codebase useful, please cite our paper:

``` latex
@article{yang2021burn,
  title={Burn After Reading: Online Adaptation for Cross-domain Streaming Data},
  author={Yang, Luyu and Gao, Mingfei and Chen, Zeyuan and Xu, Ran and Shrivastava, Abhinav and Ramaiah, Chetan},
  journal={arXiv preprint arXiv:2112.04345},
  year={2021}
}
```

## Acknowledgement
- We referred to [SHOT](https://github.com/tim-learn/SHOT) for the implementation.

## License
Our code is BSD-3 licensed. See LICENSE.txt for details.
