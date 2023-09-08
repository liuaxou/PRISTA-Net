# PRISTA-Net : Deep Iterative Shrinkage Thresholding Network for Coded Diffraction Patterns Phase Retrieval

Here we provide the pytorch implementation of the paper, PRISTA-Net : Deep Iterative Shrinkage Thresholding Network for Coded Diffraction Patterns Phase Retrieval.

## Abstract

The problem of phase retrieval (PR) involves recovering an unknown image from limited amplitude measurement data and is a challenge nonlinear inverse problem in computational imaging and image processing. However, many of the PR methods are based on black-box network models that lack interpretability and plug-and-play (PnP) frameworks that are computationally complex and require careful parameter tuning. To address this, we have developed PRISTA-Net, a deep unfolding network (DUN) based on the first-order iterative shrinkage thresholding algorithm (ISTA). This network utilizes a learnable nonlinear transformation to address the proximal-point mapping sub-problem associated with the sparse priors, and an attention mechanism to focus on phase information containing image edges, textures, and structures. Additionally, the fast Fourier transform (FFT) is used to learn global features to enhance local information, and the designed logarithmic-based loss function leads to significant improvements when the noise level is low. All parameters in the proposed PRISTA-Net framework, including the nonlinear transformation, threshold parameters, and step size, are learned end-to-end instead of being manually set. This method combines the interpretability of traditional methods with the fast inference ability of deep learning and is able to handle noise at each iteration during the unfolding stage, thus improving recovery quality. Experiments on Coded Diffraction Patterns (CDPs) measurements demonstrate that our approach outperforms the existing state-of-the-art methods in terms of qualitative and quantitative evaluations.

![The overall architecture of the proposed PRISTA-Net.](/PRISTA-Net_frame_v1.png)

Fig. 1. The overall architecture of the proposed PRISTA-Net.

## Environment

`python == 3.9  python == 1.12.1  torchvision = 0.13.1  opencv-python == 4.6.0  `

You can use `conda env create -f py39.yaml` to create conda environment

## Test

1.1、Pre-trained models:  
find the pre-trained model for our paper are in './model/PRISTA_CDP_Group_1_layer_num_7_lr_0.0010'.  
1.2、Prepare test data:  
The original test set is in './data/Test/TestImages_128' and './data/Test/TestImages_256'.  
1.3、Run the test script.  
`python test.py --end_epoch=200 --layer_num=7 --group_num=1 --SamplingRate=4 --measurement_type='CDP' --ResF=1 --atten=1 --shared_ResF=0 --shared_CBAM=0 --log_set=0 `

## Train

1.Download the dataset provide in the [link](https://pan.baidu.com/s/1LZfPvns4b08TWJwSdFw-8Q?pwd=ouee).  
2.Run the train script.  
`python train.py --end_epoch=200 --layer_num=7 --group_num=1 --SamplingRate=4 --measurement_type='CDP' --ResF=1 --atten=1 --shared_ResF=0 --shared_CBAM=0 --log_set=0`

## Acknowledgements

Thanks to the authors of ISTA-Net and PrDeep, our codes are adapted from the open source codes of ISTA-Net and PrDeep.

## Concat
The code is provided to support reproducible research. If the code is giving syntax error in your particular python configuration or some files are missing then you may open an issue or directly email me at (202121511147@smail.xtu.edu.cn)
