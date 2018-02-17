## _SqueezeSeg_: Convolutional Neural Nets with Recurrent CRF for Real-Time Road-Object Segmentation from 3D LiDAR Point Cloud

By Bichen Wu, Alvin Wan, Xiangyu Yue, Kurt Keutzer (UC Berkeley)

This repository contains a tensorflow implementation of SqueezeSeg, a convolutional neural network model for LiDAR segmentation. A demonstration of SqueezeSeg can be found below:

<p align="center">
    <img src="https://github.com/BichenWuUCB/SqueezeSeg/raw/master/readme/pr_0005.gif" width="600" />
</p>


Link to the paper: https://arxiv.org/abs/1710.07368. Link to the video introduction:https://youtu.be/Xyn5Zd3lm6s. If you find this work useful for your research, please consider citing:

    @article{wu2017squeezeseg,
        title={Squeezeseg: Convolutional neural nets with recurrent crf for real-time road-object segmentation from 3d lidar point cloud},
        author={Wu, Bichen and Wan, Alvin and Yue, Xiangyu and Keutzer, Kurt},
        journal={arXiv preprint arXiv:1710.07368},
        year={2017}
    }

## License
SqueezeSeg is released under the BSD license (See LICENSE for details). The dataset used for training, evaluation, and demostration of SqueezeSeg is modified from [KITTI](http://www.cvlibs.net/datasets/kitti/) raw dataset. For your convenience, we provide links to download the modified dataset, which is distrubited under the [Creative Commons Attribution-NonCommercial-ShareAlike 3.0 License](https://creativecommons.org/licenses/by-nc-sa/3.0/).

## Installation:

The instructions are tested on Ubuntu 16.04 with python 2.7 and tensorflow 1.0 with GPU support. 
- Clone the SqueezeSeg repository:
    ```Shell
    git clone https://github.com/BichenWuUCB/SqueezeSeg.git
    ```
    We name the root directory as `$SQSG_ROOT`.

- Setup virtual environment.
    1. By default we use Python2.7. Create the virtual environment
        ```Shell
        virtualenv env
        ```

    2. Activate the virtual environment
        ```Shell
        source env/bin/activate
        ```

- Use pip to install required Python packages:
    ```Shell
    pip install -r requirements.txt
    ```

## Demo:

## Training/Validation

