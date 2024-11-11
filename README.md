# Preparations

## Cloning the Repository
The repository contains submodules, thus please check it out with 
```shell
# SSH
git clone git@github.com:nguyentran186/LucidInpainting.git --recursive
```
or
```shell
# HTTPS
git clone https://github.com/nguyentran186/LucidInpainting.git --recursive
```
## Setup
Our default, provided install method is based on Conda package.
Firstly, you need to create an virtual environment and install the submodoules we provide. (slightly difference from original [3DGS](https://github.com/graphdeco-inria/gaussian-splatting))
```shell
conda create -n LucidDreamer python=3.9.16 cudatoolkit=11.8
conda activate LucidDreamer
pip install -r requirements.txt
pip install submodules/diff-gaussian-rasterization/
pip install submodules/simple-knn/
```


## Run
```
python train.py --opt './configs/statue.yaml'
```
