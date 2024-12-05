# [ECCV 2024] Random Walk on Pixel Manifolds for Anomaly Segmentation of Complex Driving Scenes (RWPM)


> [**ECCV'24**] [**Random Walk on Pixel Manifolds for Anomaly Segmentation of Complex Driving Scenes**](https://arxiv.org/abs/2404.17961) 
> 
> by [Zelong Zeng*](https://zelongzeng.github.io/), Kaname Tomite

<div align="center">
  <img src="assets/visualization.png" width="100%" height="100%"/>
</div><br/>

## Update
**05 Dec 2024**
* We share the presentation 🎬[video](https://www.youtube.com/watch?v=pQvQkbjaDeM) of our work **RWPM**. 
**17 July 2024**
* We share the code of our work **RWPM**. 

## Notice
In this work, we only implemented RWPM on the [RbA](https://kuis-ai.github.io/RbA/) framework. If you want to reproduce our RWPM on other frameworks, please refer to lines 146 to 176 in ```RWPM/mask2former/modeling/meta_arch/mask_former_head.py``` and modify the corresponding parts for your application framework.

## Installation

See [installation instructions](INSTALL.md) for necessary installations and setup


## Datasets Preparation

See [Dataset Preparation](datasets/README.md) for details on downloading and preparing datasets for the evaluation.

## Model Zoo and Baselines

We use the checkpoints files of RbA as the baselines in this project. Refer to the [RbA Model Zoo](MODEL_ZOO.md) for more information. All RbA based experiments in our paper are used the checkpoint named [RbA + COCO Outlier Supervision](https://drive.google.com/file/d/1d5blruLB0ll6vtGAfvRH1iID6ArclWKD/view?usp=sharing) in the [RbA Model Zoo](MODEL_ZOO.md). 

## Evaluation

  We provide `evaluate_ood.py` for evaluating on OoD datasets. A simple usage for the script is as follows:

  ```
  python evaluate_ood.py 
    --model_mode selective \ # evaluates the selected models in the models_folder
    --selected_models swin_b_1dl_rba_ood_coco \ # the pre-trained model used for evaluation
    --models_folder ckpts/ \
    --datasets_folder PATH_TO_DATASETS_ROOT \
    --dataset_mode selective \ # evaluate on the selective datasets 
    --selected_datasets road_anomaly \ # the selective datasets 
    --RWPM 1 \ # the flag of RWPM, set 1 to use RWPM, set 0 to not use RWPM
    --CALIBRATION 1 \ # the flag of CALIBRATION, set 1 to use CALIBRATION, set 0 to not use CALIBRATION
    --alpha 0.99 \ # the transition probability of RWPM
    --TT 5 \ # the imeration number of limited iteration strategy
    --temperture 0.01 \ # the temperture of Softmax
    --n_patrion 2 # the partitioning parameter 
  ```

  The script assumes the following:

  * The OoD datasets are setup as described in [Datasets Prepration](datasets/README.md)
  * The parameter `--models_folder` is a path to a folder that contains multiple folders, where each folder corresponds to a model. In a model's folder the scripts expects to files: 1) `config.yaml` and its checkpoint 2) `model_final.pth`. Setting up the models is explained in [RbA Model Zoo Introduction](MODEL_ZOO.md#introduction)

  The scripts supports more finegrained options like selecting subsets of the models in a folder or the datasets. Please check `evaluate_ood.py` for descriptions of the options.


## Acknowledgement
This code is adapted from [RbA](https://kuis-ai.github.io/RbA/). Many thanks for their great work. 

## Citation
If you find this repository helpful for your research, please consider citing our paper: 

```bibtex
@inproceedings{zeng2025random,
  title={Random Walk on Pixel Manifolds for Anomaly Segmentation of Complex Driving Scenes},
  author={Zeng, Zelong and Tomite, Kaname},
  booktitle={European Conference on Computer Vision},
  pages={306--323},
  year={2025},
  organization={Springer}
}
```
