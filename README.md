# Embedding pre-training for Few-shot Learning

In this repo, we provide a collection of scripts for pretraining embedding models for few-shot learning, some of them are used in [Label Hallucination for Few-Shot Classification](https://github.com/yiren-jian/LabelHalluc) (in [Dropbox](https://www.dropbox.com/sh/ikipligbneta9qk/AABew7LSYDMG7lbSC9BQgcMsa?dl=0)). Thanks [SKD](https://github.com/brjathu/SKD) and [Rizve et al.](https://github.com/nayeemrizve/invariance-equivariance) for their original implementation.

## TL, DR
I have updated their original code a bit so that you should run freely (latest pytorch and fix some bugs in SKD) on you machine following the training code. Newly re-trained models for tiered-ImageNet is [here](https://www.dropbox.com/sh/gxmu8d75a9grfph/AABOPOoTZmu2wnKnLYL1AKv9a?dl=0). I will keep updating it when more models are trained.

I have re-trained SKD and IER models. SKD models were trained with 4 GTX-Titan X and IER models were trained with 2 Nvidia-A100.
- [x] SKD Generation 0
- [x] SKD Generation 1 (`gamma=0.025` works best)
- [x] IER (`bsz=64`)
- [x] IER distill (`bsz=64`)

## Download Data
The data we used here is preprocessed by the repo of [MetaOptNet](https://github.com/kjunelee/MetaOptNet), Please find the renamed versions of the files in below link by [RFS](https://github.com/WangYueFt/rfs). Download and unzip the dataset, put them under ```data``` directory.

Note that **training with tiered-ImageNet requires at least 64GB of CPU RAM, ideally 128GB.**

## Install requirements
First creating a conda environment named IER
```
conda create -n IER python=3.6  ### create envs
conda activate IER    ### launch envs  
pip install -r requirements.txt   ### install requirements by IER, except for pytorch
```
For the GPU we use (RTX-A100 or A6000), it requires pytorch installation of CUDA11, thus further reinstall pytorch by
```
conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch
```

## Invariant and Equivariant Representations (IER)
We modify the `train.py` mainly to remove wandb usage and save the model every epoch to disk. Further to work with the latest pytorch version, we change the `.view()` function in `utils` to `.reshape()`.

training the model follow the original implementation (e.g. for FC100)
```
python3 train.py --model resnet12 --model_path save --dataset FC100 --data_root data --n_aug_support_samples 5 --n_ways 5 --n_shots 1 --epochs 65 --lr_decay_epochs 60 --gamma 1.0 --contrast_temp 1.0 --mvavg_rate 0.99 --memfeature_size 64 --batch_size 64 --tags FC100,INV_EQ
```

## Self-supervised Knowledge Distillation (SKD)
We use the same environment as IER. To make the code run, we have to make a few modifications and corrections, i.e, in orginal `train_distillation.py`
```
# inputs_all = torch.cat((x, x_180, x_90, x_270),0)
inputs_all = torch.cat((x, x_90),0)

# (_,_,_,_, feat_s_all), (logit_s_all, rot_s_all)  = model_s(inputs_all[:4*batch_size], rot=True)
(_,_,_,_, feat_s_all), (logit_s_all, rot_s_all)  = model_s(inputs_all[:2*batch_size], rot=True)

# loss = loss_div + opt.gamma*loss_a / 3
loss = loss_div + opt.gamma*loss_a 
```

## Note
We found that results vary (and sometimes a lot) across different runs. To try to get models matching results for what reported in these papers, we found that it's important to have multiple runs of the initial training (generation 0 in IER and SKD) and pick the best model to start with, for distillation (generation 1). 

Please consider citing the paper of SKD and IER

```bibtex
@InProceedings{Rizve_2021_CVPR,
    author    = {Rizve, Mamshad Nayeem and Khan, Salman and Khan, Fahad Shahbaz and Shah, Mubarak},
    title     = {Exploring Complementary Strengths of Invariant and Equivariant Representations for Few-Shot Learning},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2021},
    pages     = {10836-10846}
}
```

```bibtex
@article{rajasegaran2020self,
  title={Self-supervised Knowledge Distillation for Few-shot Learning},
  author={Rajasegaran, Jathushan and Khan, Salman and Hayat, Munawar and Khan, Fahad Shahbaz and Shah, Mubarak},
  journal={https://arxiv.org/abs/2006.09785},
  year = {2020}
}
```

It would be also nice if you consider reading our latest work on FSL :)
```bibtex
@inproceedings{jian2022label,
  title={Label hallucination for few-shot classification},
  author={Jian, Yiren and Torresani, Lorenzo},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={36},
  number={6},
  pages={7005--7014},
  year={2022}
}
```
