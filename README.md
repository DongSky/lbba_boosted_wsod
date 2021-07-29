# LBBA-boosted WSOD

## Summary

Our code is based on [ruotianluo/pytorch-faster-rcnn](https://github.com/ruotianluo/pytorch-faster-rcnn) and [WSCDN](https://github.com/Sunarker/Collaborative-Learning-for-Weakly-Supervised-Object-Detection)

Sincerely thanks for your resources.

Newer version of our code (based on Detectron 2) work in progress.

## Hardware

We use one RTX 2080Ti GPU (11GB) to train and evaluate our method, GPU with larger memory is better (e.g., TITAN RTX with 24GB memory)

## Requirements
- Python 3.6 or higher
- CUDA 10.1 with cuDNN 7.6.2
- PyTorch 1.2.0
- numpy 1.18.1
- opencv 3.4.2

We provide a full requirements.txt (namely lbba\_requirements.txt) in the workspace (lbba\_boosted\_wsod directory).


## Additional resources
[Google Drive](https://drive.google.com/drive/folders/14HEC4LMWtS_0sbEf281IrcTKu6cq0jVY?usp=sharing)

#### Description

- selective\_search\_data: precomputed proposals of VOC 2007/2012
- pretrained\_models/imagenet\_pretrain: imagenet pretrained models of WSOD backbone/LBBA backbone
- pretrained\_models/pretrained\_on\_wsddn: pretrained WSOD network of VOC 2007/2012, using this pretrained model usually converges faster and more stable.
- models/voc07: our pretrained WSOD
- models/lbba: our pretrained LBBA
- codes\_zip: our code template of LBBA training procedure and LBBA\-boosted WSOD training procedure


## Prepare


#### Environment
We use Anaconda to construct our experimental environment.

Install all required packages (or simply follow lbba\_requirements.txt).

#### Essential Data
We have initialized all directories with gitkeep files. 

first, cd lbba\_boosted\_wsod

then, download selective\_search\_data/* into data/selective\_search\_data

download pretrained\_models/imagenet\_pretrain/* into data/imagenet\_weights

download pretrained\_models/pretrained\_on\_wsddn/* into data/wsddn\_weights

#### Datasets

Same with [rbgirshick/py-faster-rcnn](https://github.com/rbgirshick/py-faster-rcnn#beyond-the-demo-installation-for-training-and-testing-models)

For example, PASCAL VOC 2007 dataset

1. Download the training, validation, test data and VOCdevkit

	```Shell
	wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar
	wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtest_06-Nov-2007.tar
	wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCdevkit_08-Jun-2007.tar
	```

2. Extract all of these tars into one directory named `VOCdevkit`

	```Shell
	tar xvf VOCtrainval_06-Nov-2007.tar
	tar xvf VOCtest_06-Nov-2007.tar
	tar xvf VOCdevkit_08-Jun-2007.tar
	```

3. It should have this basic structure

	```Shell
  	$VOCdevkit/                           # development kit
  	$VOCdevkit/VOCcode/                   # VOC utility code
  	$VOCdevkit/VOC2007                    # image sets, annotations, etc.
  	# ... and several other directories ...
  	```

4. Create symlinks for the PASCAL VOC dataset

	```Shell
    cd $FRCN_ROOT/data
    ln -s $VOCdevkit VOCdevkit2007
    ```

## Evaluate our WSOD

Download models/voc07/voc07\_55.8.pth to lbba\_boosted\_wsod/

```
./test_voc07.sh 0 pascal_voc vgg16 voc07_55.8.pth
```

Note that different environments might result in a slight performance drop. For example, we obtain 55.8 mAP with CUDA 10.1 but obtain 55.5 mAP using the same code with CUDA 11.

## Train WSOD

Download models/lbba/lbba_final.pth (or lbba_init.pth) to lbba\_boosted\_wsod/

```
bash train_wsod.sh 1 pascal_voc vgg16 voc07_wsddn_pre lbba_final.pth
```

Note that we provide different LBBA checkpoints (initialization stage, final stage, or even one-class adjuster mentioned in the suppl.).