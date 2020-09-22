# You Only Watch Once (YOWO)

### UCF101-24
<br/>
<div align="center" style="width:image width px;">
  <img  src="https://github.com/christina284/yowo/tree/master/examples/biking1.gif" width=240 alt="biking">
  <img  src="https://github.com/christina284/yowo/tree/master/examples/walkingWithDog.gif" width=240 alt="walking with dog">
</div>  

<div align="center" style="width:image width px;">
  <img  src="https://github.com/christina284/yowo/tree/master/examples/skate.gif" width=240 alt="skate">
  <img  src="https://github.com/christina284/yowo/tree/master/examples/poleVault.gif" width=240 alt="pole vault">
</div>

### JHMDB
<div align="center" style="width:image width px;">
  <img  src="https://github.com/christina284/yowo/tree/master/examples/run.gif" width=240 alt="run">
</div>


PyTorch implementation of the article "[You Only Watch Once: A Unified CNN Architecture for Real-Time Spatiotemporal Action Localization](https://arxiv.org/pdf/1911.06644.pdf)".

Networks used : Darknet, Resnet18

mAP is slighlty lower than presented (probably due to training with Resnet18 instead of Resnext101): 71,34%

Dataset: UCF101-24

### Run pretrained model on UCF101-24

```bash
python myTest.py --video_path "videopath"
```

### Citation
If you use this code or pre-trained models, please cite the following:

```bibtex
@InProceedings{kopuklu2019yowo,
title={You Only Watch Once: A Unified CNN Architecture for Real-Time Spatiotemporal Action Localization},
author={K{\"o}p{\"u}kl{\"u}, Okan and Wei, Xiangyu and Rigoll, Gerhard},
journal={arXiv preprint arXiv:1911.06644},
year={2019}
}
```

