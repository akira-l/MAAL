## MAAL
Homepage for MAAL: Multimodality-Aware Autoencoder-based Affordance Learning for 3D Articulated Objects

This repository is for the code and real-world experiments of MAAL.


## Insight 

* A simple AE structure is work for affordance learning. MAAL show a more efficient paradox for affordance problem. 
* Multi-modal learning strategies need to be emphasized in the learning framework. MAAL shows that the simple consideration of introduce better feature fusion already leads to improvements. 

## Environment & Training 
The environment requirements are the same to [AdaAfford](https://github.com/wangyian-me/AdaAffordCode/) (Python 3.6, Pytorch 1.7.0, SAPIEN) 

We operate the real-world experiments with a novel class-a paper box with a lid. We provide videos that the gripper operates pulling action in different directions. We also show some examples for the visual contents that captured by the RGBD camera. All contents are available at folder ‘examples’. 


## Citation

Please cite our paper if you find our MAAL is helpful in your work:

```
@inproceedings{liang2023maal,
  title={MAAL: Multimodality-Aware Autoencoder-Based Affordance Learning for 3D Articulated Objects},
  author={Liang, Yuanzhi and Wang, Xiaohan and Zhu, Linchao and Yang, Yi},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
  pages={217--227},
  year={2023}
}
```

