# DIPR Starter Code


This repository contains starter code for DIPR. The code adds several 2D disks into an image and reshades it. To insert new objects, modify the object path and related code accordingly.

```
python dipr.py --img <image_path> --outdir <output directory path>
```

The current code base uses an old [normal estimation](https://github.com/DrSleep/multi-task-refinenet) and an old intrinsic image model. If you use the current best state-of-the-art image decomposition or normal estimator, then you might have to retrain the shading consistency discriminator with those predictions.

## Citation
If you use this code or ideas from our paper, please cite our paper:
```
@inproceedings{bhattad2022cut,
  title={Cut-and-Paste Object Insertion by Enabling Deep Image Prior for Reshading},
  author={Bhattad, Anand and Forsyth, DA},
  booktitle={2022 International Conference on 3D Vision (3DV)},
  pages={332--341},
  year={2022},
  organization={IEEE}
}
```