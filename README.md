# Code for PointView-GCN [ICIP2021].
Seyed Saber Mohammadi, Yiming Wang, Alessio Del Bue. *PointView-GCN: 3D shape classification with multi-view point clouds.* You can find IEEE version of the paper [here](https://ieeexplore.ieee.org/document/9506426).

# Citation
If you find our work useful in your research, please consider citing:
```
@inproceedings{mohammadi2021pointview,
  title={Pointview-GCN: 3D Shape Classification With Multi-View Point Clouds},
  author={Mohammadi, Seyed Saber and Wang, Yiming and Del Bue, Alessio},
  booktitle={2021 IEEE International Conference on Image Processing (ICIP)},
  pages={3103--3107},
  year={2021},
  organization={IEEE}
}
```

# Dataset
You can find our dataset with partial [single-view PCDs](https://drive.google.com/file/d/1Z-Te9Vw_PhQDCIc_zxyemwiBjI-BeBLK/view?usp=sharing) generated from benchmark dataset ModelNet40. Plese download the dataset, creat a directory named *"single_view_modelnet"* and put it under *"data"* directory.

# Training 

First use the pre-trained model to extract the features from each single-view PCD:

```
cd Feature_extraction
python main.py
```

Then apply the GCN to aggregate and classify the features:
```
cd GCN
python main.py
```

You can also train the backbone from scratch:
```
cd PointNet++
python main.py
```

# Dataset generation
First download the normalize version of ModelNet40 dataset [ModelNet40_normalized](https://drive.google.com/drive/my-drive) and put it under the "data" directory. then run the following comment: 
```
cd dataset_rendering
python dataset_capturing.py --out-split-dir /train/ && python dataset_capturing.py --out-split-dir /test/

```
Note that, since the dataset generation takes a huge amount of the time, we provided the final version of the generated [single-view PCDs](https://drive.google.com/file/d/1Z-Te9Vw_PhQDCIc_zxyemwiBjI-BeBLK/view?usp=sharing).


