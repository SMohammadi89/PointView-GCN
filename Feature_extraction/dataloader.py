import numpy as np
import glob
import open3d as o3d
import torch.utils.data
import torch
import os


class SinglePoint(torch.utils.data.Dataset):
    def __init__(self, root_dir):
        # self.classnames =  ['bag','bed','bin','box','cabinet', 'chair', 'desk','display', 'door','pillow', 'shelf', 'sink', 'sofa','table', 'toilet']
        self.classnames = ['airplane', 'bathtub', 'bed', 'bench', 'bookshelf', 'bottle', 'bowl', 'car', 'chair',
                           'cone', 'cup', 'curtain', 'desk', 'door', 'dresser', 'flower_pot', 'glass_box',
                           'guitar', 'keyboard', 'lamp', 'laptop', 'mantel', 'monitor', 'night_stand',
                           'person', 'piano', 'plant', 'radio', 'range_hood', 'sink', 'sofa', 'stairs',
                           'stool', 'table', 'tent', 'toilet', 'tv_stand', 'vase', 'wardrobe', 'xbox']

        self.root_dir = root_dir
        set_ = root_dir.split('/')[-1]
        parent_dir = root_dir.rsplit('/', 2)[0]
        self.filepaths = []
        for i in range(len(self.classnames)):
            all_files = sorted(glob.glob(parent_dir + '/' + self.classnames[i] + '/' + set_ + '/*.xyz'))

            self.filepaths.extend(all_files)
    def __len__(self):
        # print(len(self.filepaths))
        return len(self.filepaths)

    def __getitem__(self, idx):
        path = self.filepaths[idx]
        class_name = path.split('/')[-3]
        name = os.path.split(path)[-1]
        name = name[:-4]
        class_id = self.classnames.index(class_name)
        point_set = np.loadtxt(self.filepaths[idx])

        return (point_set,class_id,class_name,path,path)

