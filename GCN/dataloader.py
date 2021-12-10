import numpy as np
import glob
import torch.utils.data
import torch

class MultiviewPoint(torch.utils.data.Dataset):
    def __init__(self, root_dir, num_views=20, shuffle=True):
        self.classnames = ['airplane', 'bathtub', 'bed', 'bench', 'bookshelf', 'bottle', 'bowl', 'car', 'chair',
                           'cone', 'cup', 'curtain', 'desk', 'door', 'dresser', 'flower_pot', 'glass_box',
                           'guitar', 'keyboard', 'lamp', 'laptop', 'mantel', 'monitor', 'night_stand',
                           'person', 'piano', 'plant', 'radio', 'range_hood', 'sink', 'sofa', 'stairs',
                           'stool', 'table', 'tent', 'toilet', 'tv_stand', 'vase', 'wardrobe', 'xbox']

        self.root_dir = root_dir
        self.num_views = num_views
        set_ = root_dir.split('/')[-1]
        parent_dir = root_dir.rsplit('/', 2)[0]
        self.filepaths = []

        for i in range(len(self.classnames)):
            all_files = sorted(glob.glob(parent_dir + '/' + self.classnames[i] + '/' + set_ + '/*.pth'))
            stride = int(20 / self.num_views)
            all_files = all_files[::stride]
            self.filepaths.extend(all_files)


        if shuffle == True:
            rand_idx = np.random.permutation(int(len(self.filepaths) / num_views))
            filepaths_new = []
            for i in range(len(rand_idx)):
                filepaths_new.extend(self.filepaths[rand_idx[i] * num_views:(rand_idx[i] + 1) * num_views])
            self.filepaths = filepaths_new

    def __len__(self):
        return int(len(self.filepaths) / self.num_views)

    def __getitem__(self, idx):
        path = self.filepaths[idx * self.num_views]
        class_name = path.split('/')[-3]
        class_id = self.classnames.index(class_name)
        all_point_set = []
        for i in range(20):
            point_set = torch.load(self.filepaths[idx * self.num_views + i])
            point_set = point_set.squeeze()
            all_point_set.append(torch.tensor(point_set))
        return (class_id, torch.stack(all_point_set), self.filepaths[idx * self.num_views:(idx + 1) * self.num_views])
