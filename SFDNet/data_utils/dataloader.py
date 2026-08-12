import os
import json
import warnings
import numpy as np
from torch.utils.data import Dataset
warnings.filterwarnings('ignore')


def pc_normalize(pc):
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc ** 2, axis=1)))
    pc = pc / m
    return pc


class SemanticDataset(Dataset):
    def __init__(self, root='dataset/ABC/noise_none',
                 split='train', npoints=None, normal_channel=False, add_chn=0):
        if split not in ('train', 'val', 'test'):
            raise ValueError('Unknown split: %s' % (split,))

        self.root = root
        self.split = split
        self.npoints = npoints
        self.normal_channel = normal_channel
        self.additional_channel = add_chn

        json_file = os.path.join(self.root, 'train_test_split',
                                 'shuffled_%s_file_list.json' % split)
        with open(json_file, 'r') as f:
            file_list = json.load(f)
        self.datapath = [os.path.join(self.root, f + '.txt') for f in file_list]

        self.cache = {}
        self.cache_size = 20000

        if split == 'train':
            self.labelweights = np.zeros(2, dtype=np.float32)
            for fn in self.datapath:
                try:
                    data = np.loadtxt(fn).astype(np.float32)
                except Exception as e:
                    print("Error reading file {}: {}".format(fn, e))
                    continue
                seg = data[:, -1].astype(np.int32)
                tmp, _ = np.histogram(seg, bins=[0, 1, 2])
                self.labelweights += tmp
            self.labelweights = self.labelweights / np.sum(self.labelweights)
            self.labelweights = np.power(np.amax(self.labelweights) / self.labelweights, 1 / 3.0)
            print("Computed label weights:", self.labelweights)

    def __getitem__(self, index):
        if index in self.cache:
            point_set, seg = self.cache[index]
        else:
            fn = self.datapath[index]
            data = np.loadtxt(fn).astype(np.float32)
            if not self.normal_channel:
                point_set = data[:, 0:3]
            else:
                point_set = data[:, 0:3 + self.additional_channel]
            seg = data[:, -1].astype(np.int32)
            if len(self.cache) < self.cache_size:
                self.cache[index] = (point_set.copy(), seg.copy())

        point_set[:, 0:3] = pc_normalize(point_set[:, 0:3])

        if self.npoints is not None:
            choice = np.random.choice(len(seg), self.npoints, replace=True)
            point_set = point_set[choice, :]
            seg = seg[choice]

        return point_set, seg

    def __len__(self):
        return len(self.datapath)
