import numpy as np
# import tensorflow as tf
import torch
from pathlib import Path
import torch.utils.data as data

# from .base_dataset import BaseDataset
from settings import DATA_PATH, EXPER_PATH
from utils.tools import dict_update
import cv2
from utils.utils import homography_scaling_torch as homography_scaling
from utils.utils import filter_points
from datasets.Coco import Coco


class Duck(Coco):
    default_config = {
        'labels': None,
        'cache_in_memory': False,
        'validation_size': 100,
        'truncate': None,
        'preprocessing': {
            'resize': None
        },
        'num_parallel_calls': 10,
        'augmentation': {
            'photometric': {
                'enable': False,
                'primitives': 'all',
                'params': {},
                'random_order': True,
            },
            'homographic': {
                'enable': False,
                'params': {},
                'valid_border_margin': 0,
            },
        },
        'warped_pair': {
            'enable': False,
            'params': {},
            'valid_border_margin': 0,
        },
        'homography_adaptation': {
            'enable': False
        }
    }

    def __init__(self, export=False, transform=None, task='train', **config):

        # Update config
        self.config = self.default_config
        self.config = dict_update(self.config, config)

        self.transforms = transform
        self.action = 'train' if task == 'train' else 'val'

        # get files
        base_path = Path(DATA_PATH, 'Duck/' + task + 'Duck/'+self.config["type_data"]+"/")
        # base_path = Path(DATA_PATH, 'COCO_small/' + task + '2014/')
        image_paths = list(base_path.iterdir())
        # if config['truncate']:
        #     image_paths = image_paths[:config['truncate']]
        names = [p.stem for p in image_paths]
        image_paths = [str(p) for p in image_paths]
        files = {'image_paths': image_paths, 'names': names}


        sequence_set = []
        # labels
        self.labels = False
        if self.config['labels']:
            self.labels = True
            # from models.model_wrap import labels2Dto3D
            # self.labels2Dto3D = labels2Dto3D
            print("load labels from: ", self.config['labels']+'/'+task)
            count = 0
            for (img, name) in zip(files['image_paths'], files['names']):
                p = Path(self.config['labels'], task, '{}.npz'.format(name))
                if p.exists():
                    sample = {'image': img, 'name': name, 'points': str(p)}
                    sequence_set.append(sample)
                    count += 1
                # if count > 100:
                #     print ("only load %d image!!!", count)
                #     print ("only load one image!!!")
                #     print ("only load one image!!!")
                #     break
            pass
        else:
            for (img, name) in zip(files['image_paths'], files['names']):
                sample = {'image': img, 'name': name}
                sequence_set.append(sample)
        self.samples = sequence_set

        self.init_var()

        pass