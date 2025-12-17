import json
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from itertools import repeat
from collections import OrderedDict
import matplotlib.pyplot as plt
import logging
import math

class OutputSaver:
    """
    Save the prediction and the ground truth to a file
    """
    def __init__(self):
        self.all_predictions = []
        self.all_targets = []
        self.lens_dirs = []
        self.data_dict = {}
        self.classify_data = {}

    def update(self, predictions, targets, info):
        self.all_predictions.append((predictions * 20).cpu().numpy())
        self.all_targets.append((targets * 20).cpu().numpy())
        self.lens_dirs.append(info['lens_dir'])

    def save(self, save_file):
        assert save_file is not None, "Please provide a save file"
        all_predictions = np.concatenate(self.all_predictions, axis=0)
        all_targets = np.concatenate(self.all_targets, axis=0)
        absolute_errors = np.abs(all_predictions - all_targets)
        data_dict = {
            'predict': all_predictions.tolist(),
            'truth': all_targets.astype(int).tolist(),
            'absolute_errors': absolute_errors.tolist(),
            'lens_dirs':  [item for sublist in self.lens_dirs for item in sublist] 
        }
        self.data_dict = data_dict
        with open(save_file, 'w') as json_file:
            json.dump(data_dict, json_file, indent=2)
        logging.info(f'Saved data to {save_file}')
    
    def save_classify(self, nums, save_file):
        """
         "lens_dirs": [
                "/home/SYnpy/SY2/4/SHIFT/-15 -15",
                "/home/SYnpy/SY2/4/SHIFT/-15 -14",
                ...
            ]
        """
        data_dict = self.data_dict
        # 按照 lens_dirs 中前 nums 个目录进行分类
        lens_dict = {}
        for i in range(len(data_dict["lens_dirs"])):
            lens_dir = data_dict["lens_dirs"][i]
            # 去除最后一个目录
            lens_dir = lens_dir.split("/")[:-1]
            lens_dir = "/".join(lens_dir)
            if lens_dir not in lens_dict:
                lens_dict[lens_dir] = {
                    "predict":[],
                    "truth":[],
                    "absolute_errors":[]
                }
            lens_dict[lens_dir]["predict"].append(data_dict["predict"][i])
            lens_dict[lens_dir]["truth"].append(data_dict["truth"][i])
            lens_dict[lens_dir]["absolute_errors"].append(data_dict["absolute_errors"][i])
        self.classify_data = lens_dict
        with open(save_file, 'w') as json_file:
            json.dump(lens_dict, json_file, indent=2)
        logging.info(f'Saved classified data to {save_file}')
    
    def save_classify_by_truth(self, save_file):
        classify_data = self.classify_data
        classify_dict = {}
        for lens_dir in classify_data:
            classify_dict[lens_dir] = {}
            data = classify_data[lens_dir]
            for i in range(len(data["truth"])):
                truth = data["truth"][i]
                # 把 truth 转为字符串
                truth = " ".join(map(str, truth))
                classify_dict[lens_dir][truth] = {
                    "predict":data["predict"][i],
                    "absolute_errors":data["absolute_errors"][i]
                }
        
        with open(save_file, 'w') as json_file:
            json.dump(classify_dict, json_file, indent=2)
        logging.info(f'Saved classified by truth data to {save_file}')