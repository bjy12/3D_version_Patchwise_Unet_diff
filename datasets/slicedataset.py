import os
import json
import yaml
import scipy
import pickle
import numpy as np
import pdb
import SimpleITK as sitk
from torch.utils.data import Dataset
import random
import glob
from copy import deepcopy

def get_filesname_from_txt(txt_file_path):
    files = []
    with open(txt_file_path, 'r') as f:
        file_name = f.readlines()
        for file in file_name:
            file_name = file.strip()
            #file_path = os.path.join(base_dir, file_name)
            files.append(file_name)   

    return files


def sitk_load(path, uint8=False, spacing_unit='mm'):
    # load as float32
    itk_img = sitk.ReadImage(path)
    #pdb.set_trace()
    spacing = np.array(itk_img.GetSpacing(), dtype=np.float32)
    if spacing_unit == 'm':
        spacing *= 1000.
    elif spacing_unit != 'mm':
        raise ValueError
    image = sitk.GetArrayFromImage(itk_img)
    image = image.transpose(2, 1, 0) # to [x, y, z]
    image = image.astype(np.float32)
    if uint8:
        # if data is saved as uint8, [0, 255] => [0, 1]
        image /= 255.
    return image, spacing


class SliceDataset(Dataset):
    def __init__(self,
                 root ,
                 files_list, 
                 mode = 'train'):
        super().__init__()

        self.data_root = root
        files_list = files_list
        #pdb.set_trace()
        name_list = get_filesname_from_txt(files_list)
        #pdb.set_trace()
        random.shuffle(name_list)
        self.path_list = []
        for name in name_list:
            pattern = f"{name}_*_*.npz"
            search_path = os.path.join(self.data_root, pattern)
            matched_files = glob.glob(search_path)
            self.path_list.extend(matched_files)
        self._sort_paths_by_group()
        #pdb.set_trace()
        #random.shuffle(self.path_list)
        #pdb.set_trace()
    def __len__(self):
        return len(self.path_list)
    
    def _extract_info(self, filename):
        """从文件名中提取排序信息"""
        name = filename.split('/')[-1]
        parts = name.split('_')
        case_num = int(parts[0])      # 病例编号
        axis = parts[1]               # 轴向
        slice_num = int(parts[2].split('.')[0])  # 切片编号
        
        return case_num, axis, slice_num

    def _sort_paths_by_group(self):
        """按轴向分组排序"""
        # 创建不同轴向的文件列表
        axial_files = []
        sagittal_files = []
        coronal_files = []
        
        # 分组
        for path in self.path_list:
            case_num, axis, slice_num = self._extract_info(path)
            if axis == 'axial':
                axial_files.append(path)
            elif axis == 'sagittal':
                sagittal_files.append(path)
            elif axis == 'coronal':
                coronal_files.append(path)
        
        # 对每个组分别排序
        axial_files.sort(key=lambda x: (int(x.split('/')[-1].split('_')[0]), 
                                      int(x.split('/')[-1].split('_')[-1].split('.')[0])))
        sagittal_files.sort(key=lambda x: (int(x.split('/')[-1].split('_')[0]), 
                                         int(x.split('/')[-1].split('_')[-1].split('.')[0])))
        coronal_files.sort(key=lambda x: (int(x.split('/')[-1].split('_')[0]), 
                                        int(x.split('/')[-1].split('_')[-1].split('.')[0])))
        
        # 按顺序组合所有文件
        self.path_list = axial_files + sagittal_files + coronal_files

    def __getitem__(self, index):
        path = self.path_list[index]
        # pdb.set_trace()
        name = path.split('/')[-1].split('_')[0]
        axes = path.split('/')[-1].split('_')[1]
        slice_number = path.split('/')[-1].split('_')[-1].split('.')[0]
        npz = np.load(path)
        #pdb.set_trace()
        value  = npz['intensity']
        value  = value.astype(np.float32) / 255.
        value  = value[None,...]
        coords = npz['coordinates']
        # 
        value = (value - 0.5) * 2. # [-1,1]
        # 
        coords = (coords - 0.5) * 2.  #[-1,1]

        ret_dict ={
            'name' : name ,
            'axes' : axes,
            'slice_number' : slice_number,
            'value' : value,
            'coords' : coords,
        }

        return ret_dict


def load_slice_dataset(img_root , files_list):
    dataset = SliceDataset(img_root , files_list)
    ds = dataset[0]
    return dataset