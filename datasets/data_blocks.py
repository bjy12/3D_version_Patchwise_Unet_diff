# -*- coding: utf-8 -*-

import os
import random
import numpy as np
import torch
import SimpleITK as sitk
import yaml
#from datasets.geometry import Geometry

from copy import deepcopy
from torch.utils.data import Dataset, DataLoader
from einops import rearrange
#from utils import sitk_load ,get_filesname_from_txt
import pdb
import pickle

class Geometry(object):
    def __init__(self, config):
        self.v_res = config['nVoxel'][0]    # ct scan
        self.p_res = config['nDetector'][0] # projections
        self.v_spacing = np.array(config['dVoxel'])[0]    # mm
        self.p_spacing = np.array(config['dDetector'])[0] # mm
        # NOTE: only (res * spacing) is used

        self.DSO = config['DSO'] # mm, source to origin
        self.DSD = config['DSD'] # mm, source to detector

    def project(self, points, angle):
        # points: [N, 3] ranging from [0, 1]
        # d_points: [N, 2] ranging from [-1, 1]

        d1 = self.DSO
        d2 = self.DSD
         
        points = deepcopy(points).astype(float)
        points[:, :2] -= 0.5 # [-0.5, 0.5]
        points[:, 2] = 0.5 - points[:, 2] # [-0.5, 0.5]
        points *= self.v_res * self.v_spacing # mm
        # rotate
        #pdb.set_trace()
        angle = -1 * angle # inverse direction
        rot_M = np.array([
            [np.cos(angle), -np.sin(angle), 0],
            [np.sin(angle),  np.cos(angle), 0],
            [            0,              0, 1]
        ])
        points = points @ rot_M.T
        #pdb.set_trace()
        coeff = (d2) / (d1 - points[:, 0]) # N,
        d_points = points[:, [2, 1]] * coeff[:, None] # [N, 2] float
        #pdb.set_trace()
        
        d_points /= (self.p_res * self.p_spacing)
        d_points *= 2 # NOTE: some points may fall outside [-1, 1]
        
        d_poitns_denorm = ((d_points + 1) / 2) * (self.p_res)
        #pdb.set_trace()
        return d_points , d_poitns_denorm


PATH_DICT = {
'image': 'images/{}.nii.gz',
'projs': 'projections/{}.pickle',
'projs_vis': 'projections_vis/{}.png',
'blocks_vals': 'blocks/{}',
'blocks_coords': 'blocks/blocks_coords.npy'}

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

def sitk_save(path, image, spacing=None, uint8=False):
    # default: float32 (input)
    image = image.astype(np.float32)
    image = image.transpose(2, 1, 0)
    if uint8:
        # value range should be [0, 1]
        image = (image * 255).astype(np.uint8)
    out = sitk.GetImageFromArray(image)
    if spacing is not None:
        out.SetSpacing(spacing.astype(np.float64)) # unit: mm
    sitk.WriteImage(out, path)


def get_filesname_from_txt(txt_file_path):
    files = []
    with open(txt_file_path, 'r') as f:
        file_name = f.readlines()
        for file in file_name:
            file_name = file.strip()
            #file_path = os.path.join(base_dir, file_name)
            files.append(file_name)   

    return files

class Overlap_Blocks_Dataset(Dataset):
    def __init__(self , root_data , names,path_dict , geo_path , mode='train' , out_res_scale=1.0 ):
        super().__init__()

        self.data_root = root_data
        self.names = names
        self._path_dict = deepcopy(path_dict)
        for key in self._path_dict.keys():
            path = os.path.join(self.data_root, self._path_dict[key])
            self._path_dict[key] = path
        self.blocks_path = os.path.join(self.data_root,'blocks')
        pdb.set_trace()
        # 根据names 在 blocks_path 下寻找文件 并将其放入all_blocks_name中
        all_files = os.listdir(self.blocks_path)
        self.all_blocks_name = []

        # 遍历names列表，找到对应的block文件
        for base_name in self.names:
            # 查找所有以base_name开头且以_block-和.npy结尾的文件
            matched_files = [f for f in all_files if f.startswith(f"{base_name}_block-") and f.endswith('.npy')]
            self.all_blocks_name.extend(matched_files)
        
        # 排序以确保顺序一致性
        self.all_blocks_name.sort()
        
        print(f"Found {len(self.all_blocks_name)} block files from {len(names)} base names")
        pdb.set_trace()
        
        self.blocks = np.load(os.path.join(self.data_root, self._path_dict['blocks_coords']))

        with open(geo_path , 'r') as f:
            dst_cfg  = yaml.safe_load(f)
            self.geo = Geometry(dst_cfg['projector'])


    def __len__(self):
        return len(self.all_blocks_name)
    
    def sample_projections(self, name, n_view=None):
        # -- load projections
        with open(os.path.join(self.data_root, self._path_dict['projs'].format(name)), 'rb') as f:
            data = pickle.load(f)
            projs = data['projs']         # uint8: [K, W, H]
            projs_max = data['projs_max'] # float
            angles = data['angles']       # float: [K,]

        if n_view is None:
            n_view = self.num_views

        # -- sample projections
        views = np.linspace(0, len(projs), n_view, endpoint=False).astype(int) # endpoint=False as the random_views is True during training, i.e., enabling view offsets.

        projs = projs[views].astype(np.float32) / 255.  # [ 0, 1]
        projs = projs[:, None, ...]
        angles = angles[views]

        projs = (projs * 2. ) - 1.
        # -- de-normalization
        #projs = projs * projs_max / 0.2

        return projs, angles
    
    def load_block(self, name):
        path = os.path.join(self.data_root, self._path_dict['blocks_vals'].format(name))
        #pdb.set_trace()
        block = np.load(path) # uint8
        return block
    
    def sample_points(self, points, values):
        # values [block]: uint8
        #choice = np.random.choice(len(points), size=self.npoint, replace=False)
        #points = points[choice]
        #values = values[choice]
        values = values.astype(np.float32) / 255.
        values = (values * 2.) - 1.
        points = points.reshape(-1,3)
        return points, values
    def project_points(self, points, angles):
        points_proj = []
        for a in angles:
            p = self.geo.project(points, a)
            points_proj.append(p)
        points_proj = np.stack(points_proj, axis=0) # [M, N, 2]
        return points_proj
    def __getitem__(self, index):
        block_name = self.all_blocks_name[index]

        case_name = block_name.split('_')[0]
        case_name = f"{case_name}_zoomed_s3"
        #pdb.set_trace()
        projs, angles = self.sample_projections(case_name , n_view=2)
        #pdb.set_trace()
        block_value = self.load_block(block_name) # uint8  h w d 1 
        #pdb.set_trace()
        b_idx = int(block_name.split('-')[-1].split('.')[0])
        block_coords =  self.blocks[b_idx]    # float 32  h w d 3 
        #pdb.set_trace()   
        block_coords =  (block_coords * 2.) - 1.
        p_for_projected , values = self.sample_points(block_coords , block_value) #
        #pdb.set_trace()

        points_projs , _ = self.project_points(p_for_projected , angles) 
        #pdb.set_trace()



        values = values.transpose(3,0,1,2)
        block_coords = block_coords.transpose(3,0,1,2)
        ret_dict = {
            'name': block_name ,
            'gt_idensity':values,  # [-1,1]   h w d 1
            'projs': projs,        # [-1,1]   2, 1 , uu ,vv 
            'angles': angles,       
            'points_projs': points_projs, # 2 , N , 2 
            'block_coords': block_coords  # [-1,1]  h w d 3
        
        }

        return ret_dict 

def make_dataset_files_name_list(image_path):
    file_names = os.listdir(image_path)
    #get train name and test name 
    random.shuffle(file_names)

    names = []
    for name in file_names:
        n = name.split('.')[0]
        names.append(n)


    train_file_names = names[:int(0.8*len(names))]
    test_file_names = names[int(0.8*len(names)):]

    return train_file_names , test_file_names


def save_split_files(train_files, test_files, train_path='train_files.txt', test_path='test_files.txt'):
    """
    保存划分后的训练集和测试集文件名到txt文件

    Args:
        train_files: 训练集文件名列表
        test_files: 测试集文件名列表
        train_path: 训练集文件名保存路径
        test_path: 测试集文件名保存路径
    """
    # 保存训练集文件名
    with open(train_path, 'w') as f:
        for file_name in train_files:
            f.write(file_name + '\n')
    
    # 保存测试集文件名
    with open(test_path, 'w') as f:
        for file_name in test_files:
            f.write(file_name + '\n')
    
    print(f"Train files saved to: {train_path}")
    print(f"Test files saved to: {test_path}")

def load_overlap_diffusion_blocks(root_data , files_name_path  , geo_path):
    files_name = get_filesname_from_txt(files_name_path)
    dataset = Overlap_Blocks_Dataset(root_data , files_name ,PATH_DICT , geo_path , mode='train' ,  out_res_scale=1.0)
    return dataset

if __name__ == '__main__':
    root = 'F:/Code_Space/Data_process/processed_128x128_s3.5'
    geo_path = "./geo_cfg/config_2d_256_s2.5_3d_176_3.0.yaml"
    train_case_name = './files_name/train_files.txt'
    names = get_filesname_from_txt(train_case_name)
    pdb.set_trace()
    ds = Overlap_Blocks_Dataset(root , names,path_dict=PATH_DICT ,geo_path=geo_path , mode='train' , out_res_scale=1.0 )
    pdb.set_trace()
    ds[0]
    #image_root = 'F:/Code_Space/Data_process/processed_128x128_s3.5/images'
    #train_file_names , test_file_name = make_dataset_files_name_list(image_path=image_root)
    #pdb.set_trace()
    #save_split_files(train_files=train_file_names , test_files= test_file_name)