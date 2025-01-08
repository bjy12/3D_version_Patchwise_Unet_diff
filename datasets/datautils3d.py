# -*- coding: utf-8 -*-

import os
import random
import numpy as np
import torch
import SimpleITK as sitk
import yaml
from copy import deepcopy
from torch.utils.data import Dataset, DataLoader
from einops import rearrange
#from utils import sitk_load ,get_filesname_from_txt
import pdb
import pickle

PATH_DICT = {
'image': 'images/{}.nii.gz',
'projs': 'projections/{}.pickle',
'projs_vis': 'projections_vis/{}.png',
'blocks_vals': 'blocks/{}_block-{}.npy',
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





class Self_LearningWithCoordsData(Dataset):
    def __init__(self, root_img , root_coords , files_name ,transform=None):
        super().__init__()
        self.root_img = root_img
        self.root_coords = root_coords

        self.files_name = files_name
        self.image_files_path = []
        self.coords_files_path = []

        for file_numer in self.files_name:
            pre_fix_ = f"{file_numer}_cut_"
            match_img_files = self.find_files_with_prefix(root_img , pre_fix_)
            match_coords_files = self.find_files_with_prefix(root_coords , pre_fix_)    
            self.image_files_path.extend(match_img_files)
            self.coords_files_path.extend(match_coords_files)
        #pdb.set_trace()


    def find_files_with_prefix(self, path , prefix):
        matching_files = []
        for file in os.listdir(path):
            if file.startswith(prefix):
                matching_files.append(os.path.join(path, file))
        matching_files.sort()
        return matching_files

    def __len__(self):
        return len(self.image_files_path)
    
    def __getitem__(self, index):
        #pdb.set_trace()
        image_path = self.image_files_path[index]
        coord_path = self.coords_files_path[index]
        #print("coord_path:", image_path)
        #print("image_path:",coord_path)
        #pdb.set_trace()
        image_ , _=  sitk_load(image_path)
        h , w ,d  = image_.shape
        coords = np.load(coord_path)
        coords = np.transpose(coords , (3,0,1,2))
        #pdb.set_trace()
        coords = coords - 0.5
        image_ = image_.astype(np.float32) / 255.
        image_ = (image_ * 2 ) - 1


        image_gt = image_.copy()
        image_ = image_[np.newaxis, :]
        output = np.concatenate([coords ,image_ ] , axis = 0)
        output = torch.Tensor(output)
        image_gt = image_gt[np.newaxis, :]
        image_gt = torch.Tensor(image_gt)

        return output , image_gt


class Diffusion_Unconditional_Dataset(Dataset):
    def __init__(self, root_data):
        super().__init__()
        self.root_data = root_data
        self.image_files_path = []
        for file in os.listdir(self.root_data):
            self.image_files_path.append(os.path.join(self.root_data , file))
    def __len__(self):
        return len(self.image_files_path)
    def __getitem__(self, index):
        image_path = self.image_files_path[index]
        #pdb.set_trace()
        image_ , _ =  sitk_load(image_path , uint8=True)

        image_ = ( image_ * 2 ) - 1

        image_ = image_[np.newaxis, : ]
        
        return  image_ 

        return super().__getitem__(index)    

class Diffusion_Condition_Dataset(Dataset):
    def __init__(self , root_data ,file_names , path_dict ,mode='train' , out_res_scale=1.0 , patch_size=16):
        super().__init__()

        self.files_names = file_names
        self.data_root = root_data
        self._path_dict = deepcopy(path_dict)
        for key in self._path_dict.keys():
            path = os.path.join(self.data_root, self._path_dict[key])
            self._path_dict[key] = path
        #pdb.set_trace()
         

    def __len__(self):
        return len(self.files_names)
    def load_ct(self, name):
        image, _ = sitk_load(
            os.path.join(self.data_root,  self._path_dict['image'].format(name)),
            uint8=True
        ) # float32
        return image   
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

        projs = projs[views].astype(np.float32) / 255.
        projs = projs[:, None, ...]
        angles = angles[views]

    
        # normalization to [-1 , 1]
        projs = (projs * 2) - 1
        # -- de-normalization
        #projs = projs * projs_max / 0.2

        return projs, angles     
      
    def __getitem__(self, index):
        name = self.files_names[index]
        gt_idensity = self.load_ct(name)  # scale to [0,1]
        #normalization [-1,1] follow diffusion input 
        gt_idensity = (gt_idensity * 2) - 1 
        projs, angles = self.sample_projections(name ,n_view=2)
        #pdb.set_trace()
        ret_dict = {
            'name':name,
            'gt_idensity': gt_idensity,
            'projs': projs,
            'angles': angles,
        }
        return ret_dict







def load_diffusion_random_blocks(root_data , files_list_path ):
    files_name = get_filesname_from_txt(files_list_path)
    dataset =  Self_LearningRandomBlockPoints_Dataset( root_data , files_name , PATH_DICT , block_size=64 , train_mode=True)
    #dataset[0]

    return dataset

def load_diffusion_overlap_blocks(root_data , file_list_path):
    files_name = get_filesname_from_txt(file_list_path)
    dataset =  Self_LearningOverlapBlockPoints_Dataset( root_data , files_name , PATH_DICT , block_size=64 , train_mode=True)
    #dataset[0]

    return  dataset
    

def load_diffusion_unconditional(root_data):
    dataset = Diffusion_Unconditional_Dataset(root_data)
    #pdb.set_trace()
    #dataset[0]

    return dataset

def load_diffusion_condition(root_data , files_name_path):
    files_name = get_filesname_from_txt(files_name_path)
    dataset = Diffusion_Condition_Dataset(root_data , files_name ,path_dict= PATH_DICT , mode='train') 
    #d_s_0 = dataset[0]

    return dataset



if __name__ == '__main__':
    #coords_root_path = '/root/share/pcc_gan_demo/pcc_gan_demo_centriod/train/coords'
    #img_root_path = '/root/share/pcc_gan_demo/pcc_gan_demo_centriod/train/img'
    #files_name_path = './files_list/p_train_demo.txt'
    #load_with_coord(img_root= img_root_path , coord_root = coords_root_path ,files_list_path= files_name_path)
    #loadSelf_learningData(root_path)
    # 
    # coords_idensity_path = '/root/share/pcc_gan_demo/coords_with_idensity/train/coords'
    # load_diffusion_coords_idensity(coords_idensity_path)
    # 
    # root_path = '/root/share/pcc_gan_demo/cnetrilize_blocks_64'
    # file_list_path = './files_list/p_train_demo.txt'
    # load_diffusion_random_blocks(root_path, file_list_path)

    # root_path = '/root/share/pcc_gan_demo/cnetrilize_overlap_blocks_64'
    # file_list_path = './files_list/p_train_demo.txt'
    # load_diffusion_overlap_blocks(root_path, file_list_path)

    root_path = 'F:/Data_Space/Pelvic1K/processed_128x128_s2/'
    # load_diffusion_unconditional(root_path)
    files_name = 'F:/Code_Space/diffusers_v2/diffusers_pcc-demo_exp_0/diffusers_pcc-demo_exp_0/files_name/pelvic_coord_train_16.txt'
    geo_config = 'F:/Code_Space/diffusers_v2/diffusers_pcc-demo_exp_0/diffusers_pcc-demo_exp_0/geo_cfg/config_2d_128_s2.5_3d_128_2.0_25.yaml'
    load_diffusion_condition(root_data=root_path , files_name_path=files_name , geo_config=geo_config)
