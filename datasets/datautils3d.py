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

class Self_LearningData(Dataset):
    def __init__(self, root_l ,transform=None):
        super(Self_LearningData, self).__init__()
        self.root_l = root_l
        self.image_files_path = []
        for file in os.listdir(self.root_l):
            self.image_files_path.append(os.path.join(self.root_l , file))

    def __len__(self):
        return len(self.image_files_path)
    
    def __getitem__(self, index):
        #pdb.set_trace()
        image_path = self.image_files_path[index]
        image_ ,  _  =  sitk_load(image_path)
        
        image_ = image_.astype(np.float32) / 255.
        image_ = ( image_ * 2 ) - 1

        image_t = image_.copy()
        image_ = image_[np.newaxis, :]
        image_ = torch.Tensor(image_)

        image_t = image_t[np.newaxis, :]
        image_t = torch.Tensor(image_t)


        return image_ , image_t


class Diffusion_Coords_IdensityData(Dataset):
    def __init__(self, root_path):
        super().__init__()
        self.root_path  = root_path
        files_name = os.listdir(self.root_path)
        self.files_path = []
        for file in files_name:
            f_p = os.path.join(self.root_path , file)
            self.files_path.append(f_p)

        random.shuffle(self.files_path)

    def __len__(self):
        return len(self.files_path)
    
    def __getitem__(self, index):
        npy_path = self.files_path[index]
        coords_idensity = np.load(npy_path)
        coords_idensity = torch.from_numpy(coords_idensity) #
        #pdb.set_trace()
        #print(" coords idensity : " , coords_idensity.shape )
        coords_idensity = rearrange(coords_idensity , "h w d c -> c h w d ")

        return coords_idensity

class Self_LearningRandomBlockPoints_Dataset(Dataset):
    def __init__(self, root_path,file_list, path_dict  , block_size ,train_mode=True):
        self.root_dir = root_path
        self.train_mode = train_mode
        #pdb.set_trace()

        self._path_dict = deepcopy(path_dict)
        for key in self._path_dict.keys():
            path = os.path.join(self.root_dir, self._path_dict[key])
            self._path_dict[key] = path

 
        self.blocks = np.load(self._path_dict['blocks_coords'])  # [(outres * outres * outres ) , 3 ] 
        self.blocks = (self.blocks - 0.5 ) * 2
        self.blocks_size = block_size  
        self.npoints = (block_size * block_size * block_size)      
        self.name_list = file_list   

        print(" files_data len : "   , len(self.name_list))



    def __len__(self):
        return len(self.name_list)   
    
    def load_ct(self, name):
        image, _ = sitk_load(
            os.path.join(self.root_dir, self._path_dict['image'].format(name)),
            uint8=True
        ) # float32

        return image
    
    def load_block(self, name , b_idx):
        #pdb.set_trace()
        path = self._path_dict['blocks_vals'].format(name, b_idx)
        block = np.load(path) # uint8
        return block

    def sample_points(self, points, values):
        #pdb.set_trace()
        #choice = np.random.choice(len(points), size=self.npoints, replace=False)
        points = points
        values = values
        values = values.astype(np.float32) / 255.

        return points , values


    def __getitem__(self, index):
        name = self.name_list[index]
  
        #pdb.set_trace()
        if self.train_mode is False:
            points = self.points
            points_gt = self.load_ct(name)
        else:
            b_idx = np.random.randint(len(self.blocks))
            block_values = self.load_block(name, b_idx)
            #pdb.set_trace()
            block_coords = self.blocks[b_idx] # [N, 3]
            points, points_gt = self.sample_points(block_coords, block_values)
            points_gt = points_gt[:,None]            
            
        points = torch.from_numpy(points)
        points_gt = torch.from_numpy(points_gt)

        coords_idensity = torch.concat([points , points_gt] , dim=-1)

        coords_idensity = coords_idensity.reshape(self.blocks_size , self.blocks_size , self.blocks_size , 4)
        
        coords_idensity = rearrange(coords_idensity , " h w d c -> c h w d ")
        

        return coords_idensity


class Self_LearningOverlapBlockPoints_Dataset(Dataset):
    def __init__(self, root_path,file_list, path_dict  , block_size ,train_mode=True):
        self.root_dir = root_path
        self.train_mode = train_mode
        #pdb.set_trace()

        self._path_dict = deepcopy(path_dict)
        for key in self._path_dict.keys():
            path = os.path.join(self.root_dir, self._path_dict[key])
            self._path_dict[key] = path

 
        self.blocks = np.load(self._path_dict['blocks_coords'])  # [(outres * outres * outres ) , 3 ] 
        self.blocks = (self.blocks - 0.5 ) * 2
        self.blocks_size = block_size  
        self.npoints = (block_size * block_size * block_size)      
        self.name_list = file_list   

        print(" files_data len : "   , len(self.name_list))

    def __len__(self):
        return len(self.name_list)   
    
    def load_ct(self, name):
        image, _ = sitk_load(
            os.path.join(self.root_dir, self._path_dict['image'].format(name)),
            uint8=True
        ) # float32

        return image
    
    def load_block(self, name , b_idx):
        #pdb.set_trace()
        path = self._path_dict['blocks_vals'].format(name, b_idx)
        block = np.load(path) # uint8
        return block

    def sample_points(self, points, values):
        #pdb.set_trace()
        #choice = np.random.choice(len(points), size=self.npoints, replace=False)
        points = points
        values = values
        values = values.astype(np.float32) / 255.

        return points , values

    def __getitem__(self, index):
        name = self.name_list[index]
  
        #pdb.set_trace()
        if self.train_mode is False:
            points = self.points
            points_gt = self.load_ct(name)
        else:
            b_idx = np.random.randint(len(self.blocks))
            block_values = self.load_block(name, b_idx)
            #pdb.set_trace()
            block_coords = self.blocks[b_idx] # [N, 3]
            points, points_gt = self.sample_points(block_coords, block_values)
            #points_gt = points_gt[:,None]            
            
        points = torch.from_numpy(points)
        points_gt = torch.from_numpy(points_gt)

        coords_idensity = torch.concat([points , points_gt] , dim=-1)

        coords_idensity = coords_idensity.permute(3,0,1,2)
        
        return coords_idensity


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






def loadSelf_learningData(root,batch_size , shuffle= True):
    sl_ds = Self_LearningData(root)
    #img_s , img_t = sl_ds[0]
    #pdb.set_trace()
    return DataLoader(sl_ds , batch_size=batch_size , shuffle=shuffle)


def load_with_coord(img_root ,  coord_root , files_list_path):
    files_name = get_filesname_from_txt(files_list_path)
    sl_ds = Self_LearningWithCoordsData(img_root , coord_root , files_name)
    #pdb.set_trace()
    ##sl_ds[0]
    return sl_ds

def load_diffusion_coords_idensity(root_data):
    dataset = Diffusion_Coords_IdensityData(root_data)
    
    return dataset


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

    root_path = '/root/share/pcc_gan_demo/cnetrilize_overlap_blocks_64'
    file_list_path = './files_list/p_train_demo.txt'
    load_diffusion_overlap_blocks(root_path, file_list_path)
