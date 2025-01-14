import os
import torch
import numpy as np
import SimpleITK as sitk
import pdb
from tqdm import tqdm

def get_random_batch(dataloader, logger, max_attempts=10):
    """从dataloader中随机抽取一个batch并验证其有效性"""
    dataset_size = len(dataloader.dataset)
    
    valid_batch_found = False
    attempt = 0
    last_batch = None
    
    while attempt < max_attempts and not valid_batch_found:
        # 随机选择一个索引
        random_idx = torch.randint(0, dataset_size, (1,)).item()  # 只获取一个索引值
        # 获取随机样本
        batch = dataloader.dataset[random_idx]  # 直接获取一个样本
        
        last_batch = batch
        # 检查有效性
        if not check_tensor_validity(batch['gt_idensity']):
            valid_batch_found = True
            logger.info("Found valid batch for evaluation")
        else:
            attempt += 1
            logger.info(f"Found invalid batch, retry {attempt}/{max_attempts}")
    
    if not valid_batch_found:
        logger.warning(f"Could not find valid batch after {max_attempts} attempts, using last batch")
        batch = last_batch
    pdb.set_trace()
    # 将单个样本转换为"batch"格式（添加batch维度）
    return batch, valid_batch_found
def write_img(vol, out_path, ref_path, new_spacing=None):
    img_ref = sitk.ReadImage(ref_path)
    img = sitk.GetImageFromArray(vol)
    img.SetDirection(img_ref.GetDirection())
    if new_spacing is None:
        img.SetSpacing(img_ref.GetSpacing())
    else:
        img.SetSpacing(tuple(new_spacing))
    img.SetOrigin(img_ref.GetOrigin())
    sitk.WriteImage(img, out_path)
    print('Save to:', out_path)

def check_tensor_validity(tensor):
    """检查张量是否所有值都相同
    Args:
        tensor (torch.Tensor): 输入张量
    Returns:
        bool: 如果所有值相同返回True，否则返回False
    """
    return bool((tensor == tensor.reshape(-1)[0]).all())


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



def read_img(in_path):
    img_lit = []
    filenames = os.listdir(in_path)
    for f in tqdm(filenames):
        img = sitk.ReadImage(os.path.join(in_path, f))
        img_vol = sitk.GetArrayFromImage(img)
        img_lit.append(img_vol)
    return img_lit




def make_coord(shape, ranges=None, flatten=True):
    """
    Make coordinates at grid centers.
    """
    coord_seqs = []
    for i, n in enumerate(shape):
        if ranges is None:
            v0, v1 = -1, 1
        else:
            v0, v1 = ranges[i]
        r = (v1 - v0) / (2 * n)
        seq = v0 + r + (2 * r) * torch.arange(n).float()
        coord_seqs.append(seq)
    ret = torch.stack(torch.meshgrid(*coord_seqs), dim=-1)
    if flatten:
        ret = ret.view(-1, ret.shape[-1])
    return ret


def save_pred_to_local(images ,save_path , global_steps=0 ):
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    # image size is (batch_size, 1 , h , d ,w )
    pdb.set_trace()
    for i in range(images.shape[0]):
        image = images[i, 0, :, :, :]
        save_image_path = os.path.join(save_path, str(global_steps) + '_' + str(i) + '.nii.gz')
        sitk_save(save_image_path , image , uint8=True)
