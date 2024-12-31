import os
import torch
import numpy as np
import SimpleITK as sitk
import pdb
from tqdm import tqdm


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
