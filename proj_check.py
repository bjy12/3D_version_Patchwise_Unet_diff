import pickle
import os
import sys
import matplotlib.pyplot as plt
import numpy as np
import pdb
import yaml
import tigre
import torch 
import torch.nn as nn
from datasets.geometry import Geometry
from pachify_and_projector import extract3DPatch
from tqdm import tqdm
from utils import sitk_save
import SimpleITK as sitk
def sitk_load(path, uint8=False, spacing_unit='mm'):
    # load as float32
    itk_img = sitk.ReadImage(path)
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
def visualize_slice_points(slice_data, point=None):
    """
    显示切片和点的位置（左上角为原点）
    """
    import matplotlib.pyplot as plt
    
    plt.imshow(slice_data, cmap='gray', origin='upper')
    
    if point is not None:
        x, y = point
        plt.scatter(x, y, color='red', s=100)
    
    plt.show()


def _generate_overlap_blocks(data ,ct_resolution , block_size , stride):
    block_size = np.array(block_size).astype(int)
    ct_resolution = np.array(ct_resolution).astype(int)
    ct_res = ct_resolution
    nx, ny, nz = block_size
    ct = data
    #pdb.set_trace()
    cuts_h = max(1, (ct_res - nx) // stride + 1)
    cuts_w = max(1, (ct_res - ny) // stride + 1)
    cuts_d = max(1, (ct_res - nz) // stride + 1)

    coords_h = np.arange(ct_res) 
    coords_w = np.arange(ct_res)
    coords_d = np.arange(ct_res)
    coords = np.stack(np.meshgrid(coords_h, coords_w, coords_d, indexing='ij'), axis=-1)
    #pdb.set_trace()
    total_cuts = cuts_h * cuts_w * cuts_d

    blocks_list = []
    coords_list = []
    value_list = []
    with tqdm(total=total_cuts) as pbar:
        for i in range(cuts_h):
            start_h = i * stride
            #pdb.set_trace()
            end_h = min(start_h + nx, ct_res)
            
            for j in range(cuts_w):
                start_w = j * stride
                end_w = min(start_w + ny, ct_res)
                
                for k in range(cuts_d):
                    start_d = k * stride
                    end_d = min(start_d + nz, ct_res)
                    #block_idx = i * cuts_w * cuts_d + j * cuts_d + k
                    coords_block = coords[start_h:end_h, start_w:end_w, start_d:end_d]    
                    #pdb.set_trace()
                    value_block = ct[start_h:end_h, start_w:end_w, start_d:end_d]
                    value_block = value_block[:,:,:,None]
                    value_list.append(value_block)
                    blocks_list.append(coords_block)
                    coords_list.append(coords_block.reshape(-1,3))
                    
                    pbar.update(1)   

    blocks_list = np.stack(blocks_list, axis=0) # [N, *, 3]
    #pdb.set_trace()
    blocks_list = blocks_list / (ct_resolution - 1) # coords starts from 0
    blocks_list = blocks_list.astype(np.float32)
    #pdb.set_trace()
    _block_info = {
        'coords': blocks_list,  #  M , H W D 3  
        'list': coords_list   #   a list [ N , 3 ] 
    }
    #pdb.set_trace()
    return _block_info , value_list

def visualize_slice(slice_data, title="Slice Visualization"):
    """
    可视化单个切片
    
    Parameters:
    -----------
    slice_data: 2D numpy array
        切片数据
    title: str
        图像标题
    """
    import matplotlib.pyplot as plt
    
    plt.figure(figsize=(10, 10))
    plt.imshow(slice_data, cmap='gray')
    plt.colorbar(label='Intensity')
    plt.title(title)
    plt.axis('on')  # 显示坐标轴
    plt.show()

def plot_two_views_and_save(img1, img2, proj1, proj2, point_idx, angle_idx, save_path='projections'):
    """
    并排绘制两个视角的图像和投影点并保存
    
    参数:
    img1, img2: 两个视角的2D图像
    proj1, proj2: 投影点坐标
    point_idx: 当前点的索引
    angle_idx: 当前角度的索引
    save_path: 保存图像的路径
    """
    # 创建一个包含两个子图的图像
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))
    
    # 第一个视角
    ax1.imshow(img1, origin='upper', cmap='gray')
    ax1.scatter(proj1[:,1], proj1[:,0], color='red', marker='o', s=1)
    ax1.grid(True, linestyle='--', alpha=0.5)
    ax1.set_title(f'View 1 - Point {point_idx}, Angle {angle_idx}\nCoordinates: ({proj1[0]}, {proj1[1]})')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    
    # 第二个视角
    ax2.imshow(img2, origin='upper', cmap='gray')
    ax2.scatter(proj2[:,1], proj2[:,0], color='red', marker='o', s=1)
    ax2.grid(True, linestyle='--', alpha=0.5)
    ax2.set_title(f'View 2 - Point {point_idx}, Angle {angle_idx}\nCoordinates: ({proj2[0]}, {proj2[1]})')
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    
    # 为每个子图添加颜色条
    fig.colorbar(ax1.imshow(img1, origin='upper', cmap='gray'), ax=ax1)
    fig.colorbar(ax2.imshow(img2, origin='upper', cmap='gray'), ax=ax2)
    
    # 调整布局
    plt.tight_layout()
    
    # 保存图像
    os.makedirs(save_path, exist_ok=True)
    plt.savefig(f'{save_path}/point_{point_idx}_angle_{angle_idx}.png', 
                dpi=300, 
                bbox_inches='tight')
    plt.close()

def plot_image_with_point(image, point_coord, color='red', marker='o', size=256):
    """
    在图像上绘制点，使用左上角作为原点(0,0)
    
    Parameters:
    -----------
    image : numpy.ndarray
        要显示的图像数据
    point_coord : tuple or list
        点的坐标 (x, y)，以左上角为原点
    color : str
        点的颜色，默认为红色
    marker : str
        点的标记样式，默认为'o'
    size : int
        点的大小，默认为100
    """
    # 创建图形
    plt.figure(figsize=(8, 8))
    
    # 显示图像
    # origin='upper'将使原点位于左上角
    plt.imshow(image, origin='upper', cmap='gray')
    
    # 绘制点
    # 注意：不需要转换y坐标，因为origin='upper'已经处理了这个问题
    plt.scatter(point_coord[1], point_coord[0], 
               color=color, marker=marker, s=size)
    
    # 添加网格
    plt.grid(True, linestyle='--', alpha=0.5)
    
    # 设置坐标轴标签
    plt.xlabel('X')
    plt.ylabel('Y')
    
    # 添加标题，显示点的坐标
    plt.title(f'Point at coordinates: ({point_coord[0]}, {point_coord[1]})')
    
    # 显示颜色条
    plt.colorbar()
    
    # 保持图像纵横比
    plt.axis('equal')
    
    plt.show()

def visualize_projection_with_point(proj_image_1 , proj_image_2 , proj_point_1 , proj_point_2):
    # 创建一个1行2列的子图
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
    
    # 左侧图像
    img1 = ax1.imshow(proj_image_1, cmap='gray')
    # 在左图上标记投影点
    ax1.plot(proj_point_1[:, 0], proj_point_1[:, 1], 'r.', markersize=10)
    ax1.set_title('Projection Image 1 with Point')
    ax1.axis('on')
    # 添加colorbar
    plt.colorbar(img1, ax=ax1)
    
    # 右侧图像
    img2 = ax2.imshow(proj_image_2, cmap='gray')
    # 在右图上标记投影点
    ax2.plot(proj_point_2[:, 0], proj_point_2[:, 1], 'r.', markersize=10)
    ax2.set_title('Projection Image 2 with Point')
    ax2.axis('on')
    # 添加colorbar
    plt.colorbar(img2, ax=ax2)
    
    # 调整布局
    plt.tight_layout()
    plt.show()
def view_dimensions(image, projections=None):
        """
        显示所有维度的切片视图
        """
        # 显示原始图像的三个维度
        plt.figure(figsize=(15, 5))
        
        # x-y平面 (固定z)
        plt.subplot(131)
        mid_z = image.shape[2] // 2
        plt.imshow(image[:, :, mid_z], cmap='gray')
        plt.title(f'X-Y平面 (z={mid_z})\nshape={image[:, :, mid_z].shape}')
        plt.colorbar()
        
        # x-z平面 (固定y)
        plt.subplot(132)
        mid_y = image.shape[1] // 2
        plt.imshow(image[:, mid_y, :], cmap='gray')
        plt.title(f'X-Z平面 (y={mid_y})\nshape={image[:, mid_y, :].shape}')
        plt.colorbar()
        
        # y-z平面 (固定x)
        plt.subplot(133)
        mid_x = image.shape[0] // 2
        plt.imshow(image[mid_x, :, :], cmap='gray')
        plt.title(f'Y-Z平面 (x={mid_x})\nshape={image[mid_x, :, :].shape}')
        plt.colorbar()
        
        plt.suptitle('原始图像各维度切片', fontsize=14)
        plt.tight_layout()
        plt.show()

def visualize_slices_with_point(slice_x, slice_y, slice_z, point_coords):
    # 创建一个3x1的子图
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
    
    # 显示X方向切片
    ax1.imshow(slice_x, cmap='gray')
    ax1.plot(point_coords[2], point_coords[1], 'r.', markersize=10)  # y-z平面上的点
    ax1.set_title('X Slice')
    ax1.axis('on')
    
    # 显示Y方向切片
    ax2.imshow(slice_y, cmap='gray')
    ax2.plot(point_coords[2], point_coords[0], 'r.', markersize=10)  # x-z平面上的点
    ax2.set_title('Y Slice')
    ax2.axis('on')
    
    # 显示Z方向切片
    ax3.imshow(slice_z, cmap='gray')
    ax3.plot(point_coords[1], point_coords[0], 'r.', markersize=10)  # x-y平面上的点
    ax3.set_title('Z Slice')
    ax3.axis('on')
    
    plt.tight_layout()
    plt.show()



proj_path = 'F:/Data_Space/Pelvic1K/processed_128x128_s2/projections/0001.pickle'

with open(proj_path , 'rb') as f:
    data = pickle.load(f)
    projs = data['projs']         # uint8: [K, W, H]
    projs_max = data['projs_max'] # float
    angles = data['angles']       # float: [K,]
#pdb.set_trace()

projs_all = projs
projs = projs[0].astype(np.float32) / 255.

projs = projs[None, ...]
# -- de-normalization
#pdb.set_trace()
projs = projs * projs_max / 0.2

geo_config = './geo_cfg/config_2d_128_s2.5_3d_128_2.0_25.yaml'
with open(geo_config, 'r') as f:
    dst_cfg = yaml.safe_load(f)
    out_res = np.array(dst_cfg['dataset']['resolution'])
    out_res = np.round(out_res * 1.0).astype(int) # to align the output shape with 'scipy.ndimage.zoom'
    geo = Geometry(dst_cfg['projector'])

image_path = 'F:/Data_Space/Pelvic1K/processed_128x128_s2/images/0001.nii.gz'
#pdb.set_trace()
image, space_ = sitk_load(image_path ,uint8=True)
#pdb.set_trace()
image = torch.from_numpy(image)
image = image.to('cuda')
image = image.unsqueeze(0)
start_pos = (128 - 64) // 2 
positions = {
    'i': start_pos ,
    'j': start_pos ,
    'k': start_pos ,
}   
patch_image , patch_pos , de_norm_pos =  extract3DPatch(images=image , patch_size=64  , positions=positions)
patch_image = patch_image.squeeze(0).squeeze(0)
patch_image = patch_image.cpu().numpy()
patch_pos = patch_pos.cpu().numpy()
#pdb.set_trace()
points = patch_pos
projected_points = []
#pdb.set_trace()
for i in range(points.shape[0]):
    coords = points[i]
    #pdb.set_trace()
    coords = coords.reshape(3,-1).transpose(1,0)
    for a in angles:
        p , p_d =  geo.project(coords , a)
        #pdb.set_trace()
        projected_points.append(p_d)
        #print(" proj points  min:" , p.min() ," max:", p.max())
projected_points = np.stack(projected_points ,axis=0).astype(np.float32)
pdb.set_trace()
proj1 = projected_points[0]      # 第一个投影
proj2 = projected_points[1]  # 第二个投影
proj1 = np.round(proj1).astype(int)
proj2 = np.round(proj2).astype(int)
plot_two_views_and_save(projs_all[0] , projs_all[1] , proj1 , proj2 , 0 , 0)
pdb.set_trace()

projected_image_space = np.zeros_like(image.cpu().numpy())
de_norm_pos = de_norm_pos.cpu().numpy()
de_norm_pos = de_norm_pos.squeeze(0).reshape(3,-1).transpose(1,0)
de_norm_pos = np.round(de_norm_pos).astype(int)
patch_image_list = patch_image.reshape(-1)
# 3. 将patch中的值放入对应位置
for i, pos in enumerate(de_norm_pos):
    # pos的形状应该是[3]，分别对应x,y,z坐标
    x, y, z = pos
    # patch_image的值放入对应位置
    projected_image_space[0, x, y, z] = patch_image_list[i] 

pdb.set_trace()




sitk_save('./projections/patch_.nii.gz' ,patch_image , uint8=True)
sitk_save('./projections/projected_image_space.nii.gz' ,projected_image_space[0] , uint8=True)
pdb.set_trace()
sitk_save("./projections/gt_.nii.gz" , image[0].cpu().numpy() , uint8=True)





#* check points and proj points  relative 
value_path = 'F:/Code_Space/Data_process/processed/blocks/0001_block-32.npy'
value = np.load(value_path)
pdb.set_trace()
value = value.astype(np.float32)
pdb.set_trace()
value_p = (np.where(value!=0)[0][100])
image_path = 'F:/Code_Space/Data_process/processed/images/0001.nii.gz'


image, space_ = sitk_load(image_path)
block_info , values_blocks = _generate_overlap_blocks(image ,ct_resolution=256 ,  block_size=[64 , 64,64] , stride=32)
pdb.set_trace()
overlap_points_set = block_info['coords'][173]
overlap_points_set = overlap_points_set.reshape(-1,3) * 256
overlap_points_set = overlap_points_set.astype(np.int32)
overlap_points_set = overlap_points_set[100:120]
pdb.set_trace()
overlap_v_set = values_blocks[173]
overlap_v_set = overlap_v_set.reshape(-1,1)
overlap_v_set = overlap_v_set[100:120]
pdb.set_trace()
# tigre.plotImg(image.transpose(2,1,0), dim='x')
# tigre.plotImg(image.transpose(2,1,0), dim='y')
# pdb.set_trace()
image = image.transpose(2,1,0) # z y x 
image = image[::-1,:,:]
pdb.set_trace()
#exp_image_slice = image[100]
#visualize_slice()
gt_value = image[overlap_points_set[:,0] , overlap_points_set[:,1] , overlap_points_set[:,2]]
#


#view_dimensions(image)
#image = image.transpose(2,1,0)
#view_dimensions(image)

blocks_num = 32
pdb.set_trace()

selec_blocks = points[blocks_num]

#* random blocks val
#selec_points = selec_blocks[value_p]
#selec_points = selec_points[None,:]

#* overlap blocks val 
selec_points = overlap_points_set[-4]
selec_poitns = selec_points[None]
selec_points = selec_points / 256 

selec_proj_points = selec_points.copy()
selec_proj_points = selec_proj_points[None,:]
selec_points = selec_points[None,:]

pdb.set_trace()
# change axis follow projection function 
selec_points[:,:2] -= 0.5 
selec_points[:, 2] = 0.5 - selec_points[:,2]  # 
selec_points += 0.5
selec_points *= 256 * 1.0 # x y z 


selec_points = np.round(selec_points).astype(int)
pdb.set_trace()

slic_z = image[selec_points[0,2] ,:, :]
slic_y = image[:, selec_points[0,1] , :]
slic_x = image[:,:, selec_points[0,0]]
visualize_slice(slic_z)
visualize_slice(slic_y)
visualize_slice(slic_x)

visualize_slice_points(slic_z ,selec_points[0,:2])
pdb.set_trace()
visualize_slice_points(slic_y ,np.array([selec_points[0,0] , selec_points[0,2]]))
visualize_slice_points(slic_x ,selec_points[0,1:3])

# slice_x = image[selec_points[0,0] , : , :]
# slice_y = image[:, selec_points[0,1] , :]
# slice_z = image[:, :, selec_points[0,2]]
# visualize_slices_with_point(slice_x , slice_y , slice_z , selec_points[0])
#selec_points = np.array([1.0,1.0,1.0], dtype=np.float32)

#pdb.set_trace()
angles = data['angles']
#pdb.set_trace()
a_1 = angles[0]
a_2 = angles[1]

points_proj_test_min_max = []
all_proj_points = block_info['coords']
all_proj_points = all_proj_points.reshape(-1,3)
pdb.set_trace()
for a in angles:
    p , p_d = geo.project(all_proj_points , a)
    points_proj_test_min_max.append(p)
points_proj_test_min_max = np.stack(points_proj_test_min_max , axis=0)
# 检查是否有小于-1的值
less_than_minus_one = (points_proj_test_min_max < -1).any()

# 检查是否有大于1的值
greater_than_one = (points_proj_test_min_max > 1).any()

# 打印结果
print("Contains values < -1:", less_than_minus_one)
print("Contains values > 1:", greater_than_one)

#pdb.set_trace()
points_proj = []
points_proj_d = []
for a in angles:
    p , p_d = geo.project(selec_proj_points , a)
    pdb.set_trace()
    print(" proj points  min:" , p.min() ," max:", p.max())
    points_proj.append(p)
    points_proj_d.append(p_d)
points_proj = np.stack(points_proj , axis=0)
points_proj_d = np.stack(points_proj_d , axis=0)
#pdb.set_trace()
points_proj_d = np.round(points_proj_d).astype(int)
pdb.set_trace()
plot_image_with_point(projs_all[0] , points_proj_d[0,0,:] , color='red' , marker='o' , size=256)
plot_image_with_point(projs_all[1] , points_proj_d[1,0,:] , color='red' , marker='o' , size=256)




#pdb.set_trace()
# selec_index = (selec_points * 255).astype(np.int32)
# p_1_proj_index =  (((proj_p_1 + 1) / 2 ) * 256)
# p_2_proj_index =  (((proj_p_2 + 1) / 2 ) * 256)
 
# pdb.set_trace()
# visualize_projection_with_point(projs_all[0] , projs_all[1] , p_1_proj_index , p_2_proj_index)
# # visualize_projection_with_point(projs_all[0] , p_1_proj_index)
# # visualize_projection_with_point(projs_all[1] , p_2_proj_index)
# # pdb.set_trace()


# slice_x = image[selec_index[0], :, :]
# slice_y = image[:, selec_index[1], :]
# slice_z = image[:, :, selec_index[2]]

# visualize_slices_with_point(slice_x , slice_y , slice_z ,selec_index)
# # angles = data['angles']
# # a_1 = angles[0]
# # a_2 = angles[1]
# # pdb.set_trace()
# # selec_points = selec_points[None ,:]
# # proj_p_1 = geo.project(selec_points , a_1)
# # proj_p_2 = geo.project(selec_points , a_2)
# # pdb.set_trace()
# # proj_p_1 = ((proj_p_1 + 1 ) / 2 ) * 256
# # proj_p_2 = ((proj_p_2 + 1 ) / 2 ) * 256 
# # pdb.set_trace()

# # visualize_projection_with_point(projs_all[0] , proj_p_1)
# # visualize_projection_with_point(projs_all[1] , proj_p_2)

    






