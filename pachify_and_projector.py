import torch 
import numpy as np
import yaml
from datasets.geometry import Geometry
import pdb

def pachify(images, patch_size, padding=None):
    device = images.device
    batch_size, resolution = images.size(0), images.size(2)
    #pdb.set_trace()
    if padding is not None:
        padded = torch.zeros((images.size(0), images.size(1), images.size(2) + padding * 2,
                                images.size(3) + padding * 2), dtype=images.dtype, device=device)
        padded[:, :, padding:-padding, padding:-padding] = images
    else:
        padded = images
    pdb.set_trace()
    h, w = padded.size(2), padded.size(3)
    th, tw = patch_size, patch_size
    if w == tw and h == th:
        i = torch.zeros((batch_size,), device=device).long()
        j = torch.zeros((batch_size,), device=device).long()
    else:
        i = torch.randint(0, h - th + 1, (batch_size,), device=device)
        j = torch.randint(0, w - tw + 1, (batch_size,), device=device)
    pdb.set_trace()
    rows = torch.arange(th, dtype=torch.long, device=device) + i[:, None]
    columns = torch.arange(tw, dtype=torch.long, device=device) + j[:, None]
    padded = padded.permute(1, 0, 2, 3)
    pdb.set_trace()
    padded = padded[:, torch.arange(batch_size)[:, None, None], rows[:, torch.arange(th)[:, None]],
                columns[:, None]]
    padded = padded.permute(1, 0, 2, 3)
    pdb.set_trace()
    x_pos = torch.arange(tw, dtype=torch.long, device=device).unsqueeze(0).repeat(th, 1).unsqueeze(0).unsqueeze(0).repeat(batch_size, 1, 1, 1)
    y_pos = torch.arange(th, dtype=torch.long, device=device).unsqueeze(1).repeat(1, tw).unsqueeze(0).unsqueeze(0).repeat(batch_size, 1, 1, 1)
    x_pos = x_pos + j.view(-1, 1, 1, 1)
    y_pos = y_pos + i.view(-1, 1, 1, 1)
    pdb.set_trace()
    # x_pos = (x_pos / (resolution - 1) - 0.5) * 2.
    # y_pos = (y_pos / (resolution - 1) - 0.5) * 2.
    images_pos = torch.cat((x_pos, y_pos), dim=1)

    return padded, images_pos


def pachify3D(images , patch_size , padding=None ):
    #pdb.set_trace()
    device = images.device
    batch_size = images.size(0)
    images = images.unsqueeze(1)
    resolution_d , resolution_h , resolution_w = images.size(2) , images.size(3) , images.size(4)
    #pdb.set_trace()
    if isinstance(patch_size, int):
        td , th , tw = patch_size , patch_size , patch_size
    else:
        td , th , tw = patch_size
    
    if padding is not None:
        padded = torch.zeros((images.size(0), images.size(1),
                              images.size(2)+padding*2,
                              images.size(3)+padding*2,
                              images.size(4)+padding*2), dtype=images.dtype, device=images.device)
        padded[:, :, padding:-padding, padding:-padding, padding:-padding] = images
    else:
        padded = images
    
    d , h , w = padded.size(2) , padded.size(3) , padded.size(4)

    if w == tw and h == th and d == td:
        i = torch.zeros((batch_size,), device=device).long()  # height
        j = torch.zeros((batch_size,), device=device).long()  # width
        k = torch.zeros((batch_size,), device=device).long()  # depth
    else:
        i = torch.randint(0, h - th + 1, (batch_size,), device=device)
        j = torch.randint(0, w - tw + 1, (batch_size,), device=device)
        k = torch.randint(0, d - td + 1, (batch_size,), device=device) 

    # Generate coordinate grids for patch extraction
    rows = torch.arange(th, dtype=torch.long, device=device) + i[:, None]
    columns = torch.arange(tw, dtype=torch.long, device=device) + j[:, None]
    depths = torch.arange(td, dtype=torch.long, device=device) + k[:, None]
    #pdb.set_trace()
    # Extract patches
    padded = padded.permute(1, 0, 2, 3, 4)
    # padded = padded[:, torch.arange(batch_size)[:, None, None, None],
    #                 depths[:, torch.arange(td)[:, None, None]],
    #                 rows[:, None, torch.arange(th)[:, None]],
    #                 columns[:, None, None]]
    padded = padded[:, torch.arange(batch_size)[:, None, None, None],
                    columns[:, torch.arange(tw)[:, None, None]],  # x
                    rows[:, None, torch.arange(th)[:, None]],     # y
                    depths[:, None, None]]                        # z
    padded = padded.permute(1, 0, 2, 3, 4)

    # Generate position encodings
    x_pos = torch.arange(tw, dtype=torch.long, device=device)
    y_pos = torch.arange(th, dtype=torch.long, device=device)
    z_pos = torch.arange(td, dtype=torch.long, device=device)

    # Create 3D position grid
    #z, y, x = torch.meshgrid(z_pos, y_pos, x_pos, indexing='ij')
    x, y, z = torch.meshgrid(x_pos, y_pos, z_pos, indexing='ij')

    # Expand dimensions for batch size and add offsets
    x_pos = x.unsqueeze(0).repeat(batch_size, 1, 1, 1) + j.view(-1, 1, 1, 1)
    y_pos = y.unsqueeze(0).repeat(batch_size, 1, 1, 1) + i.view(-1, 1, 1, 1)
    z_pos = z.unsqueeze(0).repeat(batch_size, 1, 1, 1) + k.view(-1, 1, 1, 1)

    # Normalize positions to [-1, 1]
    # x_pos = (x_pos / (resolution_w - 1) - 0.5) * 2.
    # y_pos = (y_pos / (resolution_h - 1) - 0.5) * 2.
    # z_pos = (z_pos / (resolution_d - 1) - 0.5) * 2.
    x_pos = (x_pos / (resolution_w - 1))   # [0,1]
    y_pos = (y_pos / (resolution_h - 1))   # [0,1]
    z_pos = (z_pos / (resolution_d - 1))   # [0,1]

    # Combine position encodings
    volumes_pos = torch.stack((x_pos, y_pos, z_pos), dim=1)    
    #pdb.set_trace()
    return padded, volumes_pos


def init_projector(geo_cfg ):
    with open(geo_cfg , 'r' ) as f :
        dst_cfg = yaml.safe_load(f)
        out_res = np.array(dst_cfg['dataset']['resolution'])
        projector = Geometry(dst_cfg['projector'])
    return projector

def project_points(points , angles , projector):
    #pdb.set_trace()
    points_proj = []
    for a in angles:
        #pdb.set_trace()
        p = projector.project(points , a)
        #pdb.set_trace()
        points_proj.append(p)
    #pdb.set_trace()
    points_proj = np.stack(points_proj , axis=0)
    
    return points_proj

def pachify3d_projected_points(clean_image , patch_size , angles , projector):
    #pdb.set_trace()
    device = clean_image.device
    patch_image , patch_pos = pachify3D(clean_image , patch_size)
    patch_image_np = patch_image.cpu().numpy()
    patch_pos_np = patch_pos.cpu().numpy() 
    angles = angles[0,:].cpu().numpy()
    #pdb.set_trace()
        # 释放GPU内存
    projs_points_batch = []
    for i in range(patch_image_np.shape[0]):
        #pdb.set_trace()
        _points = patch_pos_np[i].reshape(3,-1)
        _points = _points.transpose(1,0)
        #pdb.set_trace()
        points_projeted_ = project_points(_points , angles , projector)
        #pdb.set_trace()
        projs_points_batch.append(points_projeted_)
        #pdb.set_trace()
    projs_points_batch = np.stack(projs_points_batch , axis=0)
    #pdb.set_trace()

    # 将numpy数组转换为tensor并移到对应设备
    patch_image_tensor =  patch_image
    #patch_pos_tensor   =  (patch_pos - 0.5) / 2
    patch_pos_tensor   =  (patch_pos - 0.5) * 2

    projs_points_tensor = torch.from_numpy(projs_points_batch).to(device)
    #pdb.set_trace()
    return patch_image_tensor , patch_pos_tensor , projs_points_tensor



def full_volume_inference_process(clean_image , patch_size , block_size , angles , projector):
    pdb.set_trace()
    device = clean_image.device
    patch_image , patch_pos = pachify3D(clean_image , patch_size)
    pdb.set_trace()
    blocks_data = split_volume_into_blocks(patch_image , patch_pos , block_size)
    # 获取切分后的块
    image_blocks = blocks_data['image_blocks']
    pos_blocks = blocks_data['pos_blocks']
    blocks_info = blocks_data['blocks_info']
    # 计算每个块的投影点
    angles_np = angles[0,:].cpu().numpy()
    num_blocks = len(image_blocks)
    B = patch_image.shape[0]
    #pdb.set_trace()
    proj_points_list = []
    for block_idx in range(num_blocks):
        current_pos_block = pos_blocks[block_idx]
        current_pos_np = current_pos_block.cpu().numpy()
        #pdb.set_trace()
        for b in range(B):
            points = current_pos_np[b].reshape(3, -1).transpose(1, 0)
            points_projected = project_points(points, angles_np, projector)
            proj_points_list.append(points_projected)

    #pdb.set_trace()
    # 转换投影点为张量
    proj_points = np.stack(proj_points_list, axis=0)
    proj_points_tensor = torch.from_numpy(proj_points).to(device)
    #!应该是* 2  scale to [-1,1] 现在是 /2  [-0.25,0.25]
    pos_blocks = (pos_blocks - 0.5) / 2
    #pdb.set_trace()
    return image_blocks , pos_blocks , proj_points_tensor , blocks_info


def split_volume_into_blocks(volume , pos , block_size):
    B , C , D , H , W  = volume.shape

    # 计算每个维度需要多少个块
    num_blocks_d = D // block_size + (1 if D % block_size != 0 else 0)
    num_blocks_h = H // block_size + (1 if H % block_size != 0 else 0)
    num_blocks_w = W // block_size + (1 if W % block_size != 0 else 0)
    
    image_blocks = []
    pos_blocks = []   

    # 遍历所有可能的块
    for d in range(num_blocks_d):
        d_start = d * block_size
        d_end = min((d + 1) * block_size, D)
        
        for h in range(num_blocks_h):
            h_start = h * block_size
            h_end = min((h + 1) * block_size, H)
            
            for w in range(num_blocks_w):
                w_start = w * block_size
                w_end = min((w + 1) * block_size, W)
                
                # 提取当前块
                current_image_block = volume[:, :, d_start:d_end, h_start:h_end, w_start:w_end]
                current_pos_block = pos[:, :, d_start:d_end, h_start:h_end, w_start:w_end]
                
                # 存储块信息
                image_blocks.append(current_image_block)
                pos_blocks.append(current_pos_block)
    
    # 将所有块组合成张量
    image_blocks = torch.stack(image_blocks, dim=0)  # [num_blocks, B, C, block_d, block_h, block_w]
    pos_blocks = torch.stack(pos_blocks, dim=0)      # [num_blocks, B, 3, block_d, block_h, block_w]
    
    # 记录块的位置信息
    blocks_info = {
        'num_blocks_d': num_blocks_d,
        'num_blocks_h': num_blocks_h,
        'num_blocks_w': num_blocks_w,
        'block_size': block_size,
        'original_shape': (D, H, W)
    }
    
    return {
        'image_blocks': image_blocks,
        'pos_blocks': pos_blocks,
        'blocks_info': blocks_info
    }






if __name__ == '__main__':
    # image = torch.randn([2,4,128,128] , dtype=torch.float32).to('cuda')
    # patch_size = 32
    # pachify(image,patch_size)
    image = torch.randn([2,4,128,128,128] ,dtype=torch.float32).to('cuda')
    patch_size = 32
    pachify3D(image , patch_size)