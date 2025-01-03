import torch 
import numpy as np
import pdb
from pachify_and_projector import pachify , pachify3D
from utils import sitk_load


#main
if __name__ == '__main__':


    #coord_npy_path = 'F:/Data_Space/Pelvic1K/processed_128x128_s2/blocks/blocks_coords.npy'
    #coord_npy_path = 'F:/Data_Space/Pelvic1K/cnetrilize_overlap_blocks_64/blocks/blocks_coords.npy'
    
    #np_coords = np.load(coord_npy_path)
    #pdb.set_trace()



    nii_example = 'F:/Data_Space/Pelvic1K/processed_128x128_s2/images/0001.nii.gz'
    img , _     = sitk_load(nii_example , uint8=True)
    pdb.set_trace()
    img_tensor = torch.from_numpy(img).to('cuda')
    img_tensor = img_tensor.to(torch.float32)
    img_tensor = img_tensor.unsqueeze(0)
    pdb.set_trace()
    padded , pos = pachify3D(img_tensor , patch_size=32)
    #image_tensor = torch.randn([1,1,128,128] , dtype=torch.float32).to('cuda')
    pdb.set_trace()
    # match_tensor_1 = padded.squeeze(1)
    # match_tensor_2 = img_tensor
    # mismatch = match_tensor_1 != match_tensor_2
    # if mismatch.any():
    #     mismatch_indices = torch.nonzero(mismatch)
    #     for idx in mismatch_indices:
    #         # 将idx转换为tuple以便索引
    #         idx_tuple = tuple(idx.tolist())
    #         print(f"位置 {idx_tuple}:")
    #         print(f"Tensor1值: {match_tensor_1[idx_tuple]}")
    #         print(f"Tensor2值: {match_tensor_2[idx_tuple]}")
    #patch_image , pos_ = pachify(image_tensor , patch_size=32)
    #pdb.set_trace()

    pos = pos.squeeze(0)
    pos = pos.reshape(3,-1)
    pos = pos.transpose(1,0)

    
    coords_h = np.arange(128) 
    coords_w = np.arange(128)
    coords_d = np.arange(128)
    coords = np.stack(np.meshgrid(coords_h, coords_w, coords_d, indexing='ij'), axis=-1)
    coords = coords.reshape(-1,3)
    #pdb.set_trace()
    pos_np = pos.cpu().numpy()
    def find_matching_indices(coords, pos_np):
        # 将坐标转换为元组列表，以便进行比较
        pos_tuples = set(map(tuple, pos_np))
        coords_tuples = list(map(tuple, coords))
        
        # 找到匹配的索引
        matching_indices = [i for i, coord in enumerate(coords_tuples) if coord in pos_tuples]
        return matching_indices
    matching_indices = find_matching_indices(coords, pos_np)
    #pdb.set_trace()
    coords = coords[matching_indices]
    coords = torch.from_numpy(coords).to('cuda')
    #pdb.set_trace()
    #torch.from_numpy()

    p_image_list = []
    c_image_list = []
    pdb.set_trace()
    for p , c in zip(pos_np , coords):
        x_pos , y_pos , z_pos= p[0] , p[1] , p[2]
        x_c , y_c , z_c = c[0] , c[1] ,c[2]
        #pdb.set_trace()
        p_image = img_tensor[:,x_pos,y_pos ,z_pos]

        c_image = img_tensor[:,x_c,y_c ,z_c]        
        p_image_list.append(p_image)
        c_image_list.append(c_image)
    pdb.set_trace()
    p_image = torch.stack(p_image_list , dim=0)
    #p_image = p_image.reshape(32,32,32)
    pdb.set_trace()
    padded = padded.squeeze(0).squeeze(0)
    pdb.set_trace()
    p_image = p_image.reshape(32,32,32)
    p_image = p_image.squeeze(-1)
    
    is_eq = torch.equal(padded ,p_image)
    pdb.set_trace()








    block_size = np.array([32,32,32]).astype(np.int32)
    resolution = np.array([128,128,128]).astype(np.int32)
    nx, ny ,nz = block_size
    offsets = resolution / block_size

    base = np.mgrid[:nx,:ny,:nz]
    pdb.set_trace()






