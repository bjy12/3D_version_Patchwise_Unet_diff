import torch 
import numpy as np
import pdb

def pachify(images, patch_size, padding=None):
    device = images.device
    batch_size, resolution = images.size(0), images.size(2)
    pdb.set_trace()
    if padding is not None:
        padded = torch.zeros((images.size(0), images.size(1), images.size(2) + padding * 2,
                                images.size(3) + padding * 2), dtype=images.dtype, device=device)
        padded[:, :, padding:-padding, padding:-padding] = images
    else:
        padded = images

    h, w = padded.size(2), padded.size(3)
    th, tw = patch_size, patch_size
    if w == tw and h == th:
        i = torch.zeros((batch_size,), device=device).long()
        j = torch.zeros((batch_size,), device=device).long()
    else:
        i = torch.randint(0, h - th + 1, (batch_size,), device=device)
        j = torch.randint(0, w - tw + 1, (batch_size,), device=device)

    rows = torch.arange(th, dtype=torch.long, device=device) + i[:, None]
    columns = torch.arange(tw, dtype=torch.long, device=device) + j[:, None]
    padded = padded.permute(1, 0, 2, 3)
    padded = padded[:, torch.arange(batch_size)[:, None, None], rows[:, torch.arange(th)[:, None]],
                columns[:, None]]
    padded = padded.permute(1, 0, 2, 3)

    x_pos = torch.arange(tw, dtype=torch.long, device=device).unsqueeze(0).repeat(th, 1).unsqueeze(0).unsqueeze(0).repeat(batch_size, 1, 1, 1)
    y_pos = torch.arange(th, dtype=torch.long, device=device).unsqueeze(1).repeat(1, tw).unsqueeze(0).unsqueeze(0).repeat(batch_size, 1, 1, 1)
    x_pos = x_pos + j.view(-1, 1, 1, 1)
    y_pos = y_pos + i.view(-1, 1, 1, 1)
    x_pos = (x_pos / (resolution - 1) - 0.5) * 2.
    y_pos = (y_pos / (resolution - 1) - 0.5) * 2.
    images_pos = torch.cat((x_pos, y_pos), dim=1)

    return padded, images_pos


def pachify3D(images , patch_size , padding=None):
    device = images.device
    batch_size = images.size(0)
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

    # Extract patches
    padded = padded.permute(1, 0, 2, 3, 4)
    padded = padded[:, torch.arange(batch_size)[:, None, None, None],
                    depths[:, torch.arange(td)[:, None, None]],
                    rows[:, None, torch.arange(th)[:, None]],
                    columns[:, None, None]]
    padded = padded.permute(1, 0, 2, 3, 4)

    # Generate position encodings
    x_pos = torch.arange(tw, dtype=torch.long, device=device)
    y_pos = torch.arange(th, dtype=torch.long, device=device)
    z_pos = torch.arange(td, dtype=torch.long, device=device)

    # Create 3D position grid
    z, y, x = torch.meshgrid(z_pos, y_pos, x_pos, indexing='ij')
    
    # Expand dimensions for batch size and add offsets
    x_pos = x.unsqueeze(0).repeat(batch_size, 1, 1, 1) + j.view(-1, 1, 1, 1)
    y_pos = y.unsqueeze(0).repeat(batch_size, 1, 1, 1) + i.view(-1, 1, 1, 1)
    z_pos = z.unsqueeze(0).repeat(batch_size, 1, 1, 1) + k.view(-1, 1, 1, 1)

    # Normalize positions to [-1, 1]
    x_pos = (x_pos / (resolution_w - 1) - 0.5) * 2.
    y_pos = (y_pos / (resolution_h - 1) - 0.5) * 2.
    z_pos = (z_pos / (resolution_d - 1) - 0.5) * 2.

    # Combine position encodings
    volumes_pos = torch.stack((x_pos, y_pos, z_pos), dim=1)    

    return padded, volumes_pos




if __name__ == '__main__':
    # image = torch.randn([2,4,128,128] , dtype=torch.float32).to('cuda')
    # patch_size = 32
    # pachify(image,patch_size)
    image = torch.randn([2,4,128,128,128] ,dtype=torch.float32).to('cuda')
    patch_size = 32
    pachify3D(image , patch_size)