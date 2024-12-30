# -*- coding = utf-8 -*-

"""
PCC-GAN generator implementation
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from einops import rearrange
from timm.models.layers import to_3tuple
from timm.models.layers import trunc_normal_
from timm.models.registry import register_model
import pdb
from torch.nn.functional import silu


class PointRecuder(nn.Module):
    """
    Point Reducer is implemented by a layer of conv since it is mathmatically equal.
    Input: tensor in shape [B, in_chans, H, W, D]
    Output: tensor in shape [B, embed_dim, H/stride, W/stride, D/stride]
    """

    def __init__(self, patch_size=16, stride=16, padding=0,
                 in_chans=3, embed_dim=768, norm_layer=None):
        super().__init__()
        patch_size = to_3tuple(patch_size)
        stride = to_3tuple(stride)
        padding = to_3tuple(padding)
        self.proj = nn.Conv3d(in_chans, embed_dim, kernel_size=patch_size, stride=stride, padding=padding)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        x = self.proj(x)
        x = self.norm(x)
        return x


class PointExpander(nn.Module):
    """
    Point Expander is implemented by a layer of decov since it is mathmatically equal.
    Input: tensor in shape [B, in_chans, H, W, D]
    Output: tensor in shape [B, embed_dim, H*stride, W*stride, D*stride]
    """

    def __init__(self, patch_size=16, stride=16, padding=0,
                 in_chans=3, embed_dim=768, norm_layer=None):
        super().__init__()
        patch_size = to_3tuple(patch_size)
        stride = to_3tuple(stride)
        padding = to_3tuple(padding)
        # print(in_chans, embed_dim)
        self.proj = nn.ConvTranspose3d(in_chans, embed_dim, kernel_size=patch_size,
                                       stride=stride, padding=padding)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        x = self.proj(x)
        x = self.norm(x)
        return x


class GroupNorm(nn.GroupNorm):
    """
    Group Normalization with 1 group.
    Input: tensor in shape [B, C, H, W, D]
    """

    def __init__(self, num_channels, **kwargs):
        super().__init__(1, num_channels, **kwargs)


def pairwise_cos_sim(x1: torch.Tensor, x2: torch.Tensor):
    """
    return pair-wise similarity matrix between two tensors
    :param x1: [B,...,M,D]
    :param x2: [B,...,N,D]
    :return: similarity matrix [B,...,M,N]
    """
    x1 = F.normalize(x1, dim=-1)
    x2 = F.normalize(x2, dim=-1)

    sim = torch.matmul(x1, x2.transpose(-2, -1))
    return sim


class Cluster(nn.Module):
    def __init__(self, dim, out_dim,
                 proposal_w=2, proposal_h=2, proposal_d=2,
                 fold_w=2, fold_h=2, fold_d=2,
                 heads=4, head_dim=24,
                 return_center=False):
        """
        :param dim:  channel nubmer
        :param out_dim: channel nubmer
        :param proposal_w: the sqrt(proposals) value, we can also set a different value
        :param proposal_h: the sqrt(proposals) value, we can also set a different value
        :param proposal_d: the sqrt(proposals) value, we can also set a different value
        :param fold_w: the sqrt(number of regions) value, we can also set a different value
        :param fold_h: the sqrt(number of regions) value, we can also set a different value
        :param fold_d: the sqrt(number of regions) value, we can also set a different value
        :param heads:  heads number in context cluster
        :param head_dim: dimension of each head in context cluster
        :param return_center: if just return centers instead of dispatching back (deprecated).
        """
        super().__init__()
        self.heads = heads
        self.head_dim = head_dim
        self.f = nn.Conv3d(dim, heads * head_dim, kernel_size=1)  # for similarity
        self.proj = nn.Conv3d(heads * head_dim, out_dim, kernel_size=1)  # for projecting channel number
        self.v = nn.Conv3d(dim, heads * head_dim, kernel_size=1)  # for value
        self.sim_alpha = nn.Parameter(torch.ones(1))
        self.sim_beta = nn.Parameter(torch.zeros(1))
        self.centers_proposal = nn.AdaptiveAvgPool3d((proposal_w, proposal_h, proposal_d))
        self.fold_w = fold_w
        self.fold_h = fold_h
        self.fold_d = fold_d
        self.return_center = return_center

    def forward(self, x):  # [b,c,w,h, d]
        #pdb.set_trace()
        value = self.v(x)  # conv3d 1x1x1  b c0 w h d
        
        x = self.f(x)   # conv3d 1x1x1    b c1 w h d 
        x = rearrange(x, "b (e c) w h d -> (b e) c w h d", e=self.heads) # ( b e )  c1  w h d 
        value = rearrange(value, "b (e c) w h d -> (b e) c w h d", e=self.heads)  # ( b e ) c0 w h d
        if self.fold_w > 1 and self.fold_h > 1:
            # split the big feature maps to small local regions to reduce computations.
            b0, c0, w0, h0, d0 = x.shape
            assert w0 % self.fold_w == 0 and h0 % self.fold_h == 0 and d0 % self.fold_d == 0, \
                f"Ensure the feature map size ({w0}*{h0}*{w0}) can be divided by fold " \
                f"{self.fold_w}*{self.fold_h}*{self.fold_d}"
            x = rearrange(x, "b c (f1 w) (f2 h) (f3 d) -> (b f1 f2 f3) c w h d", f1=self.fold_w,
                          f2=self.fold_h, f3=self.fold_d)  # [bs*blocks,c,ks[0],ks[1],ks[2]]
            value = rearrange(value, "b c (f1 w) (f2 h) (f3 d) -> (b f1 f2) c w h d", f1=self.fold_w,
                              f2=self.fold_h, f3=self.fold_d)
        b, c, w, h, d = x.shape
        centers = self.centers_proposal(x)  # [b,c,C_W,C_H,C_D], we set M = C_W*C_H and N = w*h*d  # caculate cluster center 
        value_centers = rearrange(self.centers_proposal(value), 'b c w h d -> b (w h d) c')  # [b , (w * h *d )  ,c1 ] centers value 
        b, c, ww, hh, dd = centers.shape
        sim = torch.sigmoid(
            self.sim_beta +
            self.sim_alpha * pairwise_cos_sim(
                centers.reshape(b, c, -1).permute(0, 2, 1),
                x.reshape(b, c, -1).permute(0, 2, 1)
            )
        )  # [B,M,N]   # 计算聚类中心与原始特征的相似性    
        # we use mask to sololy assign each point to one center
        sim_max, sim_max_idx = sim.max(dim=1, keepdim=True)
        mask = torch.zeros_like(sim)  # binary #[B,M,N] 
        mask.scatter_(1, sim_max_idx, 1.) # 寻找每个样本最具有相似性的聚类中心
        sim = sim * mask
        value2 = rearrange(value, 'b c w h d -> b (w h d) c')  # [B,N,D] #  
        # aggregate step, out shape [B,M,D]
        # a small bug: mask.sum should be sim.sum according to Eq. (1),
        # mask can be considered as a hard version of sim in our implementation.
        out = ((value2.unsqueeze(dim=1) * sim.unsqueeze(dim=-1)).sum(dim=2) + value_centers) / (
                sim.sum(dim=-1, keepdim=True) + 1.0)  # [B,M,D]

        if self.return_center:
            out = rearrange(out, "b (w h d) c -> b c w h d", w=ww, h=hh)  # center shape
        else:
            # dispatch step, return to each point in a cluster
            out = (out.unsqueeze(dim=2) * sim.unsqueeze(dim=-1)).sum(dim=1)  # [B,N,D]
            out = rearrange(out, "b (w h d) c -> b c w h d", w=w, h=h)  # cluster shape

        if self.fold_w > 1 and self.fold_h > 1 and self.fold_d > 1:
            # recover the splited regions back to big feature maps if use the region partition.
            out = rearrange(out, "(b f1 f2 f3) c w h d -> b c (f1 w) (f2 h) (f3 d)", f1=self.fold_w,
                            f2=self.fold_h, f3=self.fold_d)
        out = rearrange(out, "(b e) c w h d -> b (e c) w h d", e=self.heads)
        out = self.proj(out)
        return out


class Mlp(nn.Module):
    """
    Implementation of MLP with nn.Linear (would be slightly faster in both training and inference).
    Input: tensor with shape [B, C, H, W, D]
    """

    def __init__(self, in_features, hidden_features=None,
                 out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.fc1(x.permute(0, 2, 3, 4, 1))
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x).permute(0, 4, 1, 2, 3)
        x = self.drop(x)
        return x


class TimeClusterBlock(nn.Module):
    def __init__(self, dim, time_emb_dim ,mlp_ratio=4.,
                 act_layer=nn.GELU, norm_layer=GroupNorm, drop=0.,
                 use_layer_scale=True, layer_scale_init_value=1e-5,
                 proposal_w=2, proposal_h=2, proposal_d=2,
                 fold_w=2, fold_h=2, fold_d=2,
                 heads=4, head_dim=24, return_center=False):
        super().__init__()
        #pdb.set_trace()
        self.norm1 = norm_layer(dim)
        self.token_mixer = Cluster(dim=dim, out_dim=dim,
                                   proposal_w=proposal_w, proposal_h=proposal_h, proposal_d=proposal_d,
                                   fold_w=fold_w, fold_h=fold_h, fold_d=fold_d,
                                   heads=heads, head_dim=head_dim, return_center=return_center)

        self.time_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim,dim)
        )      

        self.norm2 = norm_layer(dim)


        mlp_hidden_dim = int(dim * mlp_ratio)

        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        # The following technique is useful to train deep ContextClusters.
        self.use_layer_scale = use_layer_scale
        if use_layer_scale:
            self.layer_scale_1 = nn.Parameter(layer_scale_init_value * torch.ones((dim)), requires_grad=True)
            self.layer_scale_2 = nn.Parameter(layer_scale_init_value * torch.ones((dim)), requires_grad=True)
    def forward(self , x , time_emb):
        #pdb.set_trace()
        time_emb = self.time_mlp(time_emb)

        time_emb = time_emb.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
     
        if self.use_layer_scale:
            # token mixing分支
            h = self.norm1(x)
            h = self.token_mixer(h)
            # 在token mixing输出后添加时间信息
            h = h + time_emb
            x = x + self.layer_scale_1.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1) * h

            # MLP分支
            h = self.norm2(x)
            h = self.mlp(h)
            # 在MLP输出后添加时间信息
            h = h + time_emb
            x = x + self.layer_scale_2.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1) * h
        else:
            h = self.token_mixer(self.norm1(x)) + time_emb
            x = x + h
            h = self.mlp(self.norm2(x)) + time_emb
            x = x + h
        return x 


class ClusterBlock(nn.Module):
    """
    Implementation of one block.
    --dim: embedding dim
    --mlp_ratio: mlp expansion ratio
    --act_layer: activation
    --norm_layer: normalization
    --drop: dropout rate
    --use_layer_scale, --layer_scale_init_value: LayerScale,
        refer to https://arxiv.org/abs/2103.17239
    """

    def __init__(self, dim, mlp_ratio=4.,
                 act_layer=nn.GELU, norm_layer=GroupNorm, drop=0.,
                 use_layer_scale=True, layer_scale_init_value=1e-5,
                 # for context-cluster
                 proposal_w=2, proposal_h=2, proposal_d=2,
                 fold_w=2, fold_h=2, fold_d=2,
                 heads=4, head_dim=24, return_center=False):

        super().__init__()

        self.norm1 = norm_layer(dim)
        # dim, out_dim, proposal_w=2,proposal_h=2, fold_w=2, fold_h=2, heads=4, head_dim=24, return_center=False
        self.token_mixer = Cluster(dim=dim, out_dim=dim,
                                   proposal_w=proposal_w, proposal_h=proposal_h, proposal_d=proposal_d,
                                   fold_w=fold_w, fold_h=fold_h, fold_d=fold_d,
                                   heads=heads, head_dim=head_dim, return_center=return_center)
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        # The following technique is useful to train deep ContextClusters.
        self.use_layer_scale = use_layer_scale
        if use_layer_scale:
            self.layer_scale_1 = nn.Parameter(layer_scale_init_value * torch.ones((dim)), requires_grad=True)
            self.layer_scale_2 = nn.Parameter(layer_scale_init_value * torch.ones((dim)), requires_grad=True)

    def forward(self, x):
        # input  x   b embeding  h w d 
        #pdb.set_trace()
        if self.use_layer_scale:
            x = x + self.layer_scale_1.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1) * self.token_mixer(self.norm1(x))
            x = x + self.layer_scale_2.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1) * self.mlp(self.norm2(x))
        else:
            x = x + self.token_mixer(self.norm1(x))
            x = x + self.mlp(self.norm2(x))
        return x


def basic_blocks(dim, index, layers,
                 mlp_ratio=4.,
                 act_layer=nn.GELU, norm_layer=GroupNorm,
                 drop_rate=.0,
                 use_layer_scale=True, layer_scale_init_value=1e-5,
                 # for context-cluster
                 proposal_w=2, proposal_h=2, proposal_d=2,
                 fold_w=2, fold_h=2, fold_d=2,
                 heads=4, head_dim=24, return_center=False):
    blocks = []
    for block_idx in range(layers[index]):
        blocks.append(ClusterBlock(
            dim, mlp_ratio=mlp_ratio,
            act_layer=act_layer, norm_layer=norm_layer,
            drop=drop_rate,
            use_layer_scale=use_layer_scale,
            layer_scale_init_value=layer_scale_init_value,
            proposal_w=proposal_w, proposal_h=proposal_h, proposal_d=proposal_d,
            fold_w=fold_w, fold_h=fold_h, fold_d=fold_d,
            heads=heads, head_dim=head_dim, return_center=return_center
        ))
    blocks = nn.Sequential(*blocks)

    return blocks


class ContextCluster_Denoised(nn.Module):
    """
    ContextCluster, the main class of our model
    --layers: [x,x,x,x], number of blocks for the 4 stages
    --embed_dims, --mlp_ratios, the embedding dims, mlp ratios
    --norm_layer, 
    --act_layer: define the types of normalization and activation
    --in_patch_size, --in_stride, --in_pad: specify the patch embedding
        for the input image
    --down_patch_size --down_stride --down_pad:
        specify the downsample (patch embed.)
    """

    def __init__(self, layers, embed_dims=None,
                 mlp_ratios=None,
                 norm_layer='GroupNorm', act_layer=nn.GELU,
                 in_patch_size=3, in_stride=2, in_pad=1,
                 down_patch_size=3, down_stride=2, down_pad=1,
                 up_patch_size=2, up_stride=2, up_pad=0,
                 drop_rate=0.,
                 use_layer_scale=True, layer_scale_init_value=1e-5,
                 # the parameters for context-cluster
                 proposal_w=[2, 2, 2, 2], proposal_h=[2, 2, 2, 2], proposal_d=[2, 2, 2, 2],
                 fold_w=[8, 4, 2, 1], fold_h=[8, 4, 2, 1], fold_d=[8, 4, 2, 1],
                 heads=[2, 4, 6, 8], head_dim=[16, 16, 32, 32],
                 time_embed_dims=None,
                 **kwargs):
        super().__init__()
        #pdb.set_trace()
        """ Encoder """
        self.patch_embed = PointRecuder(patch_size=in_patch_size, stride=in_stride, padding=in_pad,
                                        in_chans=4, embed_dim=embed_dims[0])
        #pdb.set_trace()
        self.time_dim = time_embed_dims[0]

        self.time_mlp = nn.Sequential(
                        SinusoidalPositionEmbeddings(self.time_dim),
                        nn.Linear(self.time_dim, self.time_dim * 4),
                        nn.SiLU(),
                        nn.Linear(self.time_dim * 4, self.time_dim),
                        nn.SiLU()  # 最后需要添加激活层
                    )
        if norm_layer == 'BatchNorm3d':
            self.norm_layer = nn.BatchNorm3d
        if norm_layer == 'GroupNorm':
            self.norm_layer = GroupNorm
        # en0
        #pdb.set_trace()
        self.en0 = self._make_time_blocks(embed_dims[0],self.time_dim, 0, layers, mlp_ratio=mlp_ratios[0], act_layer=act_layer,
                                norm_layer=self.norm_layer, drop_rate=drop_rate, use_layer_scale=use_layer_scale,
                                layer_scale_init_value=layer_scale_init_value,
                                proposal_w=proposal_w[0], proposal_h=proposal_h[0], proposal_d=proposal_d[0],
                                fold_w=fold_w[0], fold_h=fold_h[0], fold_d=fold_d[0],
                                heads=heads[0], head_dim=head_dim[0], return_center=False)
        # en1
        self.down1 = PointRecuder(patch_size=down_patch_size, stride=down_stride, padding=down_pad,
                                  in_chans=embed_dims[0], embed_dim=embed_dims[1])
        self.en1 = self._make_time_blocks(embed_dims[1],self.time_dim , 1, layers, mlp_ratio=mlp_ratios[1], act_layer=act_layer,
                                norm_layer=self.norm_layer, drop_rate=drop_rate, use_layer_scale=use_layer_scale,
                                layer_scale_init_value=layer_scale_init_value,
                                proposal_w=proposal_w[1], proposal_h=proposal_h[1], proposal_d=proposal_d[1],
                                fold_w=fold_w[1], fold_h=fold_h[1], fold_d=fold_d[1],
                                heads=heads[1], head_dim=head_dim[1], return_center=False)
        # en2
        self.down2 = PointRecuder(patch_size=down_patch_size, stride=down_stride, padding=down_pad,
                                  in_chans=embed_dims[1], embed_dim=embed_dims[2])
        self.en2 = self._make_time_blocks(embed_dims[2], self.time_dim,2, layers, mlp_ratio=mlp_ratios[2], act_layer=act_layer,
                                norm_layer=self.norm_layer, drop_rate=drop_rate, use_layer_scale=use_layer_scale,
                                layer_scale_init_value=layer_scale_init_value,
                                proposal_w=proposal_w[2], proposal_h=proposal_h[2], proposal_d=proposal_d[2],
                                fold_w=fold_w[2], fold_h=fold_h[2], fold_d=fold_d[2],
                                heads=heads[2], head_dim=head_dim[2], return_center=False)
        # en3
        self.down3 = PointRecuder(patch_size=down_patch_size, stride=down_stride, padding=down_pad,
                                  in_chans=embed_dims[2], embed_dim=embed_dims[3])
        self.en3 = self._make_time_blocks(embed_dims[3],self.time_dim , 3, layers, mlp_ratio=mlp_ratios[3], act_layer=act_layer,
                                norm_layer=self.norm_layer, drop_rate=drop_rate, use_layer_scale=use_layer_scale,
                                layer_scale_init_value=layer_scale_init_value,
                                proposal_w=proposal_w[3], proposal_h=proposal_h[3], proposal_d=proposal_d[3],
                                fold_w=fold_w[3], fold_h=fold_h[3], fold_d=fold_d[3],
                                heads=heads[3], head_dim=head_dim[3], return_center=False)


        # Bottleneck
        self.bot = self._make_time_blocks(embed_dims[3],self.time_dim ,3, layers, mlp_ratios[3], act_layer,
                                        self.norm_layer, drop_rate, use_layer_scale, layer_scale_init_value,
                                        proposal_w[3], proposal_h[3], proposal_d[3],
                                        fold_w[3], fold_h[3], fold_d[3],
                                        heads[3], head_dim[3])




        # Decoder blocks
        self.up0 = PointExpander(patch_size=up_patch_size, stride=up_stride, padding=up_pad,
                              in_chans=embed_dims[3], embed_dim=embed_dims[2])
        self.de0 = self._make_time_blocks(embed_dims[2],self.time_dim ,2, layers, mlp_ratios[2], act_layer,
                                        self.norm_layer, drop_rate, use_layer_scale, layer_scale_init_value,
                                        proposal_w[2], proposal_h[2], proposal_d[2],
                                        fold_w[2], fold_h[2], fold_d[2],
                                        heads[2], head_dim[2])

        self.up1 = PointExpander(patch_size=up_patch_size, stride=up_stride, padding=up_pad,
                              in_chans=embed_dims[2], embed_dim=embed_dims[1])
        self.de1 = self._make_time_blocks(embed_dims[1],self.time_dim ,1, layers, mlp_ratios[1], act_layer,
                                        self.norm_layer, drop_rate, use_layer_scale, layer_scale_init_value,
                                        proposal_w[1], proposal_h[1], proposal_d[1],
                                        fold_w[1], fold_h[1], fold_d[1],
                                        heads[1], head_dim[1])

        self.up2 = PointExpander(patch_size=up_patch_size, stride=up_stride, padding=up_pad,
                              in_chans=embed_dims[1], embed_dim=embed_dims[0])
        self.de2 = self._make_time_blocks(embed_dims[0],self.time_dim ,0, layers, mlp_ratios[0], act_layer,
                                        self.norm_layer, drop_rate, use_layer_scale, layer_scale_init_value,
                                        proposal_w[0], proposal_h[0], proposal_d[0],
                                        fold_w[0], fold_h[0], fold_d[0],
                                        heads[0], head_dim[0])
        self.patch_expand = nn.Sequential(
            PointExpander(patch_size=up_patch_size, stride=up_stride, padding=up_pad,
                          in_chans=embed_dims[0], embed_dim=32),
        )
        self.out_norm = self.norm_layer(32)
        self.out_cov = nn.Conv3d(in_channels=32, out_channels=1, kernel_size=1, stride=1)
        #pdb.set_trace()
        self.with_coord = kwargs['with_coord']


    def _make_time_blocks(self, dim, time_emb_dim,index, layers, mlp_ratio, act_layer, norm_layer, drop_rate,
                         use_layer_scale, layer_scale_init_value, proposal_w, proposal_h, proposal_d,
                         fold_w, fold_h, fold_d, heads, head_dim,return_center=False):
        blocks = []
        for _ in range(layers[index]):
            blocks.append(TimeClusterBlock(
                dim,time_emb_dim , mlp_ratio=mlp_ratio,
                act_layer=act_layer, norm_layer=norm_layer,
                drop=drop_rate, use_layer_scale=use_layer_scale,
                layer_scale_init_value=layer_scale_init_value,
                proposal_w=proposal_w, proposal_h=proposal_h, proposal_d=proposal_d,
                fold_w=fold_w, fold_h=fold_h, fold_d=fold_d,
                heads=heads, head_dim=head_dim,return_center=return_center))
        return nn.ModuleList(blocks)
    
    def forward_block_sequence(self, x, blocks, time_emb):
        for block in blocks:
            x = block(x, time_emb)
        return x


    def forward_embeddings(self, x , with_coords = True):
        _, c, img_w, img_h, img_d = x.shape
        # print(f"img size is {c} * {img_w} * {img_h}")
        # register positional information buffer.
        #pdb.set_trace()
        if with_coords == True:
            x = self.patch_embed(x)
        else:
            range_w = torch.arange(0, img_w, step=1) / (img_w - 1.0)
            range_h = torch.arange(0, img_h, step=1) / (img_h - 1.0)
            range_d = torch.arange(0, img_d, step=1) / (img_d - 1.0)
            fea_pos = torch.stack(torch.meshgrid(range_w, range_h, range_d), dim=-1).float()
            fea_pos = fea_pos.to(x.device)
            fea_pos = fea_pos - 0.5

            pos = fea_pos.permute(3, 0, 1, 2).unsqueeze(dim=0).expand(x.shape[0], -1, -1, -1, -1)
            pdb.set_trace()
            x = self.patch_embed(torch.cat([x, pos], dim=1))  # b embed img_h , img_w , img_l

        return x

    def restore_embeddings(self, x):
        x = self.patch_expand(x)

        return x

    def forward(self, x, timesteps):
        """
        Forward pass through the denoising UNet
        Args:
            x (torch.Tensor): Input tensor [B, C, H, W, D]
            timesteps (torch.Tensor): Time embeddings [B]
        """
        # 1. Process time embeddings
        time_emb = self.time_mlp(timesteps)  # [B, time_emb_dim]
        
        # 2. Initial feature embedding with coordinates if needed
        x = self.forward_embeddings(x, self.with_coord)  # [B, embed_dim, H, W, D]
        #pdb.set_trace()
        # 3. Encoder
        # Stage 1
        skip0 = x
        x = self.forward_block_sequence(x, self.en0, time_emb)
        #pdb.set_trace()
        # Stage 2
        x = self.down1(x)
        skip1 = x 
        x = self.forward_block_sequence(x, self.en1, time_emb)
        
        # Stage 3 
        x = self.down2(x)
        skip2 = x
        x = self.forward_block_sequence(x, self.en2, time_emb)
        
        # Stage 4
        x = self.down3(x)
        x = self.forward_block_sequence(x, self.en3, time_emb)
        
        # 4. Bottleneck
        x = self.forward_block_sequence(x, self.bot, time_emb)
        
        # 5. Decoder
        # Stage 1
        x = self.up0(x)
        x = x + skip2  # Skip connection
        x = self.forward_block_sequence(x, self.de0, time_emb)
        
        # Stage 2
        x = self.up1(x)
        x = x + skip1  # Skip connection
        x = self.forward_block_sequence(x, self.de1, time_emb)
        
        # Stage 3
        x = self.up2(x)
        #pdb.set_trace()
        x = x + skip0  # Skip connection 
        x = self.forward_block_sequence(x, self.de2, time_emb)
        #pdb.set_trace()
        # 6. Final projection
        x = self.patch_expand(x)
        #pdb.set_trace()
        x = self.out_cov(silu(self.out_norm(x)))
        
        return x


class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings

@register_model
def pccgen(**kwargs):
    layers = [1, 1, 1, 1]
    norm_layer = GroupNorm
    embed_dims = [64, 128, 256, 512]
    mlp_ratios = [8, 8, 4, 4]
    downsamples = [True, True, True, True]
    proposal_w = [2, 2, 2, 2]
    proposal_h = [2, 2, 2, 2]
    proposal_d = [2, 2, 2, 2]
    fold_w = [1, 1, 1, 1]
    fold_h = [1, 1, 1, 1]
    fold_d = [1, 1, 1, 1]
    heads = [4, 4, 8, 8]
    head_dim = [24, 24, 24, 24]
    down_patch_size = 3
    down_pad = 1
    model = ContextCluster(
        layers, embed_dims=embed_dims, norm_layer=norm_layer,
        mlp_ratios=mlp_ratios, downsamples=downsamples,
        down_patch_size=down_patch_size, down_pad=down_pad,
        proposal_w=proposal_w, proposal_h=proposal_h, proposal_d=proposal_d,
        fold_w=fold_w, fold_h=fold_h, fold_d=fold_d,
        heads=heads, head_dim=head_dim,
        **kwargs)

    return model


if __name__ == '__main__':
    # 构造模型参数
    time_embed_dims = [16]  # 时间嵌入维度
    layers=[1, 1, 1, 1]
    norm_layer=GroupNorm
    embed_dims= [64, 128, 256, 512]
    mlp_ratios=[8, 8, 4, 4]
    downsamples=[True, True, True, True]
    proposal_w=[2, 2, 2, 2]
    proposal_h=[2, 2, 2, 2]
    proposal_d=[2, 2, 2, 2]
    fold_w=[1, 1, 1, 1]
    fold_h=[1, 1, 1, 1]
    fold_d=[1, 1, 1, 1]
    heads=[4, 4, 8, 8]
    head_dim=[24, 24, 24, 24]
    down_patch_size=3
    down_pad=1
    with_coord=False

    
    # 构造模型
    model = ContextCluster_Denoised(
            layers=layers, embed_dims=embed_dims, norm_layer=norm_layer,
            mlp_ratios=mlp_ratios, downsamples=downsamples,
            down_patch_size=down_patch_size, down_pad=down_pad,
            proposal_w=proposal_w, proposal_h=proposal_h, proposal_d=proposal_d,
            fold_w=fold_w, fold_h=fold_h, fold_d=fold_d,
            heads=heads, head_dim=head_dim,with_coord=with_coord,time_embed_dims=time_embed_dims
    )

    model = model.cuda()
    x = torch.randn(2,1,64,64,64).cuda()
    timestep = torch.tensor([0,500]).float().cuda()

    output = model(x , timestep)



