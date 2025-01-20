import numpy as np
import torch
import torch.nn as nn
import torch.functional as F
import torch.nn.functional as F_nn
from timm.models.layers import to_3tuple

from torch.nn.functional import silu
from model.image_encoder.u_net import UNet
from model.image_encoder.ResUnet import ResNetGLEncoder
from model.implict_function import get_embedder ,DummyNeRF , Implict_Fuc_Network
import pdb
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers import ModelMixin
from einops import rearrange

def index_2d(feat, uv):
    # https://zhuanlan.zhihu.com/p/137271718
    # feat: [B, C, H, W]
    # uv: [B, N, 2]
    uv = uv.unsqueeze(2) # [B, N, 1, 2]
    feat = feat.transpose(2, 3) # [W, H]
    samples = torch.nn.functional.grid_sample(feat, uv, align_corners=True) # [B, C, N, 1]
    return samples[:, :, :, 0] # [B, C, N]


def weight_init(shape, mode, fan_in, fan_out):
    if mode == 'xavier_uniform': return np.sqrt(6 / (fan_in + fan_out)) * (torch.rand(*shape) * 2 - 1)
    if mode == 'xavier_normal':  return np.sqrt(2 / (fan_in + fan_out)) * torch.randn(*shape)
    if mode == 'kaiming_uniform': return np.sqrt(3 / fan_in) * (torch.rand(*shape) * 2 - 1)
    if mode == 'kaiming_normal':  return np.sqrt(1 / fan_in) * torch.randn(*shape)
    raise ValueError(f'Invalid init mode "{mode}"')

class Linear(torch.nn.Module):
    def __init__(self, in_features, out_features, bias=True, init_mode='kaiming_normal', init_weight=1, init_bias=0):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        init_kwargs = dict(mode=init_mode, fan_in=in_features, fan_out=out_features)
        self.weight = torch.nn.Parameter(weight_init([out_features, in_features], **init_kwargs) * init_weight)
        self.bias = torch.nn.Parameter(weight_init([out_features], **init_kwargs) * init_bias) if bias else None

    def forward(self, x):
        x = x @ self.weight.to(x.dtype).t()
        if self.bias is not None:
            x = x.add_(self.bias.to(x.dtype))
        return x


class Conv3d(torch.nn.Module):
    def __init__(self , 
                 in_channels , out_channels , kernel , bias=True , up=False , down=False,
                 resample_filter=[1,1] , fused_resample=False , init_mode = 'kaiming_normal' , init_weight=1 , init_bias=0,
    ):  
        assert not (up and down)
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.up = up
        self.down = down       
        self.fused_resample = fused_resample

        # init kwargs 
        init_kwargs = dict(mode=init_mode, fan_in=in_channels*kernel*kernel*kernel, fan_out=out_channels*kernel*kernel*kernel)
        # 
        self.weight = torch.nn.Parameter(weight_init([out_channels, in_channels, kernel, kernel , kernel], **init_kwargs) * init_weight) if kernel else None
        self.bias = torch.nn.Parameter(weight_init([out_channels], **init_kwargs) * init_bias) if bias else None

        f = torch.as_tensor(resample_filter, dtype=torch.float32)
        #pdb.set_trace()
        f =  f.ger(f).unsqueeze(0).unsqueeze(1)
        f =  f.unsqueeze(0).repeat(1,1,2,1,1)
        #pdb.set_trace()
        f =  f / f.sum().square()
        self.register_buffer('resample_filter', f if up or down else None)



    def forward(self, x):
        w = self.weight.to(x.dtype) if self.weight is not None else None
        b = self.bias.to(x.dtype) if self.bias is not None else None
        f = self.resample_filter.to(x.dtype) if self.resample_filter is not None else None
        w_pad = w.shape[-1] // 2 if w is not None else 0
        f_pad = (f.shape[-1] - 1) // 2 if f is not None else 0
        #pdb.set_trace()
        if self.fused_resample and self.up and w is not None:
            x = torch.nn.functional.conv_transpose3d(x, f.mul(4).tile([self.in_channels, 1, 1, 1,1]), groups=self.in_channels, stride=2, padding=max(f_pad - w_pad, 0))
            x = torch.nn.functional.conv3d(x, w, padding=max(w_pad - f_pad, 0))
        elif self.fused_resample and self.down and w is not None:
            x = torch.nn.functional.conv3d(x, w, padding=w_pad+f_pad)
            x = torch.nn.functional.conv3d(x, f.tile([self.out_channels, 1, 1, 1,1]), groups=self.out_channels, stride=2)
        else:
            if self.up:
                x = torch.nn.functional.conv_transpose3d(x, f.mul(4).tile([self.in_channels, 1, 1, 1,1]), groups=self.in_channels, stride=2, padding=f_pad)
            if self.down:
                #pdb.set_trace()
                x = torch.nn.functional.conv3d(x, f.tile([self.out_channels,1,1,1,1]), groups=self.out_channels, stride=2 , padding=f_pad)
            if w is not None:
                x = torch.nn.functional.conv3d(x ,w , padding = w_pad )
        if b is not None:
            x = x.add_(b.reshape(1,-1,1,1,1))

        return x 


class GroupNorm(torch.nn.Module):
    def __init__(self, num_channels, num_groups=32, min_channels_per_group=4, eps=1e-5):
        super().__init__()
        self.num_groups = min(num_groups, num_channels // min_channels_per_group)
        self.eps = eps
        self.weight = torch.nn.Parameter(torch.ones(num_channels))
        self.bias = torch.nn.Parameter(torch.zeros(num_channels))

    def forward(self, x):
        x = torch.nn.functional.group_norm(x, num_groups=self.num_groups, weight=self.weight.to(x.dtype), bias=self.bias.to(x.dtype), eps=self.eps)
        return x



class Attention3DOp(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k):
        print(q.shape)  # Should be (n, c, q, r, s)
        print(k.shape)  # Should be (n, c, k, r, s)
        pdb.set_trace()
        w = torch.einsum('ncqrs , nckrs->nqkrs' , q.to(torch.float32) , 
                         (k / np.sqrt(k.shape[1])).to(torch.float32)).softmax(dim=2).to(q.dtype)
        ctx.save_for_backward(q, k, w)
        return w 
    @staticmethod
    def backward(ctx, dw):
        q, k, w = ctx.saved_tensors
        db = torch._softmax_backward_data(
            grad_output=dw.to(torch.float32),
            output=w.to(torch.float32),
            dim=2,
            input_dtype=torch.float32
        )
        dq = torch.einsum('nckrs,nqkrs->ncqrs', k.to(torch.float32), db).to(q.dtype) / np.sqrt(k.shape[1])
        dk = torch.einsum('ncqrs,nqkrs->nckrs', q.to(torch.float32), db).to(k.dtype) / np.sqrt(k.shape[1])
        return dq, dk

class AttentionOp(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k):
        w = torch.einsum('ncq,nck->nqk', q.to(torch.float32), (k / np.sqrt(k.shape[1])).to(torch.float32)).softmax(dim=2).to(q.dtype)
        ctx.save_for_backward(q, k, w)
        return w

    @staticmethod
    def backward(ctx, dw):
        q, k, w = ctx.saved_tensors
        db = torch._softmax_backward_data(grad_output=dw.to(torch.float32), output=w.to(torch.float32), dim=2, input_dtype=torch.float32)
        dq = torch.einsum('nck,nqk->ncq', k.to(torch.float32), db).to(q.dtype) / np.sqrt(k.shape[1])
        dk = torch.einsum('ncq,nqk->nck', q.to(torch.float32), db).to(k.dtype) / np.sqrt(k.shape[1])
        return dq, dk



class UNetBlock3D(torch.nn.Module):
    def __init__(self, in_channels , out_channels , emb_channels , up = False , down = False , attention=False,
                    num_heads=None, channels_per_head=64, dropout=0, skip_scale=1, eps=1e-5,
                    resample_filter=[1,1], resample_proj=False, adaptive_scale=True,
                    init=dict(), init_zero=dict(init_weight=0), init_attn=None
    ):  
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.emb_channels = emb_channels
        self.num_heads = 0 if not attention else num_heads if num_heads is not None else out_channels // channels_per_head
        self.dropout = dropout
        self.skip_scale = skip_scale
        self.adaptive_scale = adaptive_scale

        self.norm0 = GroupNorm(num_channels=in_channels, eps=eps)
        self.conv0 = Conv3d(in_channels=in_channels , out_channels=out_channels , kernel=3 ,
                            up=up , down=down , resample_filter=resample_filter , **init)
        
        self.affine = Linear(in_features=emb_channels, out_features=out_channels*(2 if adaptive_scale else 1), **init)

        self.norm1 = GroupNorm(num_channels=out_channels, eps=eps)
        self.conv1 = Conv3d(in_channels=out_channels , out_channels=out_channels , kernel=3 , **init_zero )

        self.skip = None

        if out_channels != in_channels or up or down:
            kernel = 1 if resample_proj or out_channels!= in_channels else 0
            self.skip = Conv3d(in_channels=in_channels, out_channels=out_channels, kernel=kernel , up=up , down=down , resample_filter=resample_filter, **init_zero)
        if self.num_heads:
            self.norm2 = GroupNorm(num_channels=out_channels , eps=eps)
            self.qkv = Conv3d(in_channels=out_channels , out_channels = out_channels * 3 ,  kernel=1 , **(init_attn if init_attn is not None else init))
            self.proj = Conv3d(in_channels=out_channels , out_channels = out_channels , kernel=1 , **init_zero)
    def forward(self, x , emb):
        #pdb.set_trace()
        orig = x
        x = self.conv0(silu(self.norm0(x)))
        #pdb.set_trace()
        params = self.affine(emb).unsqueeze(2).unsqueeze(3).unsqueeze(4).to(x.dtype)
        if self.adaptive_scale:
            scale, shift = params.chunk(chunks=2, dim=1)
            x = silu(torch.addcmul(shift, self.norm1(x), scale + 1))
        else:
            x = silu(self.norm1(x.add_(params)))

        x = self.conv1(torch.nn.functional.dropout(x, p=self.dropout, training=self.training))
        x = x.add_(self.skip(orig) if self.skip is not None else orig)
        x = x * self.skip_scale

        if self.num_heads:
            q, k, v = self.qkv(self.norm2(x)).reshape(x.shape[0] * self.num_heads, x.shape[1] // self.num_heads, 3, -1).unbind(2)
            w = AttentionOp.apply(q, k)
            a = torch.einsum('nqk,nck->ncq', w, v)
            x = self.proj(a.reshape(*x.shape)).add_(x)
            x = x * self.skip_scale

        return x


class PositionalEmbedding(torch.nn.Module):
    def __init__(self, num_channels, max_positions=10000, endpoint=False):
        super().__init__()
        self.num_channels = num_channels
        self.max_positions = max_positions
        self.endpoint = endpoint

    def forward(self, x):
        freqs = torch.arange(start=0, end=self.num_channels//2, dtype=torch.float32, device=x.device)
        freqs = freqs / (self.num_channels // 2 - (1 if self.endpoint else 0))
        freqs = (1 / self.max_positions) ** freqs
        x = x.ger(freqs.to(x.dtype))
        x = torch.cat([x.cos(), x.sin()], dim=1)
        return x


class FourierEmbedding(torch.nn.Module):
    def __init__(self, num_channels, scale=16):
        super().__init__()
        self.register_buffer('freqs', torch.randn(num_channels // 2) * scale)

    def forward(self, x):
        x = x.ger((2 * np.pi * self.freqs).to(x.dtype))
        x = torch.cat([x.cos(), x.sin()], dim=1)
        return x


class MLP(nn.Module):
    def __init__(self, mlp_list, use_bn=False):
        super().__init__()

        layers = []
        for i in range(len(mlp_list) - 1):
            layers += [nn.Conv2d(mlp_list[i], mlp_list[i + 1], kernel_size=1)]
            if use_bn:
                layers += [nn.BatchNorm2d(mlp_list[i + 1])]
            layers += [nn.LeakyReLU(inplace=True),]
        
        self.layer = nn.Sequential(*layers)

    def forward(self, x):
        return self.layer(x)



def pairwise_cos_sim(x1: torch.Tensor, x2: torch.Tensor):
    """
    return pair-wise similarity matrix between two tensors
    :param x1: [B,...,M,D]
    :param x2: [B,...,N,D]
    :return: similarity matrix [B,...,M,N]
    """
    x1 = F_nn.normalize(x1, dim=-1)
    x2 = F_nn.normalize(x2, dim=-1)

    sim = torch.matmul(x1, x2.transpose(-2, -1))
    return sim

class SpatialGuidedConditionCluster(nn.Module):
    def __init__(self, x_dim , condition_dim , out_dim , 
                 proposal_w=2 , proposal_h=2 , proposal_d=2 , 
                 fold_w = 1 , fold_h = 1 , fold_d = 1,
                 heads = 4 , head_dim=32):
        super().__init__()
        init = dict(init_mode='xavier_uniform')

        self.x_encoder = Conv3d(x_dim , head_dim * heads , kernel=1 , **init )

        self.condition_encoder = Conv3d(condition_dim , head_dim * heads , kernel=1 , **init )

        self.heads = heads

        self.center_proposal = nn.AdaptiveAvgPool3d((proposal_w , proposal_h , proposal_d))

        self.sim_alpha = nn.Parameter(torch.ones(1))

        self.sim_beta = nn.Parameter(torch.zeros(1))

        self.proj = Conv3d(head_dim * heads , out_dim , kernel=1 ,**init)

        self.fold_w = fold_w
        self.fold_h = fold_h
        self.fold_d = fold_d


    def forward(self, x , condition):
        #pdb.set_trace()
        x_feat = self.x_encoder(x)
        condition_feat  = self.condition_encoder(condition)
        x_feat = rearrange(x_feat , 'b (e c) h w d -> (b e) c w h d' , e=self.heads)
        condition_feat = rearrange(condition_feat , 'b (e c) h w d -> (b e) c w h d' , e=self.heads)

        if self.fold_w > 1 and self.fold_h > 1:
            # split the big feature maps to small local regions to reduce computations 
            _, _ , w0 , h0 , d0  =   x_feat.shape
            assert w0 % self.fold_w == 0 and h0 % self.fold_h == 0 and d0 % self.fold_d == 0, \
                f"Ensure the feature map size ({w0}*{h0}*{w0}) can be divided by fold " \
                f"{self.fold_w}*{self.fold_h}*{self.fold_d}"
            #pdb.set_trace()
            x_feat = rearrange(x_feat, "b c (f1 w) (f2 h) (f3 d) -> (b f1 f2 f3) c w h d", f1=self.fold_w,
                          f2=self.fold_h, f3=self.fold_d)  # [bs*blocks,c,ks[0],ks[1],ks[2]]
            condition_feat = rearrange(condition_feat, "b c (f1 w) (f2 h) (f3 d) -> (b f1 f2 f3) c w h d", f1=self.fold_w,
                    f2=self.fold_h, f3=self.fold_d)

        B ,C ,W ,H, D = x_feat.shape
        centers = self.center_proposal(x_feat)
        condition_centers = rearrange(self.center_proposal(condition_feat) , 'b c w h d -> b ( w h d ) c ' )

        sim = torch.sigmoid(
            self.sim_beta + 
            self.sim_alpha * pairwise_cos_sim(
                centers.reshape(B,C,-1).permute(0,2,1),
                condition_feat.reshape(B,C,-1).permute(0,2,1)
            )
        )

        sim_max , sim_max_idx = sim.max(dim=1, keepdim=True)
        mask = torch.zeros_like(sim)
        mask.scatter_(1,sim_max_idx,1.)
        sim = sim * mask

        condition_feat_for_weight = rearrange(condition_feat , 'b c w h d -> b (w h d) c')

        apply_sim_weight = ((condition_feat_for_weight.unsqueeze(dim=1) * sim.unsqueeze(dim=-1)).sum(dim=2) + condition_centers) / (
                             sim.sum(dim=-1 , keepdim=True) + 1.0)
        
        out = (apply_sim_weight.unsqueeze(dim=2) * sim.unsqueeze(dim=-1)).sum(dim=1)
        out = rearrange(out , "b ( w h d ) c -> b c w h d" , w=W , h=H)

        out = rearrange(out , '(b e) c w h d -> b (e c) w h d' , e = self.heads)
        out = self.proj(out)

        return out


class Song_Unet3D(ModelMixin , ConfigMixin):
    @register_to_config
    def __init__(self,
        img_resolution,                     # Image resolution at input/output.
        in_channels,                        # Number of color channels at input.
        out_channels,                       # Number of color channels at output.
        condition_mixer_out_channels,       # Number of summed channels at noise, coords , condition 
        label_dim           = 0,            # Number of class labels, 0 = unconditional.
        augment_dim         = 0,            # Augmentation label dimensionality, 0 = no augmentation.

        model_channels      = 128,          # Base multiplier for the number of channels.
        channel_mult        = [1,2,2,2],    # Per-resolution multipliers for the number of channels.
        channel_mult_emb    = 4,            # Multiplier for the dimensionality of the embedding vector.
        num_blocks          = 4,            # Number of residual blocks per resolution.
        attn_resolutions    = [16],         # List of resolutions with self-attention.
        dropout             = 0.10,         # Dropout probability of intermediate activations.
        label_dropout       = 0,            # Dropout probability of class labels for classifier-free guidance.

        embedding_type      = 'positional', # Timestep embedding type: 'positional' for DDPM++, 'fourier' for NCSN++.
        channel_mult_noise  = 1,            # Timestep embedding size: 1 for DDPM++, 2 for NCSN++.
        encoder_type        = 'standard',   # Encoder architecture: 'standard' for DDPM++, 'residual' for NCSN++.
        decoder_type        = 'standard',   # Decoder architecture: 'standard' for both DDPM++ and NCSN++.
        resample_filter     = [1,1],        # Resampling filter: [1,1] for DDPM++, [1,3,3,1] for NCSN++.
        implicit_mlp        = False,        # enable implicit coordinate encoding
        implict_condition_dim = [64,128,256] ,       # implict condition input dim 
        #image_encoder setting 
        img_channels = 3 ,
        latent_dim = 512,
        input_img_size = 256,
        encoder_freeze_layer = "layer1", 
        feature_layer = "layer3",
        global_feature_layer = "layer4",
        global_feature_layer_last = 22 ,
        pretrained =  "imagenet",
        weight_dir = '',
        image_encoder_output  = 64 , 
        bilinear = False,
        model_name = 'resnet101',
        #implict_func_model setting 
        pos_dim = 63 ,
        local_f_dim = 64 , 
        num_layer = 4 ,
        hidden_dim = 256 ,
        output_dim = 128 ,
        skips = [2] ,
        last_activation = 'relu',
        use_silu = False , 
        no_activation = False,       
        #spatial_guidede_cluster setting
        proposal_d=2,
        proposal_h=2,
        proposal_w=2,
        heads=4,     #   
        head_dim=16  #  heads * head_dim  
  
    ):
        super(Song_Unet3D , self).__init__()
        
        #self.image_feature_extractor = UNet(n_channels=1 , n_classes=image_encoder_output , bilinear=bilinear)

        self.image_feature_extractor = ResNetGLEncoder( in_channels= img_channels , latent_dim=latent_dim ,input_img_size=input_img_size , 
                                                        encoder_freeze_layer = encoder_freeze_layer , feature_layer = feature_layer , 
                                                        global_feature_layer = global_feature_layer , global_feature_layer_last=global_feature_layer_last , pretrained = pretrained ,
                                                        weight_dir = weight_dir  , bilinear= bilinear , n_classed=image_encoder_output ,
                                                        model_name = model_name   )

        self.combine = 'mlp'
        self.view_mixer = MLP([2, 2 // 2, 1])

        self.pos_implic , _  = get_embedder(multires=10 , i=0)

        #self.implict_fn = DummyNeRF(input_ch=127 , output_ch=128)

        self.implict_fn = Implict_Fuc_Network(pos_dim=pos_dim ,local_f_dim=local_f_dim , num_layer=num_layer , 
                                              hidden_dim=hidden_dim , output_dim=output_dim , skips=skips ,
                                              last_activation=last_activation , use_silu=use_silu , 
                                              no_activation=no_activation)
        


        #self.model = SongUNet3D(
        self.model = SongUNet3DV3(
            img_resolution = img_resolution,
            in_channels = in_channels,
            out_channels = out_channels,
            condition_mixer_out_channels = condition_mixer_out_channels,
            label_dim = label_dim,
            model_channels = model_channels,
            channel_mult = channel_mult,
            channel_mult_emb = channel_mult_emb,
            num_blocks = num_blocks,
            attn_resolutions = attn_resolutions,
            dropout = dropout,
            label_dropout = label_dropout,
            embedding_type = embedding_type,
            channel_mult_noise = channel_mult_noise,
            encoder_type = encoder_type,
            decoder_type = decoder_type,
            resample_filter = resample_filter,
            implicit_mlp = implicit_mlp,
            implict_condition_dim= implict_condition_dim,
            proposal_d=proposal_d,
            proposal_h=proposal_h,
            proposal_w=proposal_w,
            heads=heads,     #   
            head_dim=head_dim  #  heads * head_dim  
        )      

    def condition_process(self, projs , projs_points , coords_global):
        #pdb.set_trace()
        b , m , c , w , h = projs.shape
        projs = projs.reshape(b * m, c, w, h) # B', C, W, H
        _ , _ , v_w , v_h ,v_d = coords_global.shape
        coords_global = coords_global.reshape(b,3,-1)
        #pdb.set_trace()
        proj_feats , global_feats  = self.image_feature_extractor(projs)
        #pdb.set_trace()
        global_feats = global_feats.squeeze(-1).squeeze(-1)
        global_feats = global_feats.unsqueeze(1) #  B' 1  C
        view_global_feats = []
        globa_feats_ch = global_feats.shape[2]
        for i in range(0,(b*m) , m):
            #pdb.set_trace()
            view_g = global_feats[i:i+m]
            # 去掉中间的维度1
            view_g = view_g.squeeze(1)  # [m, c] 
            # 重塑并合并特征
            view_g = view_g.reshape(1, 1, m * globa_feats_ch)  # [1, 1, m*c]
            #pdb.set_trace()
            view_global_feats.append(view_g)
             
        global_feats = torch.cat(view_global_feats, dim=0)

        global_feats = global_feats.repeat(1,projs_points.shape[2],1)

        #pdb.set_trace()
        proj_feats = list(proj_feats) if type(proj_feats) is tuple else [proj_feats]
        for i in range(len(proj_feats)):
            _, c_, w_, h_ = proj_feats[i].shape
            proj_feats[i] = proj_feats[i].reshape(b, m, c_, w_, h_) # B, M, C, W, H
        
        #pdb.set_trace()
   
        points_feats = self.forward_points(proj_feats , projs_points)
        #pdb.set_trace()
        coords_global = coords_global.permute(0,2,1)
        pos_embed = self.pos_implic(coords_global)
        #pdb.set_trace()
        points_feats = points_feats.permute(0,2,1)
        pdb.set_trace()
        p_condition = self.implict_fn(pos_embed , points_feats , global_feats)
        p_condition = p_condition.permute(0,2,1)
        #pdb.set_trace()
        p_condition = p_condition.reshape(b, -1, v_w, v_h ,v_d)

        return p_condition


    def forward_points(self, proj_feats , projs_points):
        n_view = proj_feats[0].shape[1]
        # 1. query view-specific features
        p_list = []
        #pdb.set_trace()
        for i in range(n_view):
            f_list = []
            for proj_f in proj_feats:
                #pdb.set_trace()
                feat = proj_f[:, i, ...] # B, C, W, H
                p = projs_points[:, i, ...] # B, N, 2
                #pdb.set_trace()
                p_feats = index_2d(feat, p) # B, C, N
                f_list.append(p_feats)
            p_feats = torch.cat(f_list, dim=1)
            p_list.append(p_feats)
        p_feats = torch.stack(p_list, dim=-1) # B, C, N, M
        #pdb.set_trace()
         # 2. cross-view fusion
        if self.combine == 'max':
            p_feats = F.max_pool2d(p_feats, (1, n_view))
            p_feats = p_feats.squeeze(-1) # B, C, N
            condition_feats = p_feats  # 保存用于condition的特征
        elif self.combine == 'mlp':
            #pdb.set_trace()
            p_feats = p_feats.permute(0, 3, 1, 2)
            condition_feats = self.view_mixer(p_feats)  # 保存MLP输出的特征
            p_feats = condition_feats.squeeze(1)

        return p_feats

    def forward(self , x , time_step , projs , projs_points , class_labels=None):
        #pdb.set_trace()
        b , c , h , w ,d = x.shape
        coords_global = x[:,1:4,...]
        points_implict = self.condition_process(projs , projs_points , coords_global)
        #pdb.set_trace()
        pred = self.model(x , time_step , points_implict ,class_labels=class_labels)
        return pred


class ImplictConditionDownSample(nn.Module):
    """
    Point Reducer is implemented by a layer of conv since it is mathmatically equal.
    Input: tensor in shape [B, in_chans, H, W, D]
    Output: tensor in shape [B, embed_dim, H/stride, W/stride, D/stride]
    """

    def __init__(self, patch_size=2, stride=2, padding=0,
                 in_chans=3, embed_dim=768, norm_layer=None):
        super().__init__()
        patch_size = to_3tuple(patch_size)
        stride = to_3tuple(stride)
        padding = to_3tuple(padding)
        #pdb.set_trace()
        self.proj = nn.Conv3d(in_chans, embed_dim, kernel_size=patch_size, stride=stride, padding=padding)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        x = self.proj(x)
        x = self.norm(x)
        return x 

# V3 change process condition and noise x order 
# V2 first  use init condition mixer process conditions like coords or implicts function representation
# V3 添加splial condition cluster 
class SongUNet3DV3(torch.nn.Module):
    def __init__(self,
        img_resolution,                     # 3D体积分辨率
        in_channels,                        # 输入通道数
        out_channels,                       # 输出通道数
        condition_mixer_out_channels,       # condition_mixer输出的特征通道数
        label_dim           = 0,            # 类别标签维度
        augment_dim         = 0,            # 增强标签维度
        model_channels      = 128,          # 基础通道数
        channel_mult        = [1,2,2,2],    # 通道数倍增
        channel_mult_emb    = 2,            # 嵌入维度倍增
        num_blocks          = 4,            # 每个分辨率的残差块数
        attn_resolutions    = [16],         # 使用自注意力的分辨率
        dropout             = 0.10,         # dropout率
        label_dropout       = 0,            # 标签dropout率
        embedding_type      = 'positional', 
        channel_mult_noise  = 1,            
        encoder_type        = 'standard',   
        decoder_type        = 'standard',   
        resample_filter     = [1,1],      # 扩展到3D的重采样滤波器
        implicit_mlp        = False,
        implict_condition_dim = [64 , 128 , 256 ],
        #spatial_guidede_cluster setting 
        proposal_d=2,
        proposal_h=2,
        proposal_w=2,
        heads=4,     #   
        head_dim=16  #  heads * head_dim  
    ):
        super(SongUNet3DV3, self).__init__()
        assert embedding_type in ['fourier', 'positional']
        assert encoder_type in ['standard', 'skip', 'residual']
        assert decoder_type in ['standard', 'skip']
        self.label_dropout = label_dropout
        #pdb.set_trace()
        emb_channels = model_channels * channel_mult_emb
        noise_channels = model_channels * channel_mult_noise
        init = dict(init_mode='xavier_uniform')
        init_zero = dict(init_mode='xavier_uniform', init_weight=1e-5)
        init_attn = dict(init_mode='xavier_uniform', init_weight=np.sqrt(0.2))
        block_kwargs = dict(
            emb_channels=emb_channels, num_heads=1, dropout=dropout,
            skip_scale=np.sqrt(0.5), eps=1e-6,
            resample_filter=resample_filter, resample_proj=True,
            adaptive_scale=False, init=init, init_zero=init_zero,
            init_attn=init_attn,
        )

        self.map_noise = PositionalEmbedding(num_channels=noise_channels, endpoint=True) if embedding_type == 'positional' else FourierEmbedding(num_channels=noise_channels)
        self.map_label = Linear(in_features=label_dim, out_features=noise_channels, **init) if label_dim else None
        self.map_augment = Linear(in_features=augment_dim, out_features=noise_channels, bias=False, **init) if augment_dim else None
        #pdb.set_trace() 
        self.map_layer0 = Linear(in_features=noise_channels, out_features=emb_channels, **init)
        #self.map_layer1 = Linear(in_features=emb_channels, out_features=emb_channels, **init)
        self.img_resolution = img_resolution
        # 添加condition处理层
        self.implict_dim = implict_condition_dim
        self.channel_mult = channel_mult
        #pdb.set_trace()
        # 修改condition处理层，让其更好地与输入特征融合
        self.spatial_guided_cluster = SpatialGuidedConditionCluster(
            x_dim=in_channels ,
            condition_dim = self.implict_dim[0] , 
            out_dim = condition_mixer_out_channels , 
            proposal_d=proposal_d,
            proposal_h=proposal_h,
            proposal_w=proposal_w,
            heads=heads,
            head_dim=head_dim
        )
        self.condition_norm = GroupNorm(num_channels=condition_mixer_out_channels , eps=1e-6)
        self.multi_condition_down = nn.ModuleDict()
        for level , mult in enumerate(channel_mult[1:] , start=1):
            res = img_resolution >> level 
            self.multi_condition_down[f'{res}x{res}x{res}'] = ImplictConditionDownSample(
                patch_size=2,
                stride=2, 
                in_chans=self.implict_dim[level-1] ,  # 64 128 
                embed_dim=self.implict_dim[level], # 128 256   
            )

        #pdb.set_trace()
        self.multi_spatial_guided_cluster = nn.ModuleDict()
        for level , mult in enumerate(channel_mult[1:] , start=1):
            #pdb.set_trace()
            res = img_resolution >> level
            self.multi_spatial_guided_cluster[f"{res}x{res}x{res}"] = SpatialGuidedConditionCluster(
                x_dim = model_channels * channel_mult[level - 1],  # 128 256  
                condition_dim = self.implict_dim[level] ,  # 128 256 
                out_dim = self.implict_dim[level],  # 128 256 
                proposal_d=proposal_d,
                proposal_h=proposal_h,
                proposal_w=proposal_w,
                heads=heads,
                head_dim=head_dim
            )
        #pdb.set_trace()
        self.multi_condition_norm = nn.ModuleDict()
        for level , mult in enumerate(channel_mult[1:] , start=1):
            res = img_resolution >> level
            self.multi_condition_norm[f'{res}x{res}x{res}'] = GroupNorm(
                num_channels=self.implict_dim[level] ,
                eps=1e-6
            )
        

        # self.final_mixer = nn.Sequential(
        #     Conv3d(in_channels + condition_mixer_out_channels , condition_mixer_out_channels, kernel=1 , **init),
        #     nn.SiLU(),
        # )

        # Modified mapping layers to include condition
    
        self.enc = torch.nn.ModuleDict()
        self.dec = torch.nn.ModuleDict()
        if condition_mixer_out_channels is not None:
            cout = condition_mixer_out_channels     
            caux = condition_mixer_out_channels
        else:
            cout = in_channels
            caux = in_channels
        for level, mult in enumerate(channel_mult):
            res = img_resolution >> level
            #print(" level : " , level)
            #print(' multi :' , mult)
            if level == 0:
                cin = cout + 4 
                cout = model_channels
                if implicit_mlp:
                    self.enc[f'{res}x{res}x{res}_conv'] = torch.nn.Sequential(
                                                            Conv3d(in_channels=cin, out_channels=cout, kernel=1, **init),
                                                            torch.nn.SiLU(),
                                                            Conv3d(in_channels=cout, out_channels=cout, kernel=1, **init),
                                                            torch.nn.SiLU(),
                                                            Conv3d(in_channels=cout, out_channels=cout, kernel=1, **init),
                                                            torch.nn.SiLU(),
                                                            Conv3d(in_channels=cout, out_channels=cout, kernel=3, **init),
                                                    )
                    self.enc[f'{res}x{res}x{res}_conv'].out_channels = cout
                else:
                    self.enc[f'{res}x{res}x{res}_conv'] = Conv3d(in_channels=cin, out_channels=cout, kernel=3, **init)
                for idx in range(0 ,num_blocks):
                    cin =  cout
                    cout = model_channels * mult
                    attn = (res in attn_resolutions)
                    self.enc[f'{res}x{res}x{res}_block{idx}'] = UNetBlock3D(in_channels=cin, out_channels=cout, attention=attn, **block_kwargs)
            else:
                self.enc[f'{res}x{res}x{res}_down'] = UNetBlock3D(in_channels=cout, out_channels=cout, down=True, **block_kwargs)
                if encoder_type == 'skip':
                    self.enc[f'{res}x{res}x{res}_aux_down'] = Conv3d(in_channels=caux, out_channels=caux, kernel=0, down=True, resample_filter=resample_filter)
                    self.enc[f'{res}x{res}x{res}_aux_skip'] = Conv3d(in_channels=caux, out_channels=cout, kernel=1, **init)
                if encoder_type == 'residual':
                    self.enc[f'{res}x{res}x{res}_aux_residual'] = Conv3d(in_channels=caux, out_channels=cout, kernel=3, down=True, resample_filter=resample_filter, fused_resample=True, **init)
                    caux = cout
            #pdb.set_trace()
                cin  = cout + self.implict_dim[level]
                cout = model_channels * mult
                attn = (res in attn_resolutions)
                self.enc[f'{res}x{res}x{res}_block0'] = UNetBlock3D(in_channels=cin , out_channels=cout , attention=attn , **block_kwargs)
                for idx in range(1 ,num_blocks):
                #for idx in range(0 ,num_blocks):
                    cin =  cout
                    cout = model_channels * mult
                    attn = (res in attn_resolutions)
                    self.enc[f'{res}x{res}x{res}_block{idx}'] = UNetBlock3D(in_channels=cin, out_channels=cout, attention=attn, **block_kwargs)
        #pdb.set_trace()
        skips = [block.out_channels for name, block in self.enc.items() if 'aux' not in name]
        # Decoder.
        #pdb.set_trace()
        self.dec = torch.nn.ModuleDict()
        for level, mult in reversed(list(enumerate(channel_mult))):
            print(" res :" , f"{res}x{res}x{res}")
            res = img_resolution >> level
            if level == len(channel_mult) - 1:
                self.dec[f'{res}x{res}x{res}_in0'] = UNetBlock3D(in_channels=cout, out_channels=cout, attention=True, **block_kwargs)
                self.dec[f'{res}x{res}x{res}_in1'] = UNetBlock3D(in_channels=cout, out_channels=cout, **block_kwargs)
            else:
                self.dec[f'{res}x{res}x{res}_up'] = UNetBlock3D(in_channels=cout, out_channels=cout, up=True, **block_kwargs)
            #pdb.set_trace()
            for idx in range(num_blocks + 1):
                cin = cout + skips.pop()
                cout = model_channels * mult
                attn = (idx == num_blocks and res in attn_resolutions)
                self.dec[f'{res}x{res}x{res}_block{idx}'] = UNetBlock3D(in_channels=cin, out_channels=cout, attention=attn, **block_kwargs)
            #pdb.set_trace()
            if decoder_type == 'skip' or level == 0:
                if decoder_type == 'skip' and level < len(channel_mult) - 1:
                    self.dec[f'{res}x{res}x{res}_aux_up'] = Conv3d(in_channels=out_channels, out_channels=out_channels, kernel=0, up=True, resample_filter=resample_filter)
                self.dec[f'{res}x{res}x{res}_aux_norm'] = GroupNorm(num_channels=cout, eps=1e-6)
                self.dec[f'{res}x{res}x{res}_aux_conv'] = Conv3d(in_channels=cout, out_channels=out_channels, kernel=3, **init_zero)      
        #pdb.set_trace()
    def forward(self, x, noise_labels, condition , class_labels=None, augment_labels=None):
        #pdb.set_trace()
        if condition is not None:
            # use spatial_guided_cluster process condition 
            #pdb.set_trace()
            guided_condition = self.spatial_guided_cluster(x, condition)
            # need use group norm condition 
            guided_condition = silu(self.condition_norm(guided_condition))
            #x_combined = torch.cat([x , guided_condition], dim=1)
            x = torch.cat([x , guided_condition], dim=1)
            #pdb.set_trace()
            #x = self.final_mixer(x_combined)
        #pdb.set_trace()
        
        multi_scale_conditions = {}
        #pdb.set_trace()
        #逐步下采样并存储condition特征
        current_condition = condition
        for level, mult in enumerate(self.channel_mult[1:], start=1):
            res = self.img_resolution >> level
            res_key = f"{res}x{res}x{res}"
            
            # 下采样condition
            current_condition = self.multi_condition_down[res_key](current_condition)
            multi_scale_conditions[res_key] = current_condition
        #pdb.set_trace()
        # Mapping
        emb = self.map_noise(noise_labels) # time embedding 
        emb = emb.reshape(emb.shape[0], 2, -1).flip(1).reshape(*emb.shape) # swap sin/cos
        if self.map_label is not None:
            tmp = class_labels
            if self.training and self.label_dropout:
                tmp = tmp * (torch.rand([x.shape[0], 1], device=x.device) >= self.label_dropout).to(tmp.dtype)
            emb = emb + self.map_label(tmp * np.sqrt(self.map_label.in_features))
        if self.map_augment is not None and augment_labels is not None:
            emb = emb + self.map_augment(augment_labels)
        #pdb.set_trace()
        emb = silu(self.map_layer0(emb)) # 1 256 
        #emb = silu(self.map_layer1(emb)) # 1 512   # t embedding     
        #pdb.set_trace()
        
        #pdb.set_trace()
        # Encoder.
        skips = []
        aux = x
        current_level = 0
        down_block_passed = False

        #pdb.set_trace()
        for name, block in self.enc.items():
            #print(" name :" , name)
            if 'aux_down' in name:
                aux = block(aux)
            elif 'aux_skip' in name:
                x = skips[-1] = x + block(aux)
            elif 'aux_residual' in name:
                x = skips[-1] = aux = (x + block(aux)) / np.sqrt(2)
            else:
                #pdb.set_trace()
                if name.split('_')[-1] == 'down':
                    x = block(x, emb) if isinstance(block, UNetBlock3D) else block(x)
                    skips.append(x)
                    current_level += 1
                    down_block_passed = True
                else:
                    if down_block_passed and 'block0' in name:
                        curr_res = self.img_resolution >> current_level
                        res_key = f"{curr_res}x{curr_res}x{curr_res}"
                        if res_key in self.multi_spatial_guided_cluster:
                            #pdb.set_trace()
                            enhanced_condition = self.multi_spatial_guided_cluster[res_key](
                                x, multi_scale_conditions[res_key])
                            enhanced_condition = silu(self.multi_condition_norm[res_key](enhanced_condition))
                            #pdb.set_trace()
                            #? 先使用add 
                            x = torch.cat([x, enhanced_condition], dim=1)
                        down_block_passed = False
                    #pdb.set_trace()
                    x = block(x, emb) if isinstance(block, UNetBlock3D) else block(x)
                    skips.append(x)
        # pdb.set_trace()
        # Decoder.
        aux = None
        tmp = None
        for name, block in self.dec.items():
            #print(" name  " , name)
            if 'aux_up' in name:
                aux = block(aux)
            elif 'aux_norm' in name:
                tmp = block(x)
            elif 'aux_conv' in name:
                tmp = block(silu(tmp))
                aux = tmp if aux is None else tmp + aux
            else:
                if x.shape[1] != block.in_channels:
                    x = torch.cat([x, skips.pop()], dim=1)
                x = block(x, emb)
        #print("last output " , x.shape)
        #pdb.set_trace()
        return aux
def test_demo_unet3d():
    # 设置基本参数
    vol_resolution = 64    # 3D体积分辨率
    in_channels = 4       # 输入通道数(包含了图像和坐标信息)
    out_channels = 1      # 输出通道数
    condition_mixer_out_channels = 64  # condition处理后的通道数
    implict_condition_dim = 64        # condition输入的通道数
    
    # 创建模型
    model = SongUNet3DV3(
        img_resolution = vol_resolution,
        in_channels = in_channels,
        out_channels = out_channels,
        condition_mixer_out_channels = condition_mixer_out_channels,
        label_dim = 0,
        model_channels = 128,          
        channel_mult = [1,2,2],      
        channel_mult_emb = 2,          
        num_blocks = 2,                
        attn_resolutions = [16],       
        dropout = 0.1,                 
        label_dropout = 0.1,
        implict_condition_dim = implict_condition_dim
    )
    
    # 将模型移动到GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    # 创建测试数据
    batch_size = 1
    # 创建输入数据 x (包含图像和坐标信息)
    x = torch.randn(batch_size, in_channels, vol_resolution, vol_resolution, vol_resolution).to(device)
    
    # 创建时间步信息
    noise_labels = torch.randn(batch_size).to(device)
    
    # 创建condition数据
    condition = torch.randn(batch_size, implict_condition_dim, vol_resolution, vol_resolution, vol_resolution).to(device)
    
    # 前向传播
    print("Input shape:", x.shape)
    print("Condition shape:", condition.shape)
    
    output = model(x, noise_labels, condition)
    
    print("Output shape:", output.shape)
    
    # 验证输出形状
    expected_shape = (batch_size, out_channels, vol_resolution, vol_resolution, vol_resolution)
    assert output.shape == expected_shape, f"Output shape {output.shape} doesn't match expected shape {expected_shape}"
    
    print("Test passed successfully!")    
# main 
if __name__ == '__main__':
   #test_demo_unet2d()
   test_demo_unet3d()