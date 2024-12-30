import numpy as np
import torch
from torch.nn.functional import silu
import pdb
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers import ModelMixin

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
                x = torch.nn.functional.conv3d(x ,w , padding=w_pad )
        if b is not None:
            x = x.add_(b.reshape(1,-1,1,1,1))

        return x 



class Conv2d(torch.nn.Module):
    def __init__(self,
        in_channels, out_channels, kernel, bias=True, up=False, down=False,
        resample_filter=[1,1], fused_resample=False, init_mode='kaiming_normal', init_weight=1, init_bias=0,
    ):
        assert not (up and down)
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.up = up
        self.down = down
        self.fused_resample = fused_resample
        init_kwargs = dict(mode=init_mode, fan_in=in_channels*kernel*kernel, fan_out=out_channels*kernel*kernel)
        self.weight = torch.nn.Parameter(weight_init([out_channels, in_channels, kernel, kernel], **init_kwargs) * init_weight) if kernel else None
        self.bias = torch.nn.Parameter(weight_init([out_channels], **init_kwargs) * init_bias) if kernel and bias else None
        #pdb.set_trace()
        f = torch.as_tensor(resample_filter, dtype=torch.float32)
        f = f.ger(f).unsqueeze(0).unsqueeze(1) / f.sum().square()  # 1,1,2,2 
        self.register_buffer('resample_filter', f if up or down else None)

    def forward(self, x):
        w = self.weight.to(x.dtype) if self.weight is not None else None
        b = self.bias.to(x.dtype) if self.bias is not None else None
        f = self.resample_filter.to(x.dtype) if self.resample_filter is not None else None
        w_pad = w.shape[-1] // 2 if w is not None else 0
        f_pad = (f.shape[-1] - 1) // 2 if f is not None else 0

        if self.fused_resample and self.up and w is not None:
            x = torch.nn.functional.conv_transpose2d(x, f.mul(4).tile([self.in_channels, 1, 1, 1]), groups=self.in_channels, stride=2, padding=max(f_pad - w_pad, 0))
            x = torch.nn.functional.conv2d(x, w, padding=max(w_pad - f_pad, 0))
        elif self.fused_resample and self.down and w is not None:
            x = torch.nn.functional.conv2d(x, w, padding=w_pad+f_pad)
            x = torch.nn.functional.conv2d(x, f.tile([self.out_channels, 1, 1, 1]), groups=self.out_channels, stride=2)
        else:
            if self.up:
                x = torch.nn.functional.conv_transpose2d(x, f.mul(4).tile([self.in_channels, 1, 1, 1]), groups=self.in_channels, stride=2, padding=f_pad)
            if self.down:
                pdb.set_trace()  
                x = torch.nn.functional.conv2d(x, f.tile([self.in_channels, 1, 1, 1]), groups=self.in_channels, stride=2, padding=f_pad)   # f.tile  128 1 2 2 
            if w is not None:  
                x = torch.nn.functional.conv2d(x, w, padding=w_pad)
        if b is not None:
            x = x.add_(b.reshape(1, -1, 1, 1))
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



class UNetBlock(torch.nn.Module):
    def __init__(self,
        in_channels, out_channels, emb_channels, up=False, down=False, attention=False,
        num_heads=None, channels_per_head=64, dropout=0, skip_scale=1, eps=1e-5,
        resample_filter=[1,1], resample_proj=False, adaptive_scale=True,
        init=dict(), init_zero=dict(init_weight=0), init_attn=None,
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
        self.conv0 = Conv2d(in_channels=in_channels, out_channels=out_channels, kernel=3, up=up, down=down, resample_filter=resample_filter, **init)
        self.affine = Linear(in_features=emb_channels, out_features=out_channels*(2 if adaptive_scale else 1), **init)
        self.norm1 = GroupNorm(num_channels=out_channels, eps=eps)
        self.conv1 = Conv2d(in_channels=out_channels, out_channels=out_channels, kernel=3, **init_zero)

        self.skip = None
        if out_channels != in_channels or up or down:
            kernel = 1 if resample_proj or out_channels!= in_channels else 0
            self.skip = Conv2d(in_channels=in_channels, out_channels=out_channels, kernel=kernel, up=up, down=down, resample_filter=resample_filter, **init)

        if self.num_heads:
            self.norm2 = GroupNorm(num_channels=out_channels, eps=eps)
            self.qkv = Conv2d(in_channels=out_channels, out_channels=out_channels*3, kernel=1, **(init_attn if init_attn is not None else init))
            self.proj = Conv2d(in_channels=out_channels, out_channels=out_channels, kernel=1, **init_zero)

    def forward(self, x, emb):
        orig = x
        x = self.conv0(silu(self.norm0(x)))

        params = self.affine(emb).unsqueeze(2).unsqueeze(3).to(x.dtype)
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

class Song_Unet3D(ModelMixin , ConfigMixin):
    @register_to_config
    def __init__(self,
        img_resolution,                     # Image resolution at input/output.
        in_channels,                        # Number of color channels at input.
        out_channels,                       # Number of color channels at output.
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
        mode_type           = '2d'
    ):
        super(Song_Unet3D , self).__init__()
        self.model = SongUNet3D(
            img_resolution = img_resolution,
            in_channels = in_channels,
            out_channels = out_channels,
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
            implicit_mlp = implicit_mlp
        )      
    def forward(self , x , time_step , class_labels=None):
        pred = self.model(x , time_step , class_labels=class_labels)
        return pred


class Slice_UNet(ModelMixin, ConfigMixin):
    @register_to_config
    def __init__(self ,
        img_resolution,                     # Image resolution at input/output.
        in_channels,                        # Number of color channels at input.
        out_channels,                       # Number of color channels at output.
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
        mode_type           = '2d'
    ):  
        super(Slice_UNet, self).__init__()
        
        #pdb.set_trace()
        self.model = SongUNet(
            img_resolution = img_resolution,
            in_channels = in_channels,
            out_channels = out_channels,
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
            implicit_mlp = implicit_mlp)
    def forward(self , x , time_step , class_labels=None):

        pred = self.model(x , time_step , class_labels=class_labels)
        return pred

class SongUNet3D(torch.nn.Module):
    def __init__(self,
        img_resolution,                     # 3D体积分辨率
        in_channels,                        # 输入通道数
        out_channels,                       # 输出通道数
        label_dim           = 0,            # 类别标签维度
        augment_dim         = 0,            # 增强标签维度
        model_channels      = 128,          # 基础通道数
        channel_mult        = [1,2,2,2],    # 通道数倍增
        channel_mult_emb    = 4,            # 嵌入维度倍增
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
    ):
        super(SongUNet3D, self).__init__()
        assert embedding_type in ['fourier', 'positional']
        assert encoder_type in ['standard', 'skip', 'residual']
        assert decoder_type in ['standard', 'skip']
        self.label_dropout = label_dropout
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
        self.map_layer0 = Linear(in_features=noise_channels, out_features=emb_channels, **init)
        self.map_layer1 = Linear(in_features=emb_channels, out_features=emb_channels, **init)
    
        self.enc = torch.nn.ModuleDict()
        self.dec = torch.nn.ModuleDict()

        cout = in_channels
        caux = in_channels
        for level, mult in enumerate(channel_mult):
            res = img_resolution >> level
            print(" level : " , level)
            print(' multi :' , mult)
            if level == 0:
                cin = cout
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
            else:
                self.enc[f'{res}x{res}x{res}_down'] = UNetBlock3D(in_channels=cout, out_channels=cout, down=True, **block_kwargs)
                if encoder_type == 'skip':
                    self.enc[f'{res}x{res}x{res}_aux_down'] = Conv3d(in_channels=caux, out_channels=caux, kernel=0, down=True, resample_filter=resample_filter)
                    self.enc[f'{res}x{res}x{res}_aux_skip'] = Conv3d(in_channels=caux, out_channels=cout, kernel=1, **init)
                if encoder_type == 'residual':
                    self.enc[f'{res}x{res}x{res}_aux_residual'] = Conv3d(in_channels=caux, out_channels=cout, kernel=3, down=True, resample_filter=resample_filter, fused_resample=True, **init)
                    caux = cout
            for idx in range(num_blocks):
                cin = cout
                cout = model_channels * mult
                attn = (res in attn_resolutions)
                self.enc[f'{res}x{res}x{res}_block{idx}'] = UNetBlock3D(in_channels=cin, out_channels=cout, attention=attn, **block_kwargs)
        skips = [block.out_channels for name, block in self.enc.items() if 'aux' not in name]

        # Decoder.
        self.dec = torch.nn.ModuleDict()
        for level, mult in reversed(list(enumerate(channel_mult))):
            res = img_resolution >> level
            if level == len(channel_mult) - 1:
                self.dec[f'{res}x{res}x{res}_in0'] = UNetBlock3D(in_channels=cout, out_channels=cout, attention=True, **block_kwargs)
                self.dec[f'{res}x{res}x{res}_in1'] = UNetBlock3D(in_channels=cout, out_channels=cout, **block_kwargs)
            else:
                self.dec[f'{res}x{res}x{res}_up'] = UNetBlock3D(in_channels=cout, out_channels=cout, up=True, **block_kwargs)
            for idx in range(num_blocks + 1):
                cin = cout + skips.pop()
                cout = model_channels * mult
                attn = (idx == num_blocks and res in attn_resolutions)
                self.dec[f'{res}x{res}x{res}_block{idx}'] = UNetBlock3D(in_channels=cin, out_channels=cout, attention=attn, **block_kwargs)
            if decoder_type == 'skip' or level == 0:
                if decoder_type == 'skip' and level < len(channel_mult) - 1:
                    self.dec[f'{res}x{res}x{res}_aux_up'] = Conv3d(in_channels=out_channels, out_channels=out_channels, kernel=0, up=True, resample_filter=resample_filter)
                self.dec[f'{res}x{res}x{res}_aux_norm'] = GroupNorm(num_channels=cout, eps=1e-6)
                self.dec[f'{res}x{res}x{res}_aux_conv'] = Conv3d(in_channels=cout, out_channels=out_channels, kernel=3, **init_zero)       

    def forward(self, x, noise_labels, class_labels=None, augment_labels=None):

        # Mapping.
        #pdb.set_trace()
        emb = self.map_noise(noise_labels) # time embedding 
        emb = emb.reshape(emb.shape[0], 2, -1).flip(1).reshape(*emb.shape) # swap sin/cos
        if self.map_label is not None:
            tmp = class_labels
            if self.training and self.label_dropout:
                tmp = tmp * (torch.rand([x.shape[0], 1], device=x.device) >= self.label_dropout).to(tmp.dtype)
            emb = emb + self.map_label(tmp * np.sqrt(self.map_label.in_features))
        if self.map_augment is not None and augment_labels is not None:
            emb = emb + self.map_augment(augment_labels)
        emb = silu(self.map_layer0(emb)) # 1 256 
        emb = silu(self.map_layer1(emb)) # 1 512   # t embedding                                                
        # Encoder.
        skips = []
        aux = x
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
                #print( 'input blocks before : ' , x.shape)
                x = block(x, emb) if isinstance(block, UNetBlock3D) else block(x)
                #print( 'input blocks after : ' ,  x.shape)
                skips.append(x)
        #pdb.set_trace()
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


class SongUNet(torch.nn.Module):
    def __init__(self,
        img_resolution,                     # Image resolution at input/output.
        in_channels,                        # Number of color channels at input.
        out_channels,                       # Number of color channels at output.
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
    ):
        assert embedding_type in ['fourier', 'positional']
        assert encoder_type in ['standard', 'skip', 'residual']
        assert decoder_type in ['standard', 'skip']

        super().__init__()
        self.label_dropout = label_dropout
        emb_channels = model_channels * channel_mult_emb
        noise_channels = model_channels * channel_mult_noise
        init = dict(init_mode='xavier_uniform')
        init_zero = dict(init_mode='xavier_uniform', init_weight=1e-5)
        init_attn = dict(init_mode='xavier_uniform', init_weight=np.sqrt(0.2))
        block_kwargs = dict(
            emb_channels=emb_channels, num_heads=1, dropout=dropout, skip_scale=np.sqrt(0.5), eps=1e-6,
            resample_filter=resample_filter, resample_proj=True, adaptive_scale=False,
            init=init, init_zero=init_zero, init_attn=init_attn,
        )

        # Mapping.
        self.map_noise = PositionalEmbedding(num_channels=noise_channels, endpoint=True) if embedding_type == 'positional' else FourierEmbedding(num_channels=noise_channels)
        self.map_label = Linear(in_features=label_dim, out_features=noise_channels, **init) if label_dim else None
        self.map_augment = Linear(in_features=augment_dim, out_features=noise_channels, bias=False, **init) if augment_dim else None
        self.map_layer0 = Linear(in_features=noise_channels, out_features=emb_channels, **init)
        self.map_layer1 = Linear(in_features=emb_channels, out_features=emb_channels, **init)

        # Encoder.
        self.enc = torch.nn.ModuleDict()
        cout = in_channels
        caux = in_channels
        for level, mult in enumerate(channel_mult):
            res = img_resolution >> level
            print(" level : " , level)
            print(' multi :' , mult)
            if level == 0:
                cin = cout
                cout = model_channels
                if implicit_mlp:
                    self.enc[f'{res}x{res}_conv'] = torch.nn.Sequential(
                                                        Conv2d(in_channels=cin, out_channels=cout, kernel=1, **init),
                                                        torch.nn.SiLU(),
                                                        Conv2d(in_channels=cout, out_channels=cout, kernel=1, **init),
                                                        torch.nn.SiLU(),
                                                        Conv2d(in_channels=cout, out_channels=cout, kernel=1, **init),
                                                        torch.nn.SiLU(),
                                                        Conv2d(in_channels=cout, out_channels=cout, kernel=3, **init),
                                                    )
                    self.enc[f'{res}x{res}_conv'].out_channels = cout
                else:
                    self.enc[f'{res}x{res}_conv'] = Conv2d(in_channels=cin, out_channels=cout, kernel=3, **init)
            else:
                self.enc[f'{res}x{res}_down'] = UNetBlock(in_channels=cout, out_channels=cout, down=True, **block_kwargs)
                if encoder_type == 'skip':
                    self.enc[f'{res}x{res}_aux_down'] = Conv2d(in_channels=caux, out_channels=caux, kernel=0, down=True, resample_filter=resample_filter)
                    self.enc[f'{res}x{res}_aux_skip'] = Conv2d(in_channels=caux, out_channels=cout, kernel=1, **init)
                if encoder_type == 'residual':
                    self.enc[f'{res}x{res}_aux_residual'] = Conv2d(in_channels=caux, out_channels=cout, kernel=3, down=True, resample_filter=resample_filter, fused_resample=True, **init)
                    caux = cout
            for idx in range(num_blocks):
                cin = cout
                cout = model_channels * mult
                attn = (res in attn_resolutions)
                self.enc[f'{res}x{res}_block{idx}'] = UNetBlock(in_channels=cin, out_channels=cout, attention=attn, **block_kwargs)
        skips = [block.out_channels for name, block in self.enc.items() if 'aux' not in name]

        # Decoder.
        self.dec = torch.nn.ModuleDict()
        for level, mult in reversed(list(enumerate(channel_mult))):
            res = img_resolution >> level
            if level == len(channel_mult) - 1:
                self.dec[f'{res}x{res}_in0'] = UNetBlock(in_channels=cout, out_channels=cout, attention=True, **block_kwargs)
                self.dec[f'{res}x{res}_in1'] = UNetBlock(in_channels=cout, out_channels=cout, **block_kwargs)
            else:
                self.dec[f'{res}x{res}_up'] = UNetBlock(in_channels=cout, out_channels=cout, up=True, **block_kwargs)
            for idx in range(num_blocks + 1):
                cin = cout + skips.pop()
                cout = model_channels * mult
                attn = (idx == num_blocks and res in attn_resolutions)
                self.dec[f'{res}x{res}_block{idx}'] = UNetBlock(in_channels=cin, out_channels=cout, attention=attn, **block_kwargs)
            if decoder_type == 'skip' or level == 0:
                if decoder_type == 'skip' and level < len(channel_mult) - 1:
                    self.dec[f'{res}x{res}_aux_up'] = Conv2d(in_channels=out_channels, out_channels=out_channels, kernel=0, up=True, resample_filter=resample_filter)
                self.dec[f'{res}x{res}_aux_norm'] = GroupNorm(num_channels=cout, eps=1e-6)
                self.dec[f'{res}x{res}_aux_conv'] = Conv2d(in_channels=cout, out_channels=out_channels, kernel=3, **init_zero)

    def forward(self, x, noise_labels, class_labels=None, augment_labels=None):
        # Mapping.
        #pdb.set_trace()
        emb = self.map_noise(noise_labels) # time embedding 
        emb = emb.reshape(emb.shape[0], 2, -1).flip(1).reshape(*emb.shape) # swap sin/cos
        if self.map_label is not None:
            tmp = class_labels
            if self.training and self.label_dropout:
                tmp = tmp * (torch.rand([x.shape[0], 1], device=x.device) >= self.label_dropout).to(tmp.dtype)
            emb = emb + self.map_label(tmp * np.sqrt(self.map_label.in_features))
        if self.map_augment is not None and augment_labels is not None:
            emb = emb + self.map_augment(augment_labels)
        emb = silu(self.map_layer0(emb)) # 1 256 
        emb = silu(self.map_layer1(emb)) # 1 512   # t embedding 
                                                       
        # Encoder.
        skips = []
        aux = x
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
                x = block(x, emb) if isinstance(block, UNetBlock) else block(x)
                skips.append(x)

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
        #pdb.set_trace()
        return aux
    

def test_demo_unet2d():
    img_resolution = 256    # 图像分辨率
    in_channels = 4       # 输入通道数
    out_channels = 1      # 输出通道数
    label_dim = 0      # 类别标签维度


      # 创建模型  # implict_mlp is false 
    model = SongUNet(
        img_resolution = img_resolution,
        in_channels = in_channels,
        out_channels = out_channels,
        label_dim = label_dim,
        model_channels = 128,          # 基础通道数
        channel_mult = [1,2,2,2],      # 通道倍增
        channel_mult_emb = 4,          # 嵌入维度倍增
        num_blocks = 2,                # 每个分辨率的残差块数
        attn_resolutions = [16],       # 使用自注意力的分辨率
        dropout = 0.1,                 # dropout率
        label_dropout = 0.1  ,          # 标签dropout率
        implicit_mlp=False
    )  
    model = model.to('cuda')
    batch_size = 1
    x = torch.randn(batch_size, in_channels, img_resolution, img_resolution).to('cuda')  # 输入图像
    noise_labels = torch.randn(batch_size).to('cuda')  # 噪声标签
    #class_labels = torch.randn(batch_size, label_dim).to('cuda')  # 类别标签
    class_labels = None
    output = model(x, noise_labels,class_labels)

def test_demo_unet3d():
    vol_resolution = 32    # 3D体积分辨率
    in_channels = 4       # 输入通道数
    out_channels = 1      # 输出通道数
    label_dim = 0        # 类别标签维度

    # 创建3D UNet模型
    model = SongUNet3D(
        img_resolution = vol_resolution,
        in_channels = in_channels,
        out_channels = out_channels,
        label_dim = label_dim,
        model_channels = 128,          # 保持与2D版本相同的通道配置
        channel_mult = [1,2,2],      
        channel_mult_emb = 4,          
        num_blocks = 2,                
        attn_resolutions = [16],       
        dropout = 0.1,                 
        label_dropout = 0.1           
    )
    
    model = model.to('cuda')
    
    # 创建3D输入数据
    batch_size = 1
    x = torch.randn(batch_size, in_channels, vol_resolution, vol_resolution, vol_resolution).to('cuda')
    noise_labels = torch.randn(batch_size).to('cuda')
    class_labels = None
    
    output = model(x, noise_labels, class_labels)        

# main 
if __name__ == '__main__':
   #test_demo_unet2d()
   test_demo_unet3d()