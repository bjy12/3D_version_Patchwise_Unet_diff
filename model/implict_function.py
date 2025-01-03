import torch
import torch.nn as nn
import pdb

# Positional encoding (section 5.1)
class Embedder:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.create_embedding_fn()

    def create_embedding_fn(self):
        embed_fns = []
        d = self.kwargs["input_dims"]
        out_dim = 0
        if self.kwargs["include_input"]:
            embed_fns.append(lambda x: x)
            out_dim += d

        max_freq = self.kwargs["max_freq_log2"]
        N_freqs = self.kwargs["num_freqs"]

        if self.kwargs["log_sampling"]:
            freq_bands = 2.0 ** torch.linspace(0.0, max_freq, steps=N_freqs)
        else:
            freq_bands = torch.linspace(2.0 ** 0.0, 2.0 ** max_freq, steps=N_freqs)

        for freq in freq_bands:
            for p_fn in self.kwargs["periodic_fns"]:
                embed_fns.append(lambda x, p_fn=p_fn, freq=freq: p_fn(x * freq))
                out_dim += d

        self.embed_fns = embed_fns
        self.out_dim = out_dim  ## self.kwargs["input_dims"] + self.kwargs["input_dims"] * 2 * self.kwargs["num_freqs"]

    def embed(self, inputs):
        return torch.cat([fn(inputs) for fn in self.embed_fns], -1)

def get_embedder(multires, i=0):
    if i == -1:
        return nn.Identity(), 3

    embed_kwargs = {
        "include_input": True,
        "input_dims": 3,
        "max_freq_log2": multires - 1,
        "num_freqs": multires,
        "log_sampling": True,
        "periodic_fns": [torch.sin, torch.cos],
    }

    embedder_obj = Embedder(**embed_kwargs)
    embed = lambda x, eo=embedder_obj: eo.embed(x)
    return embed, embedder_obj.out_dim


# Model
class DummyNeRF(nn.Module):
    def __init__(self, input_ch , output_ch):
        """ """
        super(DummyNeRF, self).__init__()
        self.input_ch = input_ch
        self.output_ch = output_ch
        print(f"[DummyNeRF] Input ch : {self.input_ch}, Output ch : {self.output_ch}")

        ## for dimension reduction
        if self.input_ch != self.output_ch:
            self.linear = nn.Linear(self.input_ch, self.output_ch)

    def forward(self, x , l_features):
        res_features = l_features 
        # pdb.set_trace()

        if self.input_ch != self.output_ch:
            func_implict = self.linear(torch.cat([x , l_features] , dim=-1 ))
        output = torch.cat((func_implict , res_features) , dim=-1 )
        return output