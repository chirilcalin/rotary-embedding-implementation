import einops
from fancy_einsum import einsum
from dataclasses import dataclass
from easy_transformer import EasyTransformer
import torch
import torch.nn as nn
import numpy as np
import math
from easy_transformer.utils import get_corner, gelu_new, tokenize_and_concatenate
import tqdm.auto as tqdm


#this script is the main transformer big thing

reference_gpt2 = EasyTransformer.from_pretrained("gpt2-small", fold_ln=False, center_unembed=False, center_writing_weights=False)

@dataclass
class Config:
    d_model: int = 768
    debug: bool = True
    layer_norm_eps: float = 1e-5
    d_vocab: int = 50257
    init_range: float = 0.02
    n_ctx: int = 1024
    d_head: int = 64
    d_mlp: int = 3072
    n_heads: int = 12
    n_layers: int = 12

cfg = Config()


batch_size = 8
num_epochs = 1
max_steps = 1000
log_every = 10
lr = 1e-3
weight_decay = 1e-2
model_cfg = Config(debug=False, d_model=256, n_heads=4, d_head=64, d_mlp=1024, n_layers=2, n_ctx=256, d_vocab=reference_gpt2.cfg.d_vocab)


class RotaryOperation():

    def precompute_theta_position_frequencies(head_dim : int, seq_len : int , device : str, theta : float = 10000.0):
    
        assert head_dim % 2 == 0
    
        theta_numerator = torch.arange(0, head_dim, 2).float() 
        theta = 1.0 / (theta ** (theta_numerator / head_dim )).to(device) 
        m = torch.arange(seq_len, device=device)
    
        freqs = torch.outer(m, theta).float()     
    
        freqs_complex = torch.polar(torch.ones_like(freqs), freqs)
        return freqs_complex


    def apply_rotary_embeddings(x: torch.Tensor, freqs_complex: torch.Tensor, device: str):

        x_complex = torch.view_as_complex(x.float().reshape(*x.shape[:-1],-1, 2)) 
    
        freqs_complex = freqs_complex.unsqueeze(0).unsqueeze(2) 
    
        x_rotated = x_complex * freqs_complex
    
        x_out = torch.view_as_real(x_rotated)
        x_out = x_out.reshape(*x.shape)
        return x_out.type_as(x).to(device)




class LayerNorm(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.w = nn.Parameter(torch.ones(cfg.d_model))
        self.b = nn.Parameter(torch.zeros(cfg.d_model))

    def forward(self, residual):
        # residual: [batch, position, d_model]

        y = residual 
        y1 = y - torch.mean(y, dim=[2], keepdim=True) 
        y2 = y1 / torch.std(y1,  dim=[2], keepdim=True, correction=0)
        y3 = einops.einsum(y2, self.w, 'b t c, c -> b t c') 
        y4 = y3 + self.b 
        

        return y4


class Embed(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.W_E = nn.Parameter(torch.empty((cfg.d_vocab, cfg.d_model)))
        nn.init.normal_(self.W_E, std=self.cfg.init_range)

    def forward(self, tokens):
        # tokens: [batch, position]
        


        boon = []
        for z in tokens:
            zoon = [(self.W_E[x]) for x in z]
            zoon = torch.stack(zoon)
            boon.append(zoon)

        boon = torch.stack(boon)


        return boon


class RopeAttention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.W_Q = nn.Parameter(torch.empty((cfg.n_heads, cfg.d_model, cfg.d_head)))
        nn.init.normal_(self.W_Q, std=self.cfg.init_range)
        self.b_Q = nn.Parameter(torch.zeros((cfg.n_heads, cfg.d_head)))
        self.W_K = nn.Parameter(torch.empty((cfg.n_heads, cfg.d_model, cfg.d_head)))
        nn.init.normal_(self.W_K, std=self.cfg.init_range)
        self.b_K = nn.Parameter(torch.zeros((cfg.n_heads, cfg.d_head)))
        self.W_V = nn.Parameter(torch.empty((cfg.n_heads, cfg.d_model, cfg.d_head)))
        nn.init.normal_(self.W_V, std=self.cfg.init_range)
        self.b_V = nn.Parameter(torch.zeros((cfg.n_heads, cfg.d_head)))

        self.W_O = nn.Parameter(torch.empty((cfg.n_heads, cfg.d_head, cfg.d_model)))
        nn.init.normal_(self.W_O, std=self.cfg.init_range)
        self.b_O = nn.Parameter(torch.zeros((cfg.d_model)))

        self.register_buffer("IGNORE", torch.tensor(-1e5, dtype=torch.float32, device="cuda"))

    def forward(self, normalized_resid_pre):
        # normalized_resid_pre: [batch, position, d_model]

        freqs_complex = precompute_theta_position_frequencies(self.cfg.d_head, normalized_resid_pre.shape[1], "cuda", 10000)

        
        Q=einops.einsum(normalized_resid_pre, self.W_Q, 'b t c, n_heads c d_head -> b t n_heads d_head ') + self.b_Q
        K=einops.einsum(normalized_resid_pre, self.W_K, 'b t c, n_heads c d_head -> b t n_heads d_head ') + self.b_K
        V=einops.einsum(normalized_resid_pre, self.W_V, 'b t c, n_heads c d_head -> b t n_heads d_head ') + self.b_V

        Q=apply_rotary_embeddings(Q, freqs_complex, "cuda")
        K=apply_rotary_embeddings(K, freqs_complex, "cuda")
        
        zing = einops.einsum(Q, K, 'batch q n_head d_head, batch k n_head d_head -> batch n_head q k')  
        zing = zing / math.sqrt(cfg.d_head)
        masked = self.apply_causal_mask(zing)
        soft = nn.Softmax(dim=3)
        softer=soft(masked)
        maskedV = einops.einsum(softer, V, 'batch n_head qindex kindex, batch kindex n_head d_head -> batch qindex n_head d_head') 
        zolder =  einops.einsum(maskedV, self.W_O, 'batch qindex n_head d_head, n_head d_head d_model -> batch qindex d_model') + self.b_O 
        return zolder
        

    def apply_causal_mask(self, attn_scores):
        # attn_scores: [batch, n_heads, query_pos, key_pos]
        updated = torch.tril(attn_scores)
        updated[updated==0] = float('-inf')
        return updated



class MLP(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.W_in = nn.Parameter(torch.empty((cfg.d_model, cfg.d_mlp)))
        nn.init.normal_(self.W_in, std=self.cfg.init_range)
        self.b_in = nn.Parameter(torch.zeros((cfg.d_mlp)))
        self.W_out = nn.Parameter(torch.empty((cfg.d_mlp, cfg.d_model)))
        nn.init.normal_(self.W_out, std=self.cfg.init_range)
        self.b_out = nn.Parameter(torch.zeros((cfg.d_model)))

    def forward(self, normalized_resid_mid):
        # normalized_resid_mid: [batch, position, d_model]


        one1=normalized_resid_mid@self.W_in + self.b_in
        one=einops.einsum(normalized_resid_mid, self.W_in, 'b t c, c d_mlp -> b t d_mlp') + self.b_in
        gelu = nn.GELU(approximate='tanh')
        two=gelu(one)
        three = einops.einsum(two, self.W_out, 'b t d_mlp, d_mlp c -> b t c') + self.b_out
        

        return three


class RopeBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        self.ln1 = LayerNorm(cfg)
        self.attn = RopeAttention(cfg)
        self.ln2 = LayerNorm(cfg)
        self.mlp = MLP(cfg)

    def forward(self, resid_pre):
        # resid_pre [batch, position, d_model]

        norm_resid_pre = self.ln1(resid_pre)
        attended = self.attn(norm_resid_pre)
        resid_pre = resid_pre + attended
        norm_resid_mid = self.ln2(resid_pre)
        mlped = self.mlp(norm_resid_mid)
        resid_pre = resid_pre + mlped
        
        
        return resid_pre



class Unembed(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.W_U = nn.Parameter(torch.empty((cfg.d_model, cfg.d_vocab)))
        nn.init.normal_(self.W_U, std=self.cfg.init_range)
        self.b_U = nn.Parameter(torch.zeros((cfg.d_vocab), requires_grad=False))

    def forward(self, normalized_resid_final):
        # normalized_resid_final [batch, position, d_model]

        return normalized_resid_final@self.W_U + self.b_U




class RopeTransformer(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.embed = Embed(cfg)
        #self.pos_embed = PosEmbed(cfg)
        self.blocks = nn.ModuleList([RopeBlock(cfg) for _ in range(cfg.n_layers)])
        self.ln_final = LayerNorm(cfg)
        self.unembed = Unembed(cfg)

    def forward(self, tokens):
        # tokens [batch, position]


        embedded = self.embed(tokens)
        expansion = self.blocks[0](embedded)
        counter = 1
        while counter < len(self.blocks):
            expansion = self.blocks[counter](expansion)
            counter+=1
            if counter == cfg.n_layers:
                break
        

        normalized_resid_final = self.ln_final(expansion)
        logits = self.unembed(normalized_resid_final)


        
        return logits
