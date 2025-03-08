import einops
from fancy_einsum import einsum
from dataclasses import dataclass
from easy_transformer import EasyTransformer
import torch
import torch.nn as nn
import numpy as np
import math
from easy_transformer.utils import tokenize_and_concatenate
import tqdm.auto as tqdm
import time


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

        allBatches = []
        for batch in tokens:
            oneBatch = [(self.W_E[t]) for t in batch]
            oneBatch = torch.stack(oneBatch)
            allBatches.append(oneBatch)

        allBatches = torch.stack(allBatches)

        return allBatches


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

        self.f_C = None



        self.register_buffer("IGNORE", torch.tensor(-1e5, dtype=torch.float32, device="cuda"))


    def precompute_theta_position_frequencies(head_dim : int, seq_len : int , device : str, theta : float = 10000.0):
    
        #Head_dimension must be even, since we split the head into pairs of real/imaginary numbers
        assert head_dim % 2 == 0
    
        theta_numerator = torch.arange(0, head_dim, 2).float() #Creates array [0,2,4...d/2], i.e. shape still (head_dim / 2)
        theta = 1.0 / (theta ** (theta_numerator / head_dim )).to(device) # Still (head_dim/2)
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
    

    def forward(self, normalized_resid_pre):
        # normalized_resid_pre: [batch, position, d_model]

        
        Q=einops.einsum(normalized_resid_pre, self.W_Q, 'b t c, n_heads c d_head -> b t n_heads d_head ') + self.b_Q
        K=einops.einsum(normalized_resid_pre, self.W_K, 'b t c, n_heads c d_head -> b t n_heads d_head ') + self.b_K
        V=einops.einsum(normalized_resid_pre, self.W_V, 'b t c, n_heads c d_head -> b t n_heads d_head ') + self.b_V

        Q=apply_rotary_embeddings(Q, self.f_C, "cuda")
        K=apply_rotary_embeddings(K, self.f_C, "cuda")
        
        qk_circuit = einops.einsum(Q, K, 'batch q n_head d_head, batch k n_head d_head -> batch n_head q k')

        qk_circuit = qk_circuit / math.sqrt(cfg.d_head)
        masked = self.apply_causal_mask(qk_circuit)
        softmax = nn.Softmax(dim=3)
        soft = softmax(masked)
        maskedValues = einops.einsum(soft, V, 'batch n_head qindex kindex, batch kindex n_head d_head -> batch qindex n_head d_head') 
        outmatrix =  einops.einsum(maskedValues, self.W_O, 'batch qindex n_head d_head, n_head d_head d_model -> batch qindex d_model') + self.b_O 
        return outmatrix
        

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

        combineScale = einops.einsum(normalized_resid_mid, self.W_in, 'b t c, c d_mlp -> b t d_mlp') + self.b_in
        gelu = nn.GELU(approximate='tanh')
        geluOutput = gelu(combineScale)
        combineScaleLoss = einops.einsum(geluOutput, self.W_out, 'b t d_mlp, d_mlp c -> b t c') + self.b_out
        

        return combineScaleLoss


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
        self.attn.f_C = precompute_theta_position_frequencies(self.cfg.d_head, resid_pre.shape[1], "cuda", 10000)
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
