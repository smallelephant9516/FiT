import math

import torch
from torch import nn
import torchvision.transforms as T


from einops import rearrange, repeat, reduce
from einops.layers.torch import Rearrange

from Data_loader.image_transformation import crop

# helpers

def pair(t):
    return t if isinstance(t, tuple) else (t, t)


# classes

# posemb_sincos_1d copy from pytoch-vit
def posemb_sincos_1d(patches, temperature = 10000, dtype = torch.float32):
    _, n, dim, device, dtype = *patches.shape, patches.device, patches.dtype

    n = torch.arange(n, device = device)
    assert (dim % 2) == 0, 'feature dimension must be multiple of 2 for sincos emb'
    omega = torch.arange(dim // 2, device = device) / (dim // 2 - 1)
    omega = 1. / (temperature ** omega)

    n = n.flatten()[:, None] * omega[None, :]
    pe = torch.cat((n.sin(), n.cos()), dim = 1)
    return pe.type(dtype)


class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, batch_size, dim, max_len=2048, device='cuda:0'):
        super(SinusoidalPositionalEncoding, self).__init__()
        self.dim = dim
        self.max_len = max_len

        self.freq = nn.Parameter(torch.rand(batch_size, 1), requires_grad=True).to(device)
        self.shift = nn.Parameter(torch.rand(batch_size, 1), requires_grad=True).to(device)

        position = torch.arange(0, max_len, dtype=torch.float).to(device)
        position = repeat(position, 'n -> b n', b=batch_size)
        #print(position.shape)
        self.pe = torch.sin(position * self.freq + self.shift).to(device)
        #print(self.pe .shape)
        self.pe = repeat(self.pe, 'm n -> m n d', d=dim)
        #self.register_buffer('pe', pe, persistent=False)

    def forward(self, x):
        b = len(x)
        x = x + self.pe[:b, :x.size(1), :]
        return x

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim, heads=8, dropout=0.):
        super().__init__()
        self.multihead_attn = nn.MultiheadAttention(embed_dim=dim, num_heads=heads, dropout=dropout)

    def forward(self, x, mask=None):
        # nn.MultiheadAttention expects inputs in the format (Seq_len, Batch, Embedding)
        x = x.transpose(0, 1)
        # can add a linear transform to make everything in the same dimension
        attn_output, _ = self.multihead_attn(x, x, x, attn_mask=mask)
        # Transpose the output back to (Batch, Seq_len, Embedding)
        return attn_output.transpose(0, 1)

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, mlp_dim=2048, dropout=0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        self.att_layer=Attention(dim, heads=heads, dropout=dropout)
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, self.att_layer),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout))
            ]))

    def forward(self, x, mask=None):
        for attn, ff in self.layers:
            x = attn(x, mask=mask) + x
            x = ff(x) + x
        return x





class ViT(nn.Module):
    def __init__(self, *, image_height, image_width, image_patch_size, dim, depth, heads, mlp_dim, pool='cls',
                 dim_head=64, dropout=0., emb_dropout=0.):
        super().__init__()
        patch_height, patch_width = pair(image_patch_size)

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = patch_height * patch_width
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b (h p1) (w p2) -> b (h w) (p1 p2)', p1=patch_height, p2=patch_width),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim),
            nn.LayerNorm(dim),
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))

        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))

        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        self.pool = pool
        self.to_latent = nn.Identity()


    def forward(self, img):
        x = self.to_patch_embedding(img)
        b, n, _ = x.shape

        cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b=b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)

        hid_dim = self.transformer(x)

        out = hid_dim.mean(dim=1) if self.pool == 'mean' else x[:, 0]

        out = self.to_latent(out)
        return hid_dim


class ViT_3D(nn.Module):
    def __init__(self, *, image_height, image_width, image_patch_size, length, length_patch_size, dim,
                 depth, heads, batch_size, mlp_dim, pool='cls', dim_head=64, dropout=0., emb_dropout=0.,):
        super().__init__()

        self.heads=heads
        patch_height, patch_width = pair(image_patch_size)


        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'
        assert length % length_patch_size == 0, 'Frames must be divisible by frame patch size'

        num_patches = (image_height // patch_height) * (image_width // patch_width) * (length // length_patch_size)
        patch_dim = patch_height * patch_width * length_patch_size

        self.mini_patch_size = (image_height // patch_height) * (image_width // patch_width)
        self.heads = heads
        self.num_patches = num_patches
        self.length_patch_size = length_patch_size
        self.patch_height = patch_height
        self.patch_width = patch_width
        self.length = length
        self.height = image_height
        self.width = image_width
        self.num_image_patches = (image_height // patch_height) * (image_width // patch_width)

        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        self.to_patch_embedding = nn.Sequential(
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim),
            nn.LayerNorm(dim),
        )

        self.to_patch_embedding_conv = nn.Conv2d(1, dim, kernel_size=image_patch_size, stride=image_patch_size)
        self.LN_embedding = nn.LayerNorm(dim)

        phase=nn.Parameter(torch.randn((1)))
        phase_dim = phase.repeat((1,dim))

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.pos_embedding_sincos = posemb_sincos_1d
        learnable_shift_freq = SinusoidalPositionalEncoding(batch_size, dim, num_patches+1)
        learnable_shift_freq.to('cuda:0')
        self.pos_embedding_fre_shift = learnable_shift_freq
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, mlp_dim, dropout)

        self.pool = pool
        self.to_latent = nn.Identity()

    def forward(self, filaments, mask=None, reshape_filament=True):
        b, n = filaments.shape[:2]
        if reshape_filament is True:
            filaments_batch = rearrange(filaments,'b l h w -> (b l) h w')
            filaments_batch = crop(filaments_batch, self.height, self.width)
            filaments_batch = T.Normalize(mean=[0], std=[1])(filaments_batch)
            filaments = rearrange(filaments_batch, '(b l) h w -> b l h w', b=b)
            filaments = rearrange(filaments, 'b (l pl) (h p1) (w p2) -> b (l h w) (pl p1 p2)', p1=self.patch_height, p2=self.patch_width,
                      pl=self.length_patch_size)
        x = self.to_patch_embedding(filaments)
        b, n, _ = x.shape


        if mask is None:
            mask = torch.ones((b,self.heads,self.num_patches+1,self.num_patches+1))
        else:
            mask = self.matrix_mask(mask)

        mask = mask == 0
        print(mask,mask.shape)
        cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b=b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(n + 1)]
        #x = self.pos_embedding_fre_shift(x)
        x = self.dropout(x)

        hid_dim = self.transformer(x, mask)


        out = hid_dim.mean(dim=1) if self.pool == 'mean' else x[:, 0]

        out = self.to_latent(out)
        # out = self.mlp_head(out)
        return hid_dim

    def matrix_mask(self, mask):
        b, n = mask.shape
        cls = torch.ones(b, 1).to(mask.get_device())

        mask = mask.repeat_interleave(repeats=self.mini_patch_size,dim=1)
        mask = torch.cat((cls, mask), 1)
        self.mask_cls = mask

        n_new = mask.shape[-1]
        mask_matrix = torch.bmm(mask.view(b, n_new, 1), mask.view(b, 1, n_new))
        mask_matrix[:, :, 0] = 1
        mask_matrix = repeat(mask_matrix, 'b n1 n2-> b h n1 n2', h=self.heads)
        mask_matrix = rearrange(mask_matrix, 'b h n1 n2 -> (b h) n1 n2')
        return mask_matrix


# vit for vectors
class ViT_vector(nn.Module):
    def __init__(self, *, length, patch_dim, dim,
                 depth, heads, batch_size, mlp_dim, pool='cls', dim_head=64, dropout=0., emb_dropout=0.):
        super().__init__()

        self.heads=heads

        num_patches = length
        patch_dim = patch_dim

        self.mask=torch.ones((batch_size,heads,num_patches+1,num_patches+1))
        self.length = length

        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        self.to_patch_embedding = nn.Sequential(
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim),
            nn.LayerNorm(dim),
        )
        phase=nn.Parameter(torch.randn((1)))
        phase_dim = phase.repeat((1,dim))

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.pos_embedding_sincos = posemb_sincos_1d
        #learnable_shift_freq = SinusoidalPositionalEncoding(batch_size, dim, num_patches+1)
        #learnable_shift_freq.to('cuda:2')
        #self.pos_embedding_fre_shift = learnable_shift_freq
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        self.pool = pool
        self.to_latent = nn.Identity()


    def forward(self, filaments):
        x = self.to_patch_embedding(filaments)
        b, n, _ = x.shape

        cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b=b)
        x = torch.cat((cls_tokens, x), dim=1)
        #x += self.pos_embedding[:, :(n + 1)]
        x += self.pos_embedding_sincos(x)
        #x = self.pos_embedding_fre_shift(x)
        x = self.dropout(x)

        hid_dim = self.transformer(x, self.mask)

        #print(self.mask[0,0,-100:,0])

        out = hid_dim.mean(dim=1) if self.pool == 'mean' else x[:, 0]

        out = self.to_latent(out)
        # out = self.mlp_head(out)
        return hid_dim

    def matrix_mask(self,mask):
        b,n=mask.shape
        cls = torch.ones(b, 1).to(mask.get_device())

        mask = torch.cat((cls, mask), 1)
        self.mask_cls=mask

        n_new=mask.shape[-1]
        mask_matrix = torch.bmm(mask.view(b, n_new, 1), mask.view(b, 1, n_new))
        mask_matrix[:, :, 0] = 1
        mask_matrix = repeat(mask_matrix, 'b n1 n2-> b h n1 n2', h=self.heads)
        self.mask=mask_matrix

        return mask_matrix