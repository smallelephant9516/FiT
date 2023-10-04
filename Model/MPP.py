# helpers
import math

import torch
import torch.nn.functional as F
import torchvision.transforms as T
from torch import nn

from einops import rearrange, repeat, reduce
from einops.layers.torch import Rearrange

from Data_loader.ctf_fuction import ctf_correction_torch
from Data_loader.image_transformation import crop


def exists(val):
    return val is not None


def prob_mask_like(t, prob):
    batch, seq_length, _ = t.shape
    return torch.zeros((batch, seq_length)).float().uniform_(0, 1) < prob


def get_mask_subset_with_prob(patched_input, prob):
    batch, seq_len, _, device = *patched_input.shape, patched_input.device
    max_masked = math.ceil(prob * seq_len)

    rand = torch.rand((batch, seq_len), device=device)
    _, sampled_indices = rand.topk(max_masked, dim=-1)

    new_mask = torch.zeros((batch, seq_len), device=device)
    new_mask.scatter_(1, sampled_indices, 1)
    return new_mask.bool()

# device has some problem
def create_random_patches(input, mask):
    device = input.device
    b, n, _ = input.shape
    _, n_mask = mask.shape
    multi = n // n_mask
    rand_patch_id=torch.zeros((b,n),dtype=torch.int64,device=device)
    for i in range(b):
        lst=mask[i]
        pos = torch.nonzero(lst == 1).flatten().to(device)
        max_id=len(pos)
        pos_rep = pos.repeat_interleave(multi) * multi + torch.tensor(list(torch.arange(multi)) * len(pos),device=device)
        # can be modified later for padding mask in the middle
        rand_patch_id[i,:]=pos_rep[torch.randint(0,int(max_id*multi),(n,))]
        #rand_patch_id[i, :] = torch.randint(0, int(max_id * multi), (n,))
    return rand_patch_id

def image_augmentation(images,h,w,rot,h_shift,w_shift):
    # images b X D X D, rot: degree, h_shift,w_shift: percentage
    combined = T.Compose([
        T.RandomAffine(degrees=(-rot, rot), translate=(h_shift, w_shift), scale=(1, 1)),
        T.CenterCrop(size=(h, w)),
        T.RandomHorizontalFlip(),
        T.RandomVerticalFlip(),
        #T.Normalize(mean=[0], std=[1])
    ])
    return combined(images)

def image_augmentation_filament(images, h, w, rot=10, h_shift=0.01, w_shift= 0.1 ,augment=True):
    # images b X l X D X D, rot: degree, h_shift,w_shift: percentage
    b, l, dim, _ = images.shape
    images_batch = rearrange(images,'b l h w -> (b l) h w')
    if augment is True:
        images_batch = image_augmentation(images_batch,h,w,rot,h_shift,w_shift)
    else:
        images_batch = crop(images_batch, h, w)
        images_batch = T.Normalize(mean=[0], std=[1])(images_batch)
    images = rearrange(images_batch,'(b l) h w -> b l h w',b=b)
    return images


class Reshape_sequence_image(nn.Module):
    def __init__(
            self,
            patch_size_h,
            patch_size_w,
            length):
        self.length = length

    def reshape_mean(self, input):
        patch_size_h = self.patch_size_h
        patch_size_w = self.patch_size_w
        input_mean = reduce(input, 'b l (h p1) (w p2) -> b (h w) l', 'mean', p1=patch_size_h, p2=patch_size_w)
        return input_mean

    def reshape_bnd(self, input):
        patch_size_h = self.patch_size_h
        patch_size_w = self.patch_size_w
        input_bnd = rearrange(input, 'b l (h p1) (w p2) -> b (h w) (p1 p2 l)', p1=patch_size_h, p2=patch_size_w)
        return input_bnd


# mpp loss

class MPPLoss(nn.Module):
    def __init__(
            self,
            patch_size,
            channels,
            output_channel_bits,
            max_pixel_val,
            mean,
            std,
            lossF = 'l2_norm'
    ):
        super().__init__()
        self.patch_size = patch_size
        self.channels = channels
        self.output_channel_bits = output_channel_bits
        self.max_pixel_val = max_pixel_val

        self.mean = torch.tensor(mean).view(-1, 1, 1) if mean else None
        self.std = torch.tensor(std).view(-1, 1, 1) if std else None
        if lossF == 'cross_entropy':
            self.loss = F.cross_entropy
        elif lossF == 'l2_norm':
            self.loss = F.mse_loss
        elif lossF == 'l1_norm':
            self.loss = F.l1_loss
        else:
            print('no such loss function: {}'.format(lossF))

    def forward(self, predicted_patches, target, mask):
        p, c, mpv, bits, device = self.patch_size, self.channels, self.max_pixel_val, self.output_channel_bits, target.device

        # reshape target to patches
        # target = target.clamp(max = mpv) # clamp just in case
        #loss = self.loss(predicted_patches[mask], target[mask])
        loss = self.loss(predicted_patches, target)
        return loss


# main class
class MPP(nn.Module):
    def __init__(
            self,
            transformer,
            image_height,
            image_width,
            patch_size,
            dim,
            output_channel_bits=1,
            channels=1,
            max_pixel_val=1.0,
            mask_prob=0.15,
            replace_prob=0.5,
            random_patch_prob=0.5,
            inter_seg_distance = 0.01,
            mean=None,
            std=None,
            lossF = 'l2_norm'
    ):
        super().__init__()
        self.transformer = transformer
        self.loss = MPPLoss(patch_size, channels, output_channel_bits,
                            max_pixel_val, mean, std , lossF = lossF)

        # output transformation
        self.to_bits = nn.Linear(dim, patch_size ** 2)

        # vit related dimensions
        self.patch_size = patch_size
        self.image_height = image_height
        self.image_width = image_width
        self.inter_seg_distance = inter_seg_distance

        # check the dimension of the height, width and length
        assert image_height % patch_size == 0
        assert image_width % patch_size == 0

        # mpp related probabilities
        self.mask_prob = mask_prob
        self.replace_prob = replace_prob
        self.random_patch_prob = random_patch_prob

        # token ids
        self.mask_token = nn.Parameter(torch.randn(1, 1, patch_size ** 2))

    def forward(self, input, ctf=None, **kwargs):
        transformer = self.transformer
        # clone original image for loss
        img = input.clone().detach()
        img = crop(img, self.image_height, self.image_width)
        img = T.Normalize(mean=[0], std=[1])(img)
        #img = image_augmentation(img, self.image_height, self.image_width, 5, 0.01, 0.1)
        b,height,width = img.shape

        # add augmentation
        #input = crop(input, self.image_height, self.image_width)
        input = image_augmentation(input, self.image_height, self.image_width, 10, 0.01, w_shift=self.inter_seg_distance)
        # reshape raw image to patches
        p = self.patch_size
        input = rearrange(input,
                          'b (h p1) (w p2) -> b (h w) (p1 p2)',
                          p1=p,
                          p2=p)

        mask = get_mask_subset_with_prob(input, self.mask_prob)

        # mask input with mask patches with probability of `replace_prob` (keep patches the same with probability 1 - replace_prob)
        masked_input = input.clone().detach()

        # if random token probability > 0 for mpp
        if self.random_patch_prob > 0:
            random_patch_sampling_prob = self.random_patch_prob / (
                    1 - self.replace_prob)
            random_patch_prob = prob_mask_like(input,
                                               random_patch_sampling_prob).to(mask.device)

            bool_random_patch_prob = mask * (random_patch_prob == True)
            random_patches = torch.randint(0,
                                           input.shape[1],
                                           (input.shape[0], input.shape[1]),
                                           device=input.device)
            randomized_input = masked_input[
                torch.arange(masked_input.shape[0]).unsqueeze(-1),
                random_patches]
            masked_input[bool_random_patch_prob] = randomized_input[
                bool_random_patch_prob]

        # [mask] input
        replace_prob = prob_mask_like(input, self.replace_prob).to(mask.device)
        bool_mask_replace = (mask * replace_prob) == True
        masked_input[bool_mask_replace] = self.mask_token

        # linear embedding of patches
        masked_input = transformer.to_patch_embedding[1:4](masked_input)

        # add cls token to input sequence
        b, n, _ = masked_input.shape
        cls_tokens = repeat(transformer.cls_token, '() n d -> b n d', b=b)
        masked_input = torch.cat((cls_tokens, masked_input), dim=1)

        # add positional embeddings to input
        masked_input += transformer.pos_embedding[:, :(n + 1)]
        #masked_input += transformer.pos_embedding_sincos
        masked_input = transformer.dropout(masked_input)

        # get generator output and get mpp loss
        masked_input = transformer.transformer(masked_input, **kwargs)

        cls_logits = self.to_bits(masked_input)
        logits = cls_logits[:, 1:, :]

        img = rearrange(img, 'b (h p1) (w p2) -> b (h w) (p1 p2) ', p1=p, p2=p).contiguous()

        logits = rearrange(logits, 'b (h w) (p1 p2) -> b (h p1) (w p2)', p1=p, h=height//p)

        logits = rearrange(logits, 'b (h p1) (w p2) -> b (h w) (p1 p2)', p1=p, p2=p)

        mpp_loss = self.loss(logits, img, mask)

        return mpp_loss


class MPP_3D(nn.Module):
    def __init__(
            self,
            transformer,
            image_height,
            image_width,
            patch_size,
            length_patch_size,
            dim,
            output_channel_bits=1,
            channels=1,
            max_pixel_val=1.0,
            mask_prob=0.15,
            replace_prob=0.5,
            random_patch_prob=0.5,
            augment_prob=0.5,
            inter_seg_distance=0.01,
            mean=None,
            std=None,
            lossF = 'l2_norm',
            patch_emb_type = 'linear' # conv or linear
    ):
        super().__init__()
        self.transformer = transformer
        self.loss = MPPLoss(patch_size, channels, output_channel_bits,
                            max_pixel_val, mean, std, lossF = lossF)


        # output transformation
        self.to_bits = nn.Sequential(nn.LayerNorm(dim),
                                     nn.Linear(dim, length_patch_size * (patch_size ** 2)))

        self.normLayer = nn.LayerNorm(length_patch_size * (patch_size ** 2))
        # vit related dimensions
        self.patch_size = patch_size
        self.image_height = image_height
        self.image_width = image_width
        self.length_patch_size = length_patch_size
        self.patch_emb_type = patch_emb_type

        # mpp related probabilities
        self.mask_prob = mask_prob
        self.replace_prob = replace_prob
        self.random_patch_prob = random_patch_prob
        self.augment_prob=augment_prob
        self.inter_seg_distance = inter_seg_distance

        # token ids
        self.mask_token = nn.Parameter(torch.randn(1, 1, length_patch_size * (patch_size ** 2)))

    def forward(self, input, padding_mask, ctf=None, **kwargs):
        transformer = self.transformer
        # clone original image for loss
        img = input.clone().detach()
        img = image_augmentation_filament(img, self.image_height, self.image_width, augment=False)
        b,length,height,width = img.shape

        # add augmentation
        input = image_augmentation_filament(input, self.image_height, self.image_width, 10, 0.01, 0.01)

        # reshape raw image to patches
        p = self.patch_size
        pl = self.length_patch_size
        patch_length = (length//pl) * (height//p) * (width//p)
        input = rearrange(input, 'b (l pl) (h p1) (w p2) -> b (l h w) (pl p1 p2)',
                          p1=p, p2=p, pl=pl)

        mask = get_mask_subset_with_prob(input, self.mask_prob)

        # mask input with mask patches with probability of `replace_prob` (keep patches the same with probability 1 - replace_prob)
        masked_input = input.clone().detach()

        # if random token probability > 0 for mpp
        if self.random_patch_prob > 0:
            random_patch_sampling_prob = self.random_patch_prob / (
                    1 - self.replace_prob)
            random_patch_prob = prob_mask_like(input,
                                               random_patch_sampling_prob).to(mask.device)

            bool_random_patch_prob = mask * (random_patch_prob == True)
            random_patches = torch.randint(0,
                                           input.shape[1],
                                           (input.shape[0], input.shape[1]),
                                           device=input.device)
            random_patches = create_random_patches(input, padding_mask).to(mask.device)
            randomized_input = masked_input[
                torch.arange(masked_input.shape[0]).unsqueeze(-1),
                random_patches]

            masked_input[bool_random_patch_prob] = randomized_input[
                bool_random_patch_prob]

        # [mask] input
        replace_prob = prob_mask_like(input, self.replace_prob).to(mask.device)
        bool_mask_replace = (mask * replace_prob) == True

        masked_input[bool_mask_replace] = self.mask_token


        masked_input = transformer(masked_input, padding_mask,crop=False)

        cls_logits = self.to_bits(masked_input)
        logits = cls_logits[:, 1:, :]

        img = rearrange(img, 'b (l pl) (h p1) (w p2) -> b (l h w) (pl p1 p2)', p1=p, p2=p, pl=pl).contiguous()

        logits = rearrange(logits, 'b (l h w) (pl p1 p2) -> b (l pl) (h p1) (w p2)', p1=p, p2=p, h=height//p,w=width//p)
        #if ctf is not None:
        #    Apix = 1.15
        #    max_d = max(height,width)
        #    h_min,h_max,w_min,w_max = int((max_d-height)/2),int((max_d+height)/2),int((max_d-width)/2),int((max_d+width)/2)
        #    for i in range(len(logits)):
        #        ctf_i = ctf[i]
        #        n_img_ctf_corr = len(ctf_i)
        #        image_pad = torch.zeros(n_img_ctf_corr,max_d,max_d).to(logits.device)
        #        filament_tmp = logits[i,:n_img_ctf_corr]
        #        image_pad[:,h_min:h_max,w_min:w_max] = filament_tmp
        #        image_pad = ctf_correction_torch(image_pad, ctf_i, Apix)
        #        logits[i, :n_img_ctf_corr] = image_pad[:,h_min:h_max,w_min:w_max]

        logits = rearrange(logits, 'b (l pl) (h p1) (w p2) -> b (l h w) (pl p1 p2)', p1=p, p2=p, pl=pl)

        logits = self.normLayer(logits)

        mpp_loss = self.loss(logits, img, mask)

        return mpp_loss


class MPP_vector(nn.Module):
    def __init__(
            self,
            transformer,
            patch_dim,
            dim,
            output_channel_bits=1,
            channels=1,
            max_pixel_val=1.0,
            mask_prob=0.15,
            replace_prob=0.5,
            random_patch_prob=0.5,
            augment_prob=0.5,
            mean=None,
            std=None,
            lossF='l2_norm'
    ):
        super().__init__()

        self.transformer = transformer
        self.loss = MPPLoss(patch_dim, channels, output_channel_bits,
                            max_pixel_val, mean, std, lossF = lossF)

        # output transformation
        self.to_bits = nn.Sequential(nn.LayerNorm(dim),
                                     nn.Linear(dim, patch_dim))

        # vit related dimensions
        self.patch_size = patch_dim

        # mpp related probabilities
        self.mask_prob = mask_prob
        self.replace_prob = replace_prob
        self.random_patch_prob = random_patch_prob
        self.augment_prob=augment_prob

        # token ids
        self.mask_token = nn.Parameter(torch.randn(1, 1, patch_dim))

    def forward(self, input, padding_mask, **kwargs):
        transformer = self.transformer
        #print('original padding mask',padding_mask)
        matrix_mask = transformer.matrix_mask(padding_mask)
        # clone original image for loss
        img = input.clone().detach()

        mask = get_mask_subset_with_prob(input, self.mask_prob)

        # mask input with mask patches with probability of `replace_prob` (keep patches the same with probability 1 - replace_prob)
        masked_input = input.clone().detach()

        # if random token probability > 0 for mpp
        if self.random_patch_prob > 0:
            random_patch_sampling_prob = self.random_patch_prob / (
                    1 - self.replace_prob)
            random_patch_prob = prob_mask_like(input,
                                               random_patch_sampling_prob).to(mask.device)

            bool_random_patch_prob = mask * (random_patch_prob == True)
            random_patches = torch.randint(0,
                                           input.shape[1],
                                           (input.shape[0], input.shape[1]),
                                           device=input.device)
            random_patches = create_random_patches(input, padding_mask).to(mask.device)
            randomized_input = masked_input[
                torch.arange(masked_input.shape[0]).unsqueeze(-1),
                random_patches]

            masked_input[bool_random_patch_prob] = randomized_input[
                bool_random_patch_prob]

        # [mask] input
        replace_prob = prob_mask_like(input, self.replace_prob).to(mask.device)
        bool_mask_replace = (mask * replace_prob) == True

        masked_input[bool_mask_replace] = self.mask_token

        # linear embedding of patches
        masked_input = transformer.to_patch_embedding(masked_input)

        # add cls token to input sequence
        b, n, _ = masked_input.shape
        cls_tokens = repeat(transformer.cls_token, '() n d -> b n d', b=b)
        masked_input = torch.cat((cls_tokens, masked_input), dim=1)

        # add positional embeddings to input
        #masked_input += transformer.pos_embedding[:, :(n + 1)]
        masked_input += transformer.pos_embedding_sincos(masked_input)
        #masked_input = transformer.pos_embedding_fre_shift(masked_input)
        masked_input = transformer.dropout(masked_input)

        # get generator output and get mpp loss
        masked_input = transformer.transformer(masked_input, matrix_mask, **kwargs)

        cls_logits = self.to_bits(masked_input)
        logits = cls_logits[:, 1:, :]

        mpp_loss = self.loss(logits, img, mask)

        return mpp_loss
