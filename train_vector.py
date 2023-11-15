from Model.ViT_model import ViT_vector
from Model.MPP import MPP_vector
from Data_loader import EMData
from Data_loader.load_data import load_vector

import torch
import argparse
import os, sys
from torch.utils.data import DataLoader



import mrcfile
import cv2
import numpy as np
from datetime import datetime as dt


def add_args(parser):
    parser.add_argument('particles', type=os.path.abspath, help='Input particles path (.star)')
    parser.add_argument('--output', type=os.path.abspath, help='The place for output the star file')
    parser.add_argument('--dim', type=int, default=128, help='Dimension of latent variable')
    parser.add_argument('--max_len', type=int, default=0,
                        help='Number of segments in a filament, 0 means using the max length')
    parser.add_argument('--device', type=int, default=0, help='gpu device number')


    group = parser.add_argument_group('Data loader parameters')
    group.add_argument("--datadir",help="Optionally provide path to input .mrcs if loading from a .star or .cs file")
    group.add_argument("--simulation",action='store_true', help="Use the simulation dataset or not")
    parser.add_argument('--vector_path', type=os.path.abspath, default=None, help='The path for the particle vector')

    group = parser.add_argument_group('Transformer parameters')
    group.add_argument('-n', '--num_epochs', type=int, default=50, help='Number of training epochs (default: %(default)s)')
    group.add_argument('-b','--batch_size', type=int, default=32, help='Minibatch size (default: %(default)s)')
    group.add_argument('--heads', type=int, default=4, help='number of heads')
    group.add_argument('--depth', type=int, default=3, help='number of layers')
    group.add_argument('--lr', type=float, default=3e-5, help='Learning rate in Adam optimizer (default: %(default)s)')
    group.add_argument('--ignore_padding_mask', action='store_true', help='not using the padding mask to mask the transformer')
    group.add_argument('--loss', type=str, default='l2_norm', help='loss function (l2_norm, l1_norm, cross_entropy)')
    group.add_argument('--vector_cls_token', type=str, default='average', help='The token usage for the vector transformer (cls, average)')

    group = parser.add_argument_group('Mask Patch parameter')
    group.add_argument('--mask_prob', type=float, default=0.15, help='probability of using token in masked prediction task')
    group.add_argument('--random_patch_prob', type=float, default=0.3, help='probability of randomly replacing a token being used for mpp')
    group.add_argument('--replace_prob', type=float, default=0.5, help='probability of replacing a token being used for mpp with the mask token')
    group.add_argument('--augment_prob', type=float, default=0.5, help='probablility to do augmentation transforming to the image')

    return parser

# import the image
def main(args):

    # check parameter and path of the metafile
    t1 = dt.now()
    print(args)
    star_path = args.particles
    assert os.path.splitext(star_path)[1] == '.star'

    if args.ignore_padding_mask is True:
        set_mask = False
    else:
        print('using mask to mask the unwanted region in transformer')
        set_mask = True
    if args.vector_path is not None:
        vector = np.load(args.vector_path, allow_pickle=True)
    else:
        vector = None

    all_data=load_vector(args.particles,args.max_len,set_mask=set_mask, vector=vector)
    n_data, length, patch_dim = all_data.shape()
    print(n_data, length, patch_dim)

    device = torch.device('cuda:{}'.format(args.device) if torch.cuda.is_available() is True else 'cpu')

    model = ViT_vector(
        length = length,
        patch_dim = patch_dim,
        dim=args.dim,
        depth=args.depth,
        heads=args.heads,
        batch_size=args.batch_size,
        mlp_dim=2048,
        dropout=0.1,
        emb_dropout=0.1
    )
    model.to(device)
    mpp_trainer = MPP_vector(
        transformer=model,
        patch_dim=patch_dim,
        dim=args.dim,
        mask_prob=args.mask_prob,          # probability of using token in masked prediction task
        random_patch_prob=args.random_patch_prob,  # probability of randomly replacing a token being used for mpp
        replace_prob=args.replace_prob,       # probability of replacing a token being used for mpp with the mask token
        augment_prob=args.augment_prob,
        lossF=args.loss,
    )
    mpp_trainer.to(device)
    opt = torch.optim.Adam(mpp_trainer.parameters(), lr=args.lr)

    data_batch = DataLoader(all_data, batch_size=args.batch_size, shuffle=True)
    t2=dt.now()
    print('passed time for setting up parameter is {}'.format((t2-t1)))

    for epoch in range(args.num_epochs):
        total_loss = 0
        for index, batch, mask in data_batch:
            images = batch.to(device)
            mask = mask.to(device)
            loss = mpp_trainer(images,mask)
            opt.zero_grad()
            loss.backward(retain_graph=True)
            opt.step()
            total_loss += loss.item() / (length)
        print(dt.now(),dt.now()-t1,'In iteration {}, the total loss is {:.5f}'.format(epoch, total_loss))

    data_output = DataLoader(all_data, batch_size=args.batch_size, shuffle=False)
    all_value_np = np.zeros((n_data, args.dim))
    output_mode='average'
    for i, (index, batch, mask) in enumerate(data_output):
        #import image
        image = batch.to(device)
        mask = mask.to(device)
        #generate mask
        model.matrix_mask(mask)
        # pass through the model
        mask_patches=model.mask_cls
        value_hidden = model.forward(image)
        value_hidden[mask_patches == 0, :] = 0
        if output_mode == 'average':
            value_sum = value_hidden[:, 1:, :].sum(axis=1).detach().cpu().numpy()
            n_index = np.array(list(map(lambda x: len(x[x == 1]), mask)))
            value = ((1 / n_index) * value_sum.T).T
        elif output_mode =='cls':
            value = value_hidden[:, 0, :].detach().cpu().numpy()
        all_value_np[i*args.batch_size:(i+1)*args.batch_size,:] = value
    print(all_value_np.shape)


    if args.output is not None:
        save_dir = args.output
    else:
        save_dir = os.path.dirname(args.particles)
    print(dt.now(),' The output vector is saved to %s' % save_dir)
    np.save(save_dir+'/saved_vector_embedding_{}.npy'.format(epoch), all_value_np)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    main(add_args(parser).parse_args())