from Model.ViT_model import ViT, ViT_vector
from Model.MPP import MPP, MPP_vector
from Data_loader import EMData
from Data_loader.load_data import load_new_particles, load_vector, image_preprocessing
from Data_loader.image_transformation import crop

import torch
import argparse
import os, sys
from torch.utils.data import DataLoader

import numpy as np
from datetime import datetime as dt

'''Training a transformer on image and then transformer on the image vector'''

def add_args(parser):
    parser.add_argument('particles', type=os.path.abspath, help='Input particles path (.star)')
    parser.add_argument('--output', type=os.path.abspath, help='The place for output the star file')
    parser.add_argument('--dim', type=int, default=128, help='Dimension of latent variable')
    parser.add_argument('--max_len', type=int, default=0,
                        help='Number of segments in a filament, 0 means using the max length')


    group = parser.add_argument_group('Data loader parameters')
    group.add_argument('--cylinder_mask', type=int, default=64,help='mask around the helix')
    group.add_argument('--center_mask', type=int, default=128, help='mask in the center of the helix')
    group.add_argument("--lazy", action='store_true', help="not loading the data from the main memory")
    group.add_argument("--datadir",help="Optionally provide path to input .mrcs if loading from a .star or .cs file")
    group.add_argument("--simulation",action='store_true', help="Use the simulation dataset or not")
    group.add_argument("--ctf_path", type=os.path.abspath, help="Use the ctf file in dataset or not")

    group = parser.add_argument_group('Transformer parameters')
    group.add_argument('-n', '--num_epochs', type=int, default=50, help='Number of training epochs (default: %(default)s)')
    group.add_argument('-b','--batch_size', type=int, default=32, help='Minibatch size (default: %(default)s)')
    group.add_argument('--heads', type=int, default=4, help='number of heads')
    group.add_argument('--depth', type=int, default=3, help='number of layers')
    group.add_argument('--image_patch_size', type=int, default=32, help='image patch size (pix)')
    group.add_argument('--length_patch_size', type=int, default=1, help='length patch size (pix)')
    group.add_argument('--lr', type=float, default=2e-4, help='Learning rate in Adam optimizer (default: %(default)s)')
    group.add_argument('--ignore_padding_mask', action='store_true', help='Ignore the padding mask or not')
    group.add_argument('--loss', type=str, default='l2_norm', help='loss function (l2_norm, l1_norm, cross_entropy)')

    group = parser.add_argument_group('Mask Patch parameter')
    group.add_argument('--mask_prob', type=float, default=0.15, help='probablility to mask the patch')
    group.add_argument('--random_patch_prob', type=float, default=0.3, help='probablility to replace the token with anthoer')
    group.add_argument('--replace_prob', type=float, default=0.5, help='probablility to mask the patch')
    group.add_argument('--augment_prob', type=float, default=0.5, help='probablility to do augmentation transforming to the image')

    return parser

# import the image
def main(args):

    # check parameter and path of the metafile
    t1 = dt.now()
    print(args)
    star_path = args.particles
    assert os.path.splitext(star_path)[1] == '.star'

# determine whether set mask or not

    if args.ignore_padding_mask is True:
        set_mask = False
    else:
        set_mask = True

    all_data=load_new_particles(args.particles,args.cylinder_mask,args.center_mask,args.max_len,set_mask=set_mask,
                      datadir=args.datadir,simulation=args.simulation,ctf=args.ctf_path,lazy=args.lazy)
    dataframe = all_data.get_dataframe()
    if args.ctf_path is not None:
        defocus = all_data.defocus
        print(defocus.shape)

    device = torch.device('cuda:3' if torch.cuda.is_available() is True else 'cpu')

    model = ViT(
        image_height = args.cylinder_mask,
        image_width = args.center_mask,
        image_patch_size=args.image_patch_size,
        num_classes=1000,
        dim=args.dim,
        depth=args.depth,
        heads=args.heads,
        mlp_dim=2048,
        dropout=0.1,
        emb_dropout=0.1
    )
    model.to(device)
    mpp_trainer = MPP(
        transformer=model,
        image_height=args.cylinder_mask,
        image_width=args.center_mask,
        patch_size=args.image_patch_size,
        dim=args.dim,
        mask_prob=args.mask_prob,          # probability of using token in masked prediction task
        random_patch_prob=args.random_patch_prob,  # probability of randomly replacing a token being used for mpp
        replace_prob=args.replace_prob,       # probability of replacing a token being used for mpp with the mask token
        lossF=args.loss,
    )
    mpp_trainer.to(device)
    opt = torch.optim.Adam(mpp_trainer.parameters(), lr=args.lr)

    data_batch = DataLoader(all_data, batch_size=args.batch_size, shuffle=True)
    t2=dt.now()
    print('passed time for setting up parameter is {}'.format((t2-t1)))

    if args.lazy is True:
        psi_prior_all = np.array(dataframe['_rlnAnglePsiPrior']).astype('float32')

    for epoch in range(args.num_epochs):
        total_loss = 0
        count = 0
        for index, batch in data_batch:
            ctf = None
            if args.ctf_path is not None:
                ctf = defocus[index]
            if args.lazy is True:
                psi_prior = psi_prior_all[index]
                batch = image_preprocessing(batch,ctf,psi_prior)
                batch = torch.tensor(batch)
            images = batch.to(device)
            loss = mpp_trainer(images,ctf)
            opt.zero_grad()
            loss.backward(retain_graph=True)
            opt.step()
            loss_value = loss.item()
            total_loss += loss_value
            count += len(batch)
            if count % 1000 ==0:
                print('{}/{} of particles has been processed with loss of {}'.format(count,len(dataframe),loss_value/len(batch)))
        total_loss = total_loss/count
        print(dt.now()-t1,'In iteration {}, the total loss is {:.4f}'.format(epoch, total_loss))

    data_output = DataLoader(all_data, batch_size=args.batch_size, shuffle=False)
    all_value = torch.tensor([])
    for index, batch in data_output:
        #import image
        ctf = None
        if args.ctf_path is not None:
            ctf = defocus[index]
        if args.lazy is True:
            psi_prior = psi_prior_all[index]
            batch = image_preprocessing(batch, ctf, psi_prior)
            batch = torch.tensor(batch)
        image = batch.to(device)
        image = crop(image, args.cylinder_mask, args.center_mask)
        value_hidden = model.forward(image).detach().cpu()
        all_value = torch.cat((all_value, value_hidden), 0)
    print(all_value.shape)
    all_value_np=all_value.detach().numpy()
    vector = all_value_np[:,1:,:].mean(axis=1)

    all_data=load_vector(args.particles,args.max_len,set_mask=set_mask, vector=vector)
    n_data, length, patch_dim = all_data.shape()
    print(n_data, length, patch_dim)

    model = ViT_vector(
        length = length,
        patch_dim = patch_dim,
        num_classes=1000,
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

    for epoch in range(100):
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
    output_path = save_dir+'/saved_TT_embedding_{}.npy'.format(args.num_epochs-1)
    print(dt.now(),' The output vector is saved to %s' % output_path)
    np.save(output_path, all_value_np)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    main(add_args(parser).parse_args())