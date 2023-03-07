from Model.ViT_model import ViT
from Model.MPP import MPP
from Data_loader import EMData
from Data_loader.load_data import load_new

import torch
import argparse
import os, sys
from torch.utils.data import DataLoader

import numpy as np
from datetime import datetime as dt

def add_args(parser):
    parser.add_argument('particles', type=os.path.abspath, help='Input particles path (.star)')
    parser.add_argument('--output', type=os.path.abspath, help='The place for output the star file')
    parser.add_argument('--dim', type=int, default=128, help='Dimension of latent variable')
    parser.add_argument('--max_len', type=int, default=0,
                        help='Number of segments in a filament, 0 means using the max length')


    group = parser.add_argument_group('Data loader parameters')
    group.add_argument('--cylinder_mask', type=int, default=64,help='mask around the helix')
    group.add_argument('--center_mask', type=int, default=128, help='mask around the helix')
    group.add_argument("--datadir",help="Optionally provide path to input .mrcs if loading from a .star or .cs file")
    group.add_argument("--simulation",action='store_true', help="Use the simulation dataset or not")

    group = parser.add_argument_group('Transformer parameters')
    group.add_argument('-n', '--num_epochs', type=int, default=50, help='Number of training epochs (default: %(default)s)')
    group.add_argument('-b','--batch_size', type=int, default=4, help='Minibatch size (default: %(default)s)')
    group.add_argument('--heads', type=int, default=4, help='number of heads')
    group.add_argument('--depth', type=int, default=3, help='number of layers')
    group.add_argument('--image_patch_size', type=int, default=32, help='image patch size (pix)')
    group.add_argument('--length_patch_size', type=int, default=1, help='length patch size (pix)')
    group.add_argument('--lr', type=float, default=2e-4, help='Learning rate in Adam optimizer (default: %(default)s)')
    group.add_argument('--ignore_padding_mask', action='store_true', help='Ignore the padding mask or not')

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

    all_data=load_new(args.particles,args.cylinder_mask,args.center_mask,args.max_len,set_mask=set_mask,
                      datadir=args.datadir,simulation=args.simulation).get_particles()
    n_data, height, width = all_data.shape
    print(n_data, height, width)

    device = torch.device('cuda:1' if torch.cuda.is_available() is True else 'cpu')

    # check the dimension of the height, width and length
    assert height % args.image_patch_size == 0
    assert width % args.image_patch_size == 0

    model = ViT(
        image_height = height,
        image_width = width,
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
        patch_size=args.image_patch_size,
        dim=args.dim,
        mask_prob=args.mask_prob,          # probability of using token in masked prediction task
        random_patch_prob=args.random_patch_prob,  # probability of randomly replacing a token being used for mpp
        replace_prob=args.replace_prob,       # probability of replacing a token being used for mpp with the mask token
    )
    mpp_trainer.to(device)
    opt = torch.optim.Adam(mpp_trainer.parameters(), lr=args.lr)

    data_batch = DataLoader(all_data, batch_size=args.batch_size, shuffle=True)
    t2=dt.now()
    print('passed time for setting up parameter is {}'.format((t2-t1)))

    for epoch in range(args.num_epochs):
        total_loss = 0
        for batch in data_batch:
            images = batch.to(device)
            loss = mpp_trainer(images)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item() / (width * height)
        print(dt.now()-t1,'In iteration {}, the total loss is {:.4f}'.format(epoch, total_loss))

    data_output = DataLoader(all_data, batch_size=args.batch_size, shuffle=False)
    all_value = torch.tensor([])
    for batch in data_output:
        #import image
        image = batch.to(device)
        value_hidden = model.forward(image).detach().cpu()
        all_value = torch.cat((all_value, value_hidden), 0)
    print(all_value.shape)
    all_value_np=all_value.detach().numpy()

    filament = EMData.read_data_df(args.particles)
    dataframe = filament.star2dataframe()
    helicaldic, filament_id = filament.extract_helical_select(dataframe)
    filament_index = filament.filament_index(helicaldic)

    all_filament_np=[]
    for i in range(len(filament_index)):
        lst=filament_index[i]
        all_filament_np.append(all_value_np[lst].mean(axis=0))
    all_filament_np=np.array(all_filament_np)


    if args.output is not None:
        save_dir = args.output
    else:
        save_dir = os.path.dirname(args.particles)
    print('The output vector is saved to %s' % save_dir)
    np.save(save_dir+'/saved_particles_embedding_{}.npy'.format(epoch), all_filament_np)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    main(add_args(parser).parse_args())