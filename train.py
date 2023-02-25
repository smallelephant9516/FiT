from Model.ViT_model import ViT_3D
from Model.MPP import MPP_3D
from Data_loader import EMData
from Data_loader.image_transformation import crop, add_noise_SNR, padding
from Data_loader.load_data import load_new

import torch
import argparse
import os, sys
from torch.utils.data import DataLoader



import mrcfile
import cv2
import numpy as np
from datetime import datetime as dt

# python train.py /home/jiang/li3221/scratch/simmicro/10340-tau/Noise/NoNoise_1pitch/join_particles.star
# python train.py /home/jiang/li3221/scratch/simmicro/10340-tau/Noise/NoNoise_1p_fix/join_particles.star -n 10
# python train.py /home/jiang/li3221/scratch/simmicro/10340-tau/Noise/NoNoise_short_test/join_particles.star -n 10
# python train.py /home/jiang/li3221/scratch/simmicro/10340-tau/Noise/NoNoise_uneven/join_particles.star -n 100

def add_args(parser):
    parser.add_argument('particles', type=os.path.abspath, help='Input particles path (.star)')
    parser.add_argument('--dim', type=int, default=256, help='Dimension of latent variable')
    parser.add_argument('--max_len', type=int, default=0, help='Number of segments in a filament')


    group = parser.add_argument_group('Data loader parameters')
    group.add_argument('--cylinder_mask', type=int, default=256,help='mask around the helix')
    group.add_argument('--center_mask', type=int, default=32, help='mask around the helix')

    group = parser.add_argument_group('Transformer parameters')
    group.add_argument('-n', '--num_epochs', type=int, default=50, help='Number of training epochs (default: %(default)s)')
    group.add_argument('-b','--batch_size', type=int, default=4, help='Minibatch size (default: %(default)s)')
    group.add_argument('--heads', type=int, default=4, help='number of heads')
    group.add_argument('--depth', type=int, default=3, help='number of layers')
    group.add_argument('--image_patch_size', type=int, default=32, help='image patch size (pix)')
    group.add_argument('--length_patch_size', type=int, default=1, help='length patch size (pix)')
    group.add_argument('--lr', type=float, default=3e-4, help='Learning rate in Adam optimizer (default: %(default)s)')
    group.add_argument('--ignore_padding_mask', action='store_false', help='Parallelize training across all detected GPUs')

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

    ## import the images
    #path=os.path.dirname(star_path)
    #type1_path=path+'/type1.mrcs'
    #type2_path=path+'/type2.mrcs'
    #with mrcfile.open(type1_path) as mrc:
    #    type1 = mrc.data
    #with mrcfile.open(type2_path) as mrc:
    #    type2 = mrc.data
#
    ## crop the image
    #set_height = args.cylinder_mask
    #set_width = args.center_mask
    #all_data_image = np.concatenate((type1, type2), axis=0)
    #all_data_image = crop(all_data_image, set_height, set_width)
    #print(np.shape(all_data_image))
    #all_data_image = np.concatenate((all_data_image,np.zeros((1,set_height,set_width))),axis=0)
#
    ## reset the image to the give filament shape
    #filmanet_meta=EMData.read_data_df(star_path)
    #dataframe=filmanet_meta.star2dataframe()
    #helicaldic, filament_index=filmanet_meta.extract_helical_select(dataframe)
    #filament_index=filmanet_meta.filament_index(helicaldic)
#
#
    #max_len=args.max_len
    #if max_len>0:
    #    print('use max length to cut the filament: %s' % max_len)
    #    filament_index_new = filmanet_meta
    #    n_filaments_new = len(filament_index)
    #    all_data, all_mask = padding(all_data_image, filament_index_new, max_len, set_height, set_width,
    #                                 set_mask=args.ignore_padding_mask)
    #else:
    #    max_len = max(map(len,filament_index))
    #    n_filaments = len(filament_index)
    #    all_data, all_mask = padding(all_data_image, filament_index, max_len, set_height, set_width,
    #                                 set_mask=args.ignore_padding_mask)
    #    print('The max length is: %s' % all_data.shape[1])
#
    #all_data = cv2.normalize(all_data, None, 0, 1, cv2.NORM_MINMAX)
    #all_data = add_noise_SNR(all_data, 0.005)
    # all_data=cv2.normalize(all_data,None,0,1,cv2.NORM_MINMAX)

    all_data=load_new(args.particles,args.cylinder_mask,args.center_mask,args.max_len,set_mask=args.ignore_padding_mask)
    n_data, length, height, width = all_data.shape()
    print(n_data, length, height, width)

    device = torch.device('cuda:1' if torch.cuda.is_available() is True else 'cpu')

    # check the dimension of the height, width and length
    assert height % args.image_patch_size == 0
    assert width % args.image_patch_size == 0
    assert length % args.length_patch_size == 0

    model = ViT_3D(
        image_height = height,
        image_width = width,
        image_patch_size=args.image_patch_size,
        length = length,
        length_patch_size = args.length_patch_size,
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
    mpp_trainer = MPP_3D(
        transformer=model,
        patch_size=args.image_patch_size,
        length_patch_size = args.length_patch_size,
        dim=args.dim,
        mask_prob=args.mask_prob,          # probability of using token in masked prediction task
        random_patch_prob=args.random_patch_prob,  # probability of randomly replacing a token being used for mpp
        replace_prob=args.replace_prob,       # probability of replacing a token being used for mpp with the mask token
        augment_prob=args.augment_prob
    )
    mpp_trainer.to(device)
    opt = torch.optim.Adam(mpp_trainer.parameters(), lr=args.lr)

    data_batch = DataLoader(all_data, batch_size=args.batch_size, shuffle=True)
    t2=dt.now()
    print('passed time for setting up parameter is {}'.format((t2-t1)))

    for i in range(args.num_epochs):
        total_loss = 0
        for index, batch, mask in data_batch:
            images = batch.to(device)
            mask = mask.to(device)
            loss = mpp_trainer(images,mask)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item() / (length * height)
        print(dt.now()-t1,'In iteration {}, the total loss is {:.4f}'.format(i, total_loss))

    data_output = DataLoader(all_data, batch_size=args.batch_size, shuffle=False)
    all_value = torch.tensor([])
    for index, batch, mask in data_output:
        #import image
        image = batch.to(device)
        mask = mask.to(device)
        #generate mask
        model.matrix_mask(mask)
        # pass through the model
        mask_patches=model.mask_cls
        value_hidden = model.forward(image)
        value_hidden[mask_patches == 0] = 0
        #print(value_hidden[0,-100:,0])
        value = value_hidden[:, 1:, :].mean(axis=1).detach().cpu()
        all_value = torch.cat((all_value, value), 0)
    print(all_value.shape)


    import umap
    from sklearn.decomposition import PCA
    from sklearn.cluster import KMeans
    import matplotlib.pyplot as plt

    all_value_np=all_value.detach().numpy()

    #for i in range(len(all_value_np)):
    #    print(i, np.isnan(all_value_np[i].mean()), np.isinf(all_value_np[i].mean()))
    #pca = PCA(n_components=3)
    #data_pca=pca.fit_transform(all_value_np)
    #print(data_pca[0],data_pca[1])

    fit = umap.UMAP(n_neighbors=15)
    data_umap = fit.fit_transform(all_value_np)

    #plt.figure(figsize=(10,10))
    label=[0]*50+[1]*50
    plt.scatter(data_umap[:,0],data_umap[:,1],c=label)
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    main(add_args(parser).parse_args())