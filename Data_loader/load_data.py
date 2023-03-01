import mrcfile
import cv2
import numpy as np
from datetime import datetime as dt
import os,sys
from Data_loader import EMData
from Data_loader.image_transformation import crop, add_noise_SNR, padding, inplane_rotate
from Data_loader.mrcs import LazyImage, parse_header
from Data_loader.ctf_fuction import ctf_correction

class load():
    def __init__(self,data):
        self.data = data

    def __getitem__(self, index):
        data = self.data[index]
        return index, data

    def __len__(self):
        return len(self.data)

    def shape(self):
        return np.shape(self.data)

class load_new():
    def __init__(self,path,set_height,set_width,max_len,set_mask=True,datadir=None,simulation=True):
        if simulation is True:
            dataset = load_simulation(path,set_height,set_width,max_len,set_mask=set_mask,datadir=datadir)
        else:
            dataset = load_mrcs(path,set_height,set_width,max_len,set_mask=set_mask,datadir=datadir)
        self.data = dataset.data
        self.mask = dataset.mask

    def __getitem__(self, index):
        data = self.data[index]
        mask = self.mask[index]
        return index, data, mask

    def __len__(self):
        return len(self.data)

    def shape(self):
        print(type(self.data))
        return np.shape(self.data)


class load_simulation():
    def __init__(self,path,set_height,set_width,max_len,set_mask,datadir):
        if datadir is None:
            folder = os.path.dirname(path)
        else:
            folder=datadir

        type1_path=folder+'/type1.mrcs'
        type2_path=folder+'/type2.mrcs'
        with mrcfile.open(type1_path) as mrc:
            type1 = mrc.data
        with mrcfile.open(type2_path) as mrc:
            type2 = mrc.data

        # crop the image
        all_data_image = np.concatenate((type1, type2), axis=0)
        all_data_image = crop(all_data_image, set_height, set_width)
        self.n_img, self.D, _ =np.shape(all_data_image)
        print('total number of particles are {}, with {} dimensions'.format(self.n_img,self.D))
        all_data_image = np.concatenate((all_data_image, np.zeros((1, set_height, set_width))), axis=0)

        # reset the image to the give filament shape
        filmanet_meta=EMData.read_data_df(path)
        dataframe=filmanet_meta.star2dataframe()
        helicaldic, filament_id=filmanet_meta.extract_helical_select(dataframe)
        filament_index=filmanet_meta.filament_index(helicaldic)


        if max_len>0:
            print('use max length to cut the filament: %s' % max_len)
            filament_index_new = filmanet_meta
            self.data, self.mask = padding(all_data_image, filament_index_new, max_len, set_height, set_width,
                                         set_mask=set_mask)
        else:
            max_len = max(map(len,filament_index))
            self.data, self.mask = padding(all_data_image, filament_index, max_len, set_height, set_width,
                                         set_mask=set_mask)
            print('The max length is: %s' % self.data.shape[1])
        print(self.data.min(),self.data.max())
        self.data = cv2.normalize(self.data, None, 0, 1, cv2.NORM_MINMAX)
        self.data = add_noise_SNR(self.data, 0.05)
        # all_data=cv2.normalize(all_data,None,0,1,cv2.NORM_MINMAX)

    def __getitem__(self):
        data = self.data
        mask = self.mask
        return data,mask


class load_mrcs():
    def __init__(self,path,set_height,set_width,max_len,set_mask,datadir):

        self.set_height = set_height
        self.set_width = set_width
        self.set_mask = set_mask

        if datadir is None:
            folder_list = path.split('/')
            self.folder = '/'.join(folder_list[:-3])+'/'
        else:
            self.folder = datadir+'/'

        # reset the image to the give filament shape
        filmanet_meta=EMData.read_data_df(path)
        self.dataframe=filmanet_meta.star2dataframe()
        helicaldic, filament_id=filmanet_meta.extract_helical_select(self.dataframe)
        self.filament_index=filmanet_meta.filament_index(helicaldic)

        if max_len > 0:
            print('use max length to cut the filament: %s' % max_len)
            self.max_len = max_len
        else:
            self.max_len = max(map(len,self.filament_index))
            print('The max length is: %s' % self.max_len)

        self.get_image()

    def get_image(self):
        image_order = list(self.dataframe['filename'])
        mic_id_order = list(self.dataframe['pid'])
        header = parse_header(self.folder+image_order[0])
        D = header.D  # image size along one dimension in pixels
        dtype = header.dtype
        stride = dtype().itemsize * D * D

        lazy=False
        dataset = [
            LazyImage(self.folder + f, (D, D), dtype, 1024 + ii * stride) for ii, f in zip(mic_id_order, image_order)]
        if not lazy:
            all_data_image = np.array([x.get() for x in dataset])

        Apix=6.9
        all_data_image = ctf_correction(all_data_image, self.dataframe, Apix, mode = 'first')
        all_data_image = inplane_rotate(all_data_image, self.dataframe['_rlnAnglePsiPrior'].astype('float32'))

        self.n_img, self.D, _ =np.shape(all_data_image)
        print('total number of particles are {}, with {} diameter of'.format(self.n_img,self.D))

        #all_data_image = cv2.normalize(all_data_image, None, 0, 1, cv2.NORM_MINMAX)
        #all_data_image = np.concatenate((all_data_image, np.zeros((1, self.D, self.D))), axis=0)

        self.data, self.mask = padding(all_data_image, self.filament_index, self.max_len, self.set_height,
                                       self.set_width,set_mask=self.set_mask)

        print(self.data.min(), self.data.max())

    def __getitem__(self):
        data = self.data
        mask = self.mask
        return data,mask