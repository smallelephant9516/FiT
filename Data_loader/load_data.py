import mrcfile
import cv2
import numpy as np
from datetime import datetime as dt
import os,sys
from Data_loader import EMData
from Data_loader.image_transformation import crop, add_noise_SNR, normalize_all_image, padding, padding_vector, inplane_rotate, normalize_image
from Data_loader.mrcs import LazyImage, parse_header
from Data_loader.ctf_fuction import ctf_correction, low_pass_filter_images

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
            self.dataset = load_simulation(path,set_height,set_width,max_len,set_mask=set_mask,datadir=datadir)
        else:
            self.dataset = load_mrcs(path,set_height,set_width,max_len,set_mask=set_mask,datadir=datadir)
        self.data = self.dataset.data
        self.mask = self.dataset.mask

    def __getitem__(self, index):
        data = self.data[index]
        mask = self.mask[index]
        return index, data, mask

    def __len__(self):
        return len(self.data)

    def shape(self):
        return np.shape(self.data)

    def get_particles(self):
        self.dataset.all_data_image = cv2.normalize(self.dataset.all_data_image, None, 0, 1, cv2.NORM_MINMAX)
        return self.dataset.all_data_image

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
        self.all_data_image = np.concatenate((type1, type2), axis=0)
        self.n_img, self.D, _ =np.shape(self.all_data_image)
        print('total number of particles are {}, with {} dimensions'.format(self.n_img,self.D))

        # reset the image to the give filament shape
        filmanet_meta=EMData.read_data_df(path)
        dataframe=filmanet_meta.star2dataframe()
        helicaldic, filament_id=filmanet_meta.extract_helical_select(dataframe)
        filament_index=filmanet_meta.filament_index(helicaldic)

        print(self.all_data_image.min(), self.all_data_image.max())
        self.all_data_image = add_noise_SNR(self.all_data_image, 0.05)
        print(self.all_data_image.min(), self.all_data_image.max())

        # crop the image based on the dimension provided
        self.all_data_image = crop(self.all_data_image,set_height, set_width)

        if max_len>0:
            print('use max length to cut the filament: %s' % max_len)
            filament_index_new = filmanet_meta
            self.data, self.mask = padding(self.all_data_image, filament_index_new, max_len,
                                         set_mask=set_mask)
        else:
            max_len = max(map(len,filament_index))
            self.data, self.mask = padding(self.all_data_image, filament_index, max_len,
                                         set_mask=set_mask)
            print('The max length is: %s' % self.data.shape[1])

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
        filament=EMData.read_data_df(path)
        self.dataframe=filament.star2dataframe()
        helicaldic, filament_id=filament.extract_helical_select(self.dataframe)
        self.filament_index=filament.filament_index(helicaldic)

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
            self.all_data_image = np.array([x.get() for x in dataset])

        Apix=2.3
        # mode is first of phase flip or till first peak
        self.all_data_image = ctf_correction(self.all_data_image, self.dataframe, Apix, mode = 'phase flip')

        # apply low pass filter
        self.all_data_image = low_pass_filter_images(self.all_data_image, 20, apix=Apix)

        # circular normalization
        #self.all_data_image = normalize_image(self.all_data_image, 5)

        # rotate the image based on the prior
        self.all_data_image = inplane_rotate(self.all_data_image, self.dataframe['_rlnAnglePsiPrior'])

        # crop the image based on the dimension provided
        self.all_data_image = crop(self.all_data_image, self.set_height, self.set_width)
        self.all_data_image = self.all_data_image.astype('float32')

        self.n_img, _, _ =np.shape(self.all_data_image)
        print('total number of particles are {}'.format(self.n_img))

        self.data, self.mask = padding(self.all_data_image, self.filament_index, self.max_len,set_mask=self.set_mask)

        print(self.data.min(), self.data.max())

    def particle_images(self):
        return self.all_data_image

    def __getitem__(self):
        return self.data,self.mask

class load_vector():
    def __init__(self,path,max_len, set_mask=True,vector=None):

        filmanet_meta=EMData.read_data_df(path)
        dataframe=filmanet_meta.star2dataframe()
        helicaldic, filament_id=filmanet_meta.extract_helical_select(dataframe)
        filament_index=filmanet_meta.filament_index(helicaldic)
        max_len = max(map(len, filament_index))
        if vector is None:
            class_list = np.array(dataframe['_rlnClassNumber']).astype('int')
            unique_class = list(np.unique(class_list))
            n_class = len(unique_class)
            class_index = np.array([unique_class.index(class_list[i]) for i in range(len(class_list))]).astype('int')
            print('there are {} number of 2D classes'.format(n_class))
            n_class_matrix=np.identity(n_class)
            vector = n_class_matrix[class_index]

        if max_len > 0:
            print('use max length to cut the filament: %s' % max_len)
            self.max_len = max_len
        else:
            self.max_len = max(map(len,self.filament_index))
            print('The max length is: %s' % self.max_len)


        assert len(vector) == len(dataframe)

        self.data, self.mask = padding_vector(vector,filament_index,self.max_len,set_mask=set_mask)


    def __getitem__(self, index):
        data = self.data[index]
        mask = self.mask[index]
        return index, data, mask

    def __len__(self):
        return len(self.data)

    def shape(self):
        return np.shape(self.data)