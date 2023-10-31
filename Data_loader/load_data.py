import mrcfile
import cv2
import numpy as np
from datetime import datetime as dt
import torchvision.transforms as T
from Data_loader import EMData
from Data_loader.image_transformation import add_noise_SNR, normalize_all_image, padding, padding_vector, padding_lazy, inplane_rotate, normalize_image,defocus_filament
from Data_loader.mrcs import LazyImage, parse_header
from Data_loader.ctf_fuction import ctf_correction, low_pass_filter_images, ctf_correction_torch_pf

def image_preprocessing(images,defocus,psi_prior):
    if len(defocus.shape)==1:
        defocus = np.expand_dims(defocus, axis=0)
        #images = np.expand_dims(images, axis=0)
    Apix = defocus[0, 1]
    #images = np.array([x.get() for x in images])
    images = ctf_correction(images, defocus, Apix)
    #images = low_pass_filter_images(images, 20, apix=Apix)
    images = normalize_image(images, 3)
    # rotate the image based on the prior
    images = inplane_rotate(images, psi_prior)
    # crop the image based on the dimension provided
    images = images.astype('float32')
    return images

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
    def __init__(self,path,set_height,set_width,max_len,set_mask=True,datadir=None,simulation=True,
                 ctf = None, lazy=False):
        self.lazy = lazy
        self.dataset = load_mrcs(path,set_height,set_width,max_len,set_mask=set_mask,datadir=datadir,
                                 simulation=simulation, ctf=ctf, lazy=lazy)
        self.data = self.dataset.data
        self.mask = self.dataset.mask
        if ctf is not None:
            self.defocus = self.dataset.defocus
            self.defocus_filament = self.dataset.defocus_filament

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

class load_new_particles():
    def __init__(self,path,set_height,set_width,max_len,set_mask=True,datadir=None,simulation=True,
                 ctf = None,lazy=False):
        self.lazy = lazy
        self.dataset = load_mrcs(path,set_height,set_width,max_len,set_mask=set_mask,datadir=datadir,
                                 simulation=simulation, ctf=ctf, lazy=lazy)
        self.data = self.dataset.all_data_image
        if ctf is not None:
            self.defocus = self.dataset.defocus

    def __getitem__(self, index):
        if self.lazy is True:
            data = self.data[index].get()
        else:
            data = self.data[index]
        return index, data

    def __len__(self):
        return len(self.data)

    def shape(self):
        return np.shape(self.data)

    def get_dataframe(self):
        return self.dataset.dataframe

class load_mrcs():
    def __init__(self,path,set_height,set_width,max_len,set_mask,datadir,simulation,ctf,lazy=False):

        self.lazy = lazy
        self.set_height = set_height
        self.set_width = set_width
        self.set_mask = set_mask
        self.simulation = simulation
        self.ctf = ctf

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
        if self.simulation is True:
            #type1_path=self.folder+'/type1.mrcs'
            #type2_path=self.folder+'/type2.mrcs'
            #with mrcfile.open(type1_path) as mrc:
            #   type1 = mrc.data
            #with mrcfile.open(type2_path) as mrc:
            #   type2 = mrc.data
            #self.all_data_image = np.concatenate((type1, type2), axis=0)

            # load simulated particles directly
            data_path = self.folder + 'noise/' + 'images_ctf.mrcs'
            with mrcfile.open(data_path) as mrc:
                self.all_data_image = mrc.data

            # print(self.all_data_image.min(), self.all_data_image.max())
            self.all_data_image = add_noise_SNR(self.all_data_image, 0.05)
            # print(self.all_data_image.min(), self.all_data_image.max())
        else:
            image_order = list(self.dataframe['filename'])
            mic_id_order = list(self.dataframe['pid'])
            header = parse_header(self.folder + image_order[0])
            D = header.D  # image size along one dimension in pixels
            dtype = header.dtype
            stride = dtype().itemsize * D * D

            lazy=self.lazy
            dataset = [
                LazyImage(self.folder + f, (D, D), dtype, 1024 + ii * stride) for ii, f in zip(mic_id_order, image_order)]
            if lazy:
                self.all_data_image = dataset
                self.data, self.mask = padding_lazy(self.all_data_image,self.filament_index,self.max_len,set_mask=self.set_mask)
                if self.ctf is not None:
                    self.defocus = np.load(self.ctf, allow_pickle=True)
                    Apix = self.defocus[0,1]
                    defocus = self.defocus
                    self.defocus_filament = defocus_filament(defocus, self.filament_index, self.max_len)
            else:
                self.all_data_image = np.array([x.get() for x in dataset])
        if self.ctf is not None:
            self.defocus = np.load(self.ctf, allow_pickle=True)
            Apix = self.defocus[0,1]
            defocus = self.defocus
            self.defocus_filament = defocus_filament(defocus, self.filament_index, self.max_len)

            print('doing ctf correction on image')
            #np.save(self.folder+'before_correction.npy', self.all_data_image[0])
            #defocus = np.array(self.dataframe[['_rlnDefocusU', '_rlnDefocusV', '_rlnDefocusAngle']]).astype('float32')
            # mode is first of phase flip or till first peak
            self.all_data_image = ctf_correction(self.all_data_image, defocus, Apix, mode = 'phase flip')
            # apply low pass filter
            self.all_data_image = low_pass_filter_images(self.all_data_image, 20, apix=Apix)
            #np.save(self.folder + 'after_correction_pf.npy', self.all_data_image[0])

        # circular normalization
        self.all_data_image = normalize_image(self.all_data_image, 5)

        # rotate the image based on the prior
        self.all_data_image = inplane_rotate(self.all_data_image, self.dataframe['_rlnAnglePsiPrior'])
        # crop the image based on the dimension provided
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