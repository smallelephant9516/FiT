import mrcfile
import cv2
import numpy as np
from datetime import datetime as dt
import os,sys
from Data_loader import EMData
from Data_loader.image_transformation import crop, add_noise_SNR, padding

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
    def __init__(self,path,set_height,set_width,max_len,set_mask=True):
        self.data,self.mask = load_simulation(path,set_height,set_width,max_len,set_mask=set_mask)

    def __getitem__(self, index):
        data = self.data[index]
        mask = self.mask[index]
        return index, data, mask

    def __len__(self):
        return len(self.data)

    def shape(self):
        return np.shape(self.data)


class load_simulation():
    def __init__(self,path,set_height,set_width,max_len,set_mask=True):
        folder=os.path.dirname(path)
        type1_path=folder+'/type1.mrcs'
        type2_path=folder+'/type2.mrcs'
        with mrcfile.open(type1_path) as mrc:
            type1 = mrc.data
        with mrcfile.open(type2_path) as mrc:
            type2 = mrc.data

        # crop the image
        all_data_image = np.concatenate((type1, type2), axis=0)
        all_data_image = crop(all_data_image, set_height, set_width)
        print(np.shape(all_data_image))
        all_data_image = np.concatenate((all_data_image,np.zeros((1,set_height,set_width))),axis=0)

        # reset the image to the give filament shape
        filmanet_meta=EMData.read_data_df(path)
        dataframe=filmanet_meta.star2dataframe()
        helicaldic, filament_index=filmanet_meta.extract_helical_select(dataframe)
        filament_index=filmanet_meta.filament_index(helicaldic)


        if max_len>0:
            print('use max length to cut the filament: %s' % max_len)
            filament_index_new = filmanet_meta
            n_filaments_new = len(filament_index)
            all_data, all_mask = padding(all_data_image, filament_index_new, max_len, set_height, set_width,
                                         set_mask=set_mask)
        else:
            max_len = max(map(len,filament_index))
            n_filaments = len(filament_index)
            self.data, self.mask = padding(all_data_image, filament_index, max_len, set_height, set_width,
                                         set_mask=set_mask)
            print('The max length is: %s' % self.data.shape[1])

        self.data = cv2.normalize(self.data, None, 0, 1, cv2.NORM_MINMAX)
        self.data = add_noise_SNR(self.data, 0.005)
        # all_data=cv2.normalize(all_data,None,0,1,cv2.NORM_MINMAX)

    def __getitem__(self):
        return self.data, self.mask