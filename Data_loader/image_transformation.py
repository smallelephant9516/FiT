import numpy as np

def crop(images, set_height, set_width):
    img_dim = np.shape(images)[-1]
    h1 = int((img_dim - set_height) / 2)
    h2 = int((img_dim + set_height) / 2)
    w1 = int((img_dim - set_width) / 2)
    w2 = int((img_dim + set_width) / 2)
    images = images[:, h1:h2, w1:w2]
    return images

def add_noise_SNR(image, snr):
    b,l,x,y=np.shape(image)
    xpower = np.var(image)
    npower = xpower / snr
    print('noise sigma is: ',npower)
    images_new=image + np.random.normal(0,1,(b,l,x,y)) * np.sqrt(npower)
    images_new=images_new.astype('float32')
    return images_new

def padding(all_data_image,filament_index,length,height,width,mode='pad',set_mask=True):
    n_filament = len(filament_index)
    output=np.zeros((n_filament,length,height,width))
    mask = np.zeros((n_filament, length))
    if mode == 'pad':
        for i in range(n_filament):
            lst=filament_index[i]
            output[i,:len(lst),:,:]=all_data_image[lst]
            if set_mask is True:
                mask[i, :len(lst)] = 1
            else:
                mask[i, :] = 1
    return output, mask