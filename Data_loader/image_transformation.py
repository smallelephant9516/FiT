import numpy as np
import cv2

def crop(images, set_height, set_width):
    img_dim = np.shape(images)[-1]
    assert (img_dim >= set_height) & (img_dim >= set_width)
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

def cut_corpus(corpus,cut_length):
    cut_index=[]
    new_corpus=[]
    cut_length=cut_length
    print(len(corpus))
    for i in range(len(corpus)):
        lst=corpus[i]
        n=len(lst)
        if n<=cut_length:
            new_corpus.append(lst)
            continue
        if n%cut_length==0:
            cut_amount=int(n/cut_length)
        else:
            cut_amount=int((n-n%cut_length)/cut_length)+1
        for j in range(cut_amount-1):
            cut_index.append(i)
            new_corpus.append(lst[j*cut_length:(j+1)*cut_length])
        new_corpus.append(lst[(cut_amount-1)*cut_length:])
    print(len(new_corpus))
    return new_corpus,cut_index

def padding(all_data_image, filament_index, length, height, width, set_mask=True):

    max_len = max(map(len, filament_index))
    #all_data_image = all_data_image.astype('float32')
    #all_data_image = crop(all_data_image, height, width)

    #all_data_image = cv2.normalize(all_data_image, None, 0, 1, cv2.NORM_MINMAX)
    #all_data_image = np.concatenate((all_data_image, np.zeros((1, height, width))), axis=0)

    # change the filament index accordingly
    if length >= max_len:
        filament_index = filament_index
    elif length < max_len:
        filament_index, cut_index = cut_corpus(filament_index, length)

    n_filament = len(filament_index)
    output = np.zeros((n_filament, length, height, width))
    mask = np.zeros((n_filament, length))
    for i in range(n_filament):
        lst = filament_index[i]
        output[i, :len(lst), :, :] = all_data_image[lst]
        if set_mask is True:
            mask[i, :len(lst)] = 1
        else:
            mask[i, :] = 1
    output=output.astype('float32')
    return output, mask


# circular mask
def get_circular_mask(data, R=None):
    n, m = data.shape[1:]
    if R:
        radius = R
    else:
        radius = min(n, m) / 2
    y_grid, x_grid = np.ogrid[:n, :m]
    center = np.array([n / 2, m / 2])
    dist = np.sqrt((center[0] - y_grid) ** 2 + (center[1] - x_grid) ** 2)
    mask = dist <= radius
    background = dist > radius
    return mask, background


def create_mask(all_image):
    mask, background = get_circular_mask(all_image)
    batchmask = np.stack([mask] * len(all_image))
    data_mask = all_image * batchmask
    return data_mask


def rotate_image(image, angle):
    image_center = tuple(np.array(image.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(image_center, -angle, 1.0)
    result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
    return result


#def rotate_image_torch(image, angle):
#    transform = T.RandomRotation(degrees=(angle, angle))
#    result = transform(image)
#    return result

def normalize_image(data,sigma=None):
    print('normalizing the image')
    mask, background = get_circular_mask(data)
    data_new = np.zeros(np.shape(data))
    for i in range(len(data)):
        img = data[i]
        background_value = img[background].flatten()
        mean = background_value.mean()
        std = background_value.std()
        img[background] = 0
        img[mask] = (img[mask]-mean)/std
        if sigma is not None:
            img[img > sigma] = 0
            img[img < -sigma] = 0
        data_new[i]=img
    return data_new

def inplane_rotate(data, theta):
    assert len(data) == len(theta), 'image and theta label should be equal'
    print('rotating the image based on the psi prior')
    data_new = np.zeros(np.shape(data))
    data = create_mask(data)
    for i in range(len(data)):
        img = data[i]
        img_rot = rotate_image(img, theta[i])
        data_new[i] = img_rot
    return data_new


def get_horizen_cylinder_mask(data, R=None):
    n, m = data.shape[1:]
    if R:
        radius = R
    else:
        radius = min(n, m) / 2
    center = int(n / 2)
    mask = np.zeros(np.shape(data))
    mask[:, center - R:center + R, :] = 1
    return mask