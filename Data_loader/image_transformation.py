import numpy as np
import cv2

def normalize_all_image(data, sigma = None):
    n_img,n,m = data.shape
    data_new = []
    mean = data.mean()
    std = data.std()
    print('mean and std are',mean,std)
    for i in range(n_img):
        img = data[i]
        img = (img - mean)/std
        data_new.append(img)
        if sigma is not None:
            img[img > sigma] = sigma
            img[img < -sigma] = -sigma
    data_new = np.array(data_new)
    return data_new

def norm_min_max(data):
    min = data.min()
    max = data.max()
    data = (data-min)/(max-min)
    return data

def crop(images, set_height, set_width):
    #images = normalize_filament(images, set_height, set_width,5)
    #print('cropping the image based on the given cylinder and center mask')
    img_dim = np.shape(images)[-1]
    assert (img_dim >= set_height) & (img_dim >= set_width)
    h1 = int((img_dim - set_height) / 2)
    h2 = int((img_dim + set_height) / 2)
    w1 = int((img_dim - set_width) / 2)
    w2 = int((img_dim + set_width) / 2)
    images = images[:, h1:h2, w1:w2]
    #images = cv2.normalize(images, None, 0, 1, cv2.NORM_MINMAX)
    return images

def add_noise_SNR(image, snr):
    b,x,y=np.shape(image)
    xpower = np.var(image)
    npower = xpower / snr
    print('noise sigma is: ',npower)
    images_new=image + np.random.normal(0,1,(b,x,y)) * np.sqrt(npower)
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

def padding(all_data_image, filament_index, length, set_mask=True):

    n_img, height, width = all_data_image.shape
    all_data_image = np.concatenate((all_data_image, np.zeros((1, height, width))), axis=0)

    # change the filament index accordingly
    max_len = max(map(len, filament_index))
    if length >= max_len:
        filament_index = filament_index
    elif length < max_len:
        filament_index, cut_index = cut_corpus(filament_index, length)

    n_filament = len(filament_index)
    output = np.zeros((n_filament, length, height, width),dtype=np.float32)
    mask = np.zeros((n_filament, length))
    print('There are {} number of filaments'.format(n_filament))

    for i in range(n_filament):
        if i % 100 ==0:
            print('creating {} filaments'.format(i))
        lst = filament_index[i]
        output[i, :len(lst), :, :] = all_data_image[lst]
        if set_mask is True:
            mask[i, :len(lst)] = 1
        else:
            mask[i, :] = 1
    output=output.astype('float32')
    #output = norm_min_max(output)
    #output = cv2.normalize(output, None, 0, 1, cv2.NORM_MINMAX)
    return output, mask

def padding_vector(vector, filament_index, length, set_mask=True):

    n_img, dim =vector.shape
    max_len = max(map(len, filament_index))
    vector = np.concatenate((vector, np.zeros((1, dim))), axis=0)

    # change the filament index accordingly
    if length >= max_len:
        filament_index = filament_index
    elif length < max_len:
        filament_index, cut_index = cut_corpus(filament_index, length)

    n_filament = len(filament_index)
    output = np.zeros((n_filament, length ,dim))
    mask = np.zeros((n_filament, length))
    for i in range(n_filament):
        lst = np.array(filament_index[i])
        lst_ones = np.ones(length)* n_img
        lst_ones[:len(lst)] = lst
        output[i, :len(lst), :] = vector[lst]
        if set_mask is True:
            mask[i, (lst_ones < n_img)] = 1
        else:
            mask[i, :] = 1
    output=output.astype('float32')
    return output, mask

def padding_lazy(lazy_image, filament_index, length, set_mask=True):
    vector = lazy_image.copy()
    n_img = len(lazy_image)
    max_len = max(map(len, filament_index))
    vector.append(None)
    vector = np.array(vector)

    # change the filament index accordingly
    if length >= max_len:
        filament_index = filament_index
    elif length < max_len:
        filament_index, cut_index = cut_corpus(filament_index, length)

    n_filament = len(filament_index)
    output = np.empty((n_filament, length), dtype=object)
    mask = np.zeros((n_filament, length))
    for i in range(n_filament):
        lst = np.array(filament_index[i])
        lst_ones = np.ones(length) * n_img
        lst_ones[:len(lst)] = lst
        output[i, :len(lst)] = vector[lst]
        if set_mask is True:
            mask[i, (lst_ones < n_img)] = 1
        else:
            mask[i, :] = 1
    return output, mask


def defocus_filament(defocus, filament_index, length):
    #print('checking ctf parameter',defocus[0])
    n_img,n_parameters  = defocus.shape
    max_len = max(map(len, filament_index))
    defocus = np.concatenate((defocus, np.zeros((1, n_parameters))), axis=0)

    # change the filament index accordingly
    if length >= max_len:
        filament_index = filament_index
    elif length < max_len:
        filament_index, cut_index = cut_corpus(filament_index, length)

    n_filament = len(filament_index)
    output = []
    for i in range(n_filament):
        lst = np.array(filament_index[i])
        lst_ones = np.ones(length)* n_img
        lst_ones[:len(lst)] = lst
        output.append(defocus[lst].astype('float32'))
    output = np.array(output, dtype = object)
    return output

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

def get_filament_mask(data, height, width, only_cylinder = False):
    n, m = data.shape[1:]
    y_grid, x_grid = np.ogrid[:n, :m]
    center = np.array([n / 2, m / 2])
    mask = (height/2 <= np.abs(center[0]-y_grid)) & (width/2 <= np.abs(center[1]-x_grid))
    background = (height/2 > np.abs(center[0]-y_grid)) & (width/2 > np.abs(center[1]-x_grid))
    if only_cylinder is True:
        mask = (height/2 <= np.abs(center[0]-y_grid)) & (m/2 <= np.abs(center[1]-x_grid))
        background = (height/2 > np.abs(center[0]-y_grid)) & (m/2 <= np.abs(center[1]-x_grid))
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
    #print('normalizing the image')
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

def normalize_filament(data,height,width,sigma=None):
    #print('normalizing the filament')
    mask, background = get_filament_mask(data,height,width)
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
    if len(data) == 1:
        theta = [theta]
    assert len(data) == len(theta), 'image and theta label should be equal'
    #print('rotating the image based on the psi prior')
    data_new = np.zeros(np.shape(data))
    data = create_mask(data)
    for i in range(len(data)):
        img = data[i]
        img_rot = rotate_image(img, float(theta[i]))
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