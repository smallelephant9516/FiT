import numpy as np
import cv2
import torch
import torchvision.transforms as T

# copy from cryoDRGN

def compute_ctf_np(freqs, dfu, dfv, dfang, volt, cs, w, phase_shift=0, bfactor=None, apix=1):
    '''
    Compute the 2D CTF

    Input:
        freqs (np.ndarray) Nx2 array of 2D spatial frequencies
        dfu (float): DefocusU (Angstrom)
        dfv (float): DefocusV (Angstrom)
        dfang (float): DefocusAngle (degrees)
        volt (float): accelerating voltage (kV)
        cs (float): spherical aberration (mm)
        w (float): amplitude contrast ratio
        phase_shift (float): degrees
        bfactor (float): envelope fcn B-factor (Angstrom^2)
    '''
    # convert units
    volt = volt * 1000
    cs = cs * 10 ** 7
    dfang = dfang * np.pi / 180
    phase_shift = phase_shift * np.pi / 180

    # lam = sqrt(h^2/(2*m*e*Vr)); Vr = V + (e/(2*m*c^2))*V^2
    lam = 12.2639 / np.sqrt(volt + 0.97845e-6 * volt ** 2)
    x = freqs[:, 0] / apix
    y = freqs[:, 1] / apix
    ang = np.arctan2(y, x)
    s2 = x ** 2 + y ** 2
    df = .5 * (dfu + dfv + (dfu - dfv) * np.cos(2 * (ang - dfang)))
    gamma = 2 * np.pi * (-.5 * df * lam * s2 + .25 * cs * lam ** 3 * s2 ** 2) - phase_shift
    ctf = np.sqrt(1 - w ** 2) * np.sin(gamma) - w * np.cos(gamma)
    if bfactor is not None:
        ctf *= np.exp(-bfactor / 4 * s2)
    return np.require(ctf, dtype=freqs.dtype)


def fft2_center(img):
    return np.fft.fftshift(np.fft.fft2(np.fft.fftshift(img, axes=(-1, -2))), axes=(-1, -2))


def fftn_center(img):
    return np.fft.fftshift(np.fft.fftn(np.fft.fftshift(img)))


def ifftn_center(V):
    V = np.fft.ifftshift(V)
    V = np.fft.ifftn(V)
    V = np.fft.ifftshift(V)
    return V


def ht2_center(img):
    f = fft2_center(img)
    return f.real - f.imag


def htn_center(img):
    f = np.fft.fftshift(np.fft.fftn(np.fft.fftshift(img)))
    return f.real - f.imag


def iht2_center(img):
    img = fft2_center(img)
    img /= (img.shape[-1] * img.shape[-2])
    return img.real - img.imag

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
    return mask

def get_circular_mask_single(data, R=None):
    n, m = data.shape[:]
    if R:
        radius = R
    else:
        radius = min(n, m) / 2
    y_grid, x_grid = np.ogrid[:n, :m]
    center = np.array([n / 2, m / 2])
    dist = np.sqrt((center[0] - y_grid) ** 2 + (center[1] - x_grid) ** 2)
    mask = dist <= radius
    return mask

def low_pass_filter(image, ang, apix=1):
    D, _ = np.shape(image)
    R_pix = int(np.ceil(D / (ang / apix)))
    image_fft = ht2_center(image)
    mask = get_circular_mask_single(image_fft, R=R_pix)
    new_image_fft = np.zeros((D, D))
    new_image_fft[mask] = image_fft[mask]
    return iht2_center(new_image_fft)

def compute_ctf_first(freqs, dfu, dfv, dfang, volt, cs, w, phase_shift=0, bfactor=None, apix=1):
    '''
    Compute the 2D CTF

    Input:
        freqs (np.ndarray) Nx2 array of 2D spatial frequencies
        dfu (float): DefocusU (Angstrom)
        dfv (float): DefocusV (Angstrom)
        dfang (float): DefocusAngle (degrees)
        volt (float): accelerating voltage (kV)
        cs (float): spherical aberration (mm)
        w (float): amplitude contrast ratio
        phase_shift (float): degrees
        bfactor (float): envelope fcn B-factor (Angstrom^2)
    '''
    # convert units
    volt = volt * 1000
    cs = cs * 10 ** 7
    dfang = dfang * np.pi / 180
    phase_shift = phase_shift * np.pi / 180

    # lam = sqrt(h^2/(2*m*e*Vr)); Vr = V + (e/(2*m*c^2))*V^2
    lam = 12.2639 / np.sqrt(volt + 0.97845e-6 * volt ** 2)
    x = freqs[:, 0] / apix
    y = freqs[:, 1] / apix
    ang = np.arctan2(y, x)
    s2 = x ** 2 + y ** 2
    df = .5 * (dfu + dfv + (dfu - dfv) * np.cos(2 * (ang - dfang)))
    gamma = 2 * np.pi * (-.5 * df * lam * s2 + .25 * cs * lam ** 3 * s2 ** 2) - phase_shift
    ctf = np.sqrt(1 - w ** 2) * np.sin(gamma) - w * np.cos(gamma)
    df_1D = .5 * (dfu + dfv)
    a = 0.5 * df_1D * lam
    b = -.25 * cs * lam ** 3
    c = 1 - phase_shift / (np.pi / 2)
    s2_peak = (-b - np.sqrt(b * b - 4 * a * c)) / (2 * a)
    s2_peak_pix = int(np.ceil(256 * s2_peak / 0.5))
    if bfactor is not None:
        ctf *= np.exp(-bfactor / 4 * s2)
    D = int(np.sqrt(len(ctf)))
    ctf = ctf.reshape(D, D)
    mask = get_circular_mask_single(ctf, R=s2_peak_pix + 2)
    ctf_out = -np.ones(np.shape(ctf))
    ctf_out[mask] = ctf[mask]
    return np.require(ctf_out, dtype=freqs.dtype)


def ctf_correction(all_data_image, dataframe, Apix, mode='first'):
    dfU = dataframe['_rlnDefocusU'].astype('float32')
    dfV = dataframe['_rlnDefocusV'].astype('float32')
    dfang = dataframe['_rlnDefocusAngle'].astype('float32')
    extent = 0.5
    image_number, D, _ = np.shape(all_data_image)
    x0, x1 = np.meshgrid(np.linspace(-extent, extent, D, endpoint=True),
                         np.linspace(-extent, extent, D, endpoint=True))
    coords = np.stack([x0.ravel(), x1.ravel()], 1).astype(np.float32)
    all_image_pf = np.zeros(np.shape(all_data_image))
    print('doing CTF correction with the {} mode'.format(mode))
    if mode == 'phase flip':
        for i in range(len(all_data_image)):
            image = all_data_image[i]
            image_fft = ht2_center(image).astype(np.float32)
            ctf = compute_ctf_np(coords, dfU[i], dfV[i], dfang[i], 300, 2.7, 0.1, apix=Apix)
            ctf = ctf.reshape((D, D))
            image_flip = -image_fft * np.sign(ctf)
            image_pf = iht2_center(image_flip)
            all_image_pf[i] = image_pf
    elif mode == 'first':
        for i in range(len(all_data_image)):
            image = all_data_image[i]
            image_fft = ht2_center(image).astype(np.float32)
            ctf = compute_ctf_first(coords, dfU[i], dfV[i], dfang[i], 300, 2.7, 0.1, apix=Apix, bfactor=100)
            ctf = ctf.reshape((D, D))
            image_flip = -image_fft / ctf
            image_pf = iht2_center(image_flip)
            all_image_pf[i] = image_pf
    return all_image_pf

def low_pass_filter_images(images, ang, apix=1):
    print('doing low pass filter on the images')
    lowpass_images=np.zeros(images.shape)
    for i in range(len(images)):
        lowpass_images[i] = low_pass_filter(images[i],ang,apix)
    return lowpass_images

