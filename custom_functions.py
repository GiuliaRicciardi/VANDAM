import functools
from typing import Union
from pathlib import Path

import multiprocess as mp
import numpy as np
from astropy.io import fits
from scipy import optimize


def roll_2d(data: np.ndarray, dx: int, dy: int, num=np.NaN):
    """
    Shift the element of a 2D array. The missing data are replaced by num.

    Args:
        data: 2d-array
            data to shift
        dx: int
            x shift
        dy: int
            y shift
        num: float
            what replaces the data

    Returns:
        shifted_data: 2d array
            shifted array

    """

    # roll sposta tutti gli elementi di una matrice lungo una direzione (quelli che superano il bordo sbucano dall'altra
    # parte)
    shifted_data = np.roll(data, dx, axis=1)

    # questi if mettono NaN al posto degli elementi che sbucano dall'altra parte
    if dx < 0:
        shifted_data[:, dx:] = num
    elif dx > 0:
        shifted_data[:, 0:dx] = num

    shifted_data = np.roll(shifted_data, dy, axis=0)
    if dy < 0:
        shifted_data[dy:, :] = num
    elif dy > 0:
        shifted_data[0:dy, :] = num

    return shifted_data


def acf_distance(data: np.ndarray, dx: int, dy: int, beam_area_pix: int = None):
    """
    Computes the autocorrelation for pixels with a given displacement.

    Args:
        data: 2d-array
            noise image
        dx: int
            x displacement
        dy: int
            y displacement
        beam_area_pix: int, optional
            beam area in pixels. Needed for the std of the autocorrelation
    Returns:
        acf_mean: float
            mean autocorrelation for this displacement
        acf_std: float
            mean standard deviation.
    """

    # questa e una matrice con quanto ogni pixel e correlato con quello alla distanza data, poi prendiamo la media per
    # avere la correlazione media per questa distanza
    acf_img = data * roll_2d(data, dx, dy, num=np.NaN)

    if beam_area_pix is not None:
        return np.nanmean(acf_img), np.nanstd(acf_img) / np.sqrt(acf_img.size / beam_area_pix)
    return np.nanmean(acf_img), np.NaN


def acf(noise, beam_width_pix, beam_area_pix: int = None, procs: int = 8):
    """
    Computes the autocorrelation function.

    Args:
        noise: 2d-array
        beam_width_pix: int
        beam_area_pix: int, opt
        procs: int, opt
            number of processes to speed up the calculation

    Returns:
        acf_image 3d-array
            images of the acf and its std
    """

    # creo un array con due immagini vuote dove salvare l'acf e la std
    acf_img = np.zeros([int(2 * beam_width_pix + 1), int(2 * beam_width_pix + 1), 2])

    # creo una matrice con tutte le distanze rispetto rispetto al centro, in pixel
    iacf, jacf = np.meshgrid(
        [i for i in range(-int((acf_img.shape[1] - 1) / 2) + 1, int((acf_img.shape[1] - 1) / 2) + 1)],
        [i for i in range(-int((acf_img.shape[1] - 1) / 2) + 1, int((acf_img.shape[1] - 1) / 2) + 1)], indexing='ij')
    # trasformo queste coordinate in modo di avere una lista di coppie [[0, 0], [0, 1], [0, 2], ...]
    coord_list = np.column_stack((iacf.ravel(), jacf.ravel()))

    # quante parallelizzazioni fare
    p = mp.Pool(procs)

    # applico la funzione acf_distance a tutte le distanze
    # p.starmap serve ad  applicare ad una funzione (acf_distance) un set di parametri (coord_list)
    # functools.partial serve a dire alla funzioone acf_distance di usare sempre lo stesso noise la stessa beam_area_pix
    result = np.array(list(p.starmap(functools.partial(acf_distance, noise, beam_area_pix=beam_area_pix), coord_list)))

    # a sinistra abbiamo una lista di 2 pixel (per l'acf e la acf_std) per ogni coordinata, a destra i risultati
    # calcolati per quei pixel per le rispettive coodinate
    acf_img[int((acf_img.shape[0] - 1) / 2) + iacf.ravel(), int((acf_img.shape[0] - 1) / 2) + jacf.ravel(), :] = result

    return acf_img


def gaussian(height, center_x, center_y, width_x, width_y):
    """Returns a gaussian function with the given parameters"""
    width_x = float(width_x)
    width_y = float(width_y)
    return lambda x, y: height * np.exp(
        -(((center_x - x) / width_x) ** 2 + ((center_y - y) / width_y) ** 2) / 2)


def moments(data):
    """Returns (height, x, y, width_x, width_y)
    the gaussian parameters of a 2D distribution by calculating its
    moments """
    total = data.sum()
    X, Y = np.indices(data.shape)
    x = (X * data).sum() / total
    y = (Y * data).sum() / total
    col = data[:, int(y)]
    width_x = np.sqrt(np.abs((np.arange(col.size) - x) ** 2 * col).sum() / col.sum())
    row = data[int(x), :]
    width_y = np.sqrt(np.abs((np.arange(row.size) - y) ** 2 * row).sum() / row.sum())
    height = data.max()
    return height, x, y, width_x, width_y


def fitgaussian(data, x0: list):
    """Returns (height, x, y, width_x, width_y)
    the gaussian parameters of a 2D distribution found by a fit"""

    # params = moments(data)
    errorfunction = lambda p: np.ravel(gaussian(*p)(*np.indices(data.shape)) -
                                       data)

    p, success = optimize.leastsq(errorfunction, x0)
    return p


def subtract_gaussians(data: np.ndarray, params: list, factor: float = 3):
    """
    Subract a gaussian for each of the parameter listed.

    Args:
        data: 2d-array
        params: list
            parameters of the gaussians
        factor: float
            how many std to cut around the sources

    Returns:
        data_no_sources: 2d-array
    """

    data_no_sources = np.copy(data)
    for _param in params:
        x_edges = int(_param[1] - 3 * _param[3]), int(_param[1] + 3 * _param[3])
        y_edges = int(_param[2] - 3 * _param[4]), int(_param[2] + 3 * _param[4])
        xx, yy = np.mgrid[x_edges[0]:x_edges[1], y_edges[0]:y_edges[1]]

        data_no_sources[xx, yy] = data_no_sources[xx, yy] - gaussian(*_param)(xx, yy)

    return data_no_sources


def acf_variance(mask: np.array, acf_img: np.array):
    """

    Args:
        mask: source_with_nans * 0
        acf_img:

    Returns:

    """

    # iacf, jacf = np.meshgrid(
    #     [i for i in range(-int((mask.shape[1] - 1) / 2) + 1, int((mask.shape[1] - 1) / 2) + 1)],
    #     [i for i in range(-int((mask.shape[1] - 1) / 2) + 1, int((mask.shape[1] - 1) / 2) + 1)], indexing='ij')

    # più propriamente, varianza + covarianza
    variance = acf_img.sum() * mask.sum()  # acf [0, 0], la varianza

    # variance = acf_img[int((len(acf_img) - 1) / 2), int(
    #     (len(acf_img) - 1) / 2)] * mask.sum()  # acf [0, 0], la varianza
    # covariance = acf_img[iacf.ravel() + int((len(acf_img) - 1) / 2), jacf.ravel() + int(
    #     (len(acf_img) - 1) / 2)].sum()  # la covarianza,  tutti i termini i != j rispetto al centro

    return variance


def create_circular_mask(data, center: tuple = None, r_edges: tuple = None):
    if center is None:
        center = (int((data.shape[0] - 1) / 2) + 1, int((data.shape[1] - 1) / 2) + 1)
    if r_edges is None:
        r_edges = (0, min(center[0], center[1], data.shape[0] - center[0], data.shape[1] - center[1]))

    Y, X = np.ogrid[:data.shape[0], :data.shape[1]]
    dist_from_center = np.sqrt((X - center[0]) ** 2 + (Y - center[1]) ** 2)

    mask = dist_from_center <= r_edges[1]
    mask[np.asarray(dist_from_center <= r_edges[0]).nonzero()] = False
    # mask[np.asarray(dist_from_center <= r_edges[0]).nonzero()] = np.NaN
    # mask = dist_from_center >= r_edges[0]
    return mask


def read_data(fpath: Union[str, Path]):
    fpath = Path(fpath)
    if fpath.suffix != '.fits':
        raise TypeError("The file must be in fits format.")
    with fits.open(fpath) as hdul:
        fits_file = hdul[0]
        data = fits_file.data[0][0]
    return data


def iterate_files(root: Union[str, Path]):
    root = Path(root)
    if not root.is_dir():
        raise TypeError("The root must be a directory.")
    data_list = []
    for _path in root.glob("*.fits"):
        data_list.append(read_data(_path))
    return data_list


def variance_nimages(root: Union[str, Path]):
    data_list = iterate_files(root)
    return np.nanstd(data_list, 0)
