import skimage.external.tifffile as tiff
import numpy as np
from skimage.filters import threshold_otsu, gaussian, sobel
from skimage.morphology import watershed
from skimage import measure
from scipy import ndimage as ndi


def labelNuclei(img):
    with tiff.TiffFile as tif:
        images = tif.asarray().astype(float)

    img_gauss = gaussian(images[0], sigma=3)
    thresh = threshold_otsu(img_gauss)

    markers = np.zeros_like(img_gauss)
    markers[img_gauss < thresh] = 1
    markers[img_gauss > thresh] = 2
    elevation_map = sobel(img_gauss)

    segmentation = watershed(elevation_map, markers)
    segmentation = ndi.binary_fill_holes(segmentation - 1)
    labeled_nuclei, _ = ndi.label(segmentation)

    return labeled_nuclei

