import skimage.external.tifffile as tiff
import numpy as np
from skimage.filters import threshold_otsu, gaussian, sobel
from skimage.morphology import watershed
from skimage import measure
from scipy import ndimage as ndi


def importImage(img, ppm=None):
    if ppm:
        with tiff.TiffFile(img) as tif:
            images = tif.asarray().astype(float)

        return images, ppm
    else: 
        with tiff.TiffFile(img) as tif:
            images = tif.asarray().astype(float)
            metadata = tif[0].tags

        pixel_per_micron = metadata['x_resolution'].value[0] / metadata['x_resolution'].value[1]

        return images, pixel_per_micron


def labelNuclei(nucleusImage):
    img_gauss = gaussian(nucleusImage, sigma=3)
    thresh = threshold_otsu(img_gauss)

    markers = np.zeros_like(img_gauss)
    markers[img_gauss < thresh] = 1
    markers[img_gauss > thresh] = 2
    elevation_map = sobel(img_gauss)

    segmentation = watershed(elevation_map, markers)
    segmentation = ndi.binary_fill_holes(segmentation - 1)
    labeled_nuclei, nr = ndi.label(segmentation)

    return labeled_nuclei, nr


def distanceSegmentation(labeled_nuclei, nr, radius, pixel_per_micron):
    coms = ndi.measurements.center_of_mass(labeled_nuclei, labeled_nuclei, range(1, nr + 1))   
    dist = np.ones(labeled_nuclei.shape)
    for i in coms:
        dist[int(i[0]), int(i[1])] = 0
    dist = ndi.morphology.distance_transform_edt(dist)

    r = radius * pixel_per_micron

    mask = dist < r
    final_labels = np.zeros(labeled_nuclei.shape)
    for i in coms:
        final_labels[int(i[0]), int(i[1])] = 1
    final_labels, _ = ndi.label(final_labels)
    final_segmentation = watershed(dist, final_labels, mask=mask)

    return final_segmentation


def segmentCells(img, nuclearChannel, radius, ppm=None):
    images, ppm = importImage(img, ppm)
    nuclei, nr = labelNuclei(images[nuclearChannel])
    return distanceSegmentation(nuclei, nr, radius, ppm)
