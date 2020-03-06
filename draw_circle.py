import numpy as np
import skimage.external.tifffile as tiff
from skimage.filters import threshold_otsu, gaussian, sobel
from skimage.morphology import watershed
from skimage import measure, draw
from scipy import ndimage as ndi
from scipy import optimize
import matplotlib.pyplot as plt
from intersection import calcIntersections

testfile = "img/test_dapi_gm130_tgn46.tif"
#testfile = "img/test_balls.tif"
with tiff.TiffFile(testfile) as tif:
    images = tif.asarray()
    metadata = tif[0].tags

pixel_per_micron = metadata['x_resolution'].value[0] / metadata['x_resolution'].value[1]

img_gauss = gaussian(images[0], sigma=3)
thresh = threshold_otsu(img_gauss)

markers = np.zeros_like(img_gauss)
markers[img_gauss < thresh] = 1
markers[img_gauss > thresh] = 2
elevation_map = sobel(img_gauss)
segmentation = watershed(elevation_map, markers)
segmentation = ndi.binary_fill_holes(segmentation - 1)

labeled_nuclei, nr = ndi.label(segmentation)
coms = ndi.measurements.center_of_mass(labeled_nuclei, labeled_nuclei, range(1, nr+1))

dist = np.ones(labeled_nuclei.shape)
for i in coms:
    dist[int(i[0]), int(i[1])] = 0
dist = ndi.morphology.distance_transform_edt(dist)

r = 45 * pixel_per_micron

mask = dist < r

final_labels = np.zeros(labeled_nuclei.shape)
for i in coms:
    final_labels[int(i[0]), int(i[1])] = 1
final_labels, _ = ndi.label(final_labels)
final_segmentation = watershed(dist, final_labels, mask=mask)

print(final_segmentation)
# fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(nrows=2, ncols=2)
# ax1.imshow(images[0])
# ax2.imshow(final_segmentation)
# ax3.imshow(images[1])
# ax4.imshow(images[2])


# plt.show()