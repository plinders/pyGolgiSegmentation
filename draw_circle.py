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

labeled_nuclei, _ = ndi.label(segmentation)
regions = measure.regionprops(labeled_nuclei)
r = 45 * pixel_per_micron

fig, ax = plt.subplots()
perim = np.zeros((labeled_nuclei.shape[0], labeled_nuclei.shape[1]))
processed_regions = []
for props in regions:
    y0, x0 = props.centroid
    processed_regions.append(props.centroid)
    for props2 in processed_regions:
        if props2 == props.centroid:
            break
        elif type(props2) is not None:
            try:
                x3, y3, x4, y4 = calcIntersections(props.centroid, props2, r, r)
                if (x3, x4, y3, y4) is not None:
                    ax.plot(x3, y3, x4, y4, 'o', color='r')
            except TypeError:
                pass
        else:
            pass
    rr, cc = draw.circle(y0, x0, r, shape=labeled_nuclei.shape)
    perim[rr, cc] = props.label

ax.imshow(perim)


#ax.imshow(labeled_nuclei, cmap = "gray")
plt.show()

