
from itertools import combinations_with_replacement
from ofast import *
from scipy import ndimage as ndi
from skimage._shared.utils import convert_to_float
from skimage.transform import resize
import math
import numpy as np
import os

this_dir = os.path.dirname(__file__)
POS = np.loadtxt(os.path.join(this_dir, "orb_descriptor_positions.txt"),
                 dtype=np.int8)
POS0 = np.ascontiguousarray(POS[:, :2])
POS1 = np.ascontiguousarray(POS[:, 2:])


def _orb_loop(image, keypoints, orientations):

    descriptors = np.zeros((keypoints.shape[0], POS.shape[0]), dtype=np.uint8)
    for i in range(descriptors.shape[0]):
        angle = orientations[i]
        sin_a = math.sin(angle)
        cos_a = math.cos(angle)

        kr = keypoints[i, 1]
        kc = keypoints[i, 0]

        for j in range(descriptors.shape[1]):
            pr0 = POS0[j, 0]
            pc0 = POS0[j, 1]
            pr1 = POS1[j, 0]
            pc1 = POS1[j, 1]

            spr0 = (sin_a * pr0 + cos_a * pc0)
            spc0 = (cos_a * pr0 - sin_a * pc0)
            spr1 = (sin_a * pr1 + cos_a * pc1)
            spc1 = (cos_a * pr1 - sin_a * pc1)
            if(kr + int(spr0) < image.shape[0] and kr + int(spr1) < image.shape[0] and kc + int(spc0) < image.shape[1] and kc + int(spc1) < image.shape[1]):
                if image[kr + int(spr0), kc + int(spc0)] < image[kr + int(spr1), kc + int(spc1)]:
                    descriptors[i, j] = True

    return np.asarray(descriptors)

def _smooth(image, sigma, mode, cval):
    """Return image smoothed by the Gaussian filter."""
    smoothed = np.empty_like(image)
    ndi.gaussian_filter(image, sigma, output=smoothed,
                        mode=mode, cval=cval)
    return smoothed

def pyramid_reduce(image, downscale=2, sigma=None, order=1,mode='reflect', cval=0, preserve_range=False):
    image = convert_to_float(image, preserve_range)
    out_shape = tuple([math.ceil(d / float(downscale)) for d in image.shape])
    if sigma is None:
        sigma = 2 * downscale / 6.0

    smoothed = _smooth(image, sigma, mode, cval)
    out = resize(smoothed, out_shape, order=order, mode=mode, cval=cval,
                 anti_aliasing=False)

    return out

def pyramid_gaussian(image, max_layer=-1, downscale=2, sigma=None, order=1,mode='reflect', cval=0,preserve_range=False):
    """Yield images of the Gaussian pyramid formed by the input image."""
    image = convert_to_float(image, preserve_range)

    layer = 0
    current_shape = image.shape

    prev_layer_image = image
    yield image

    # build downsampled images until max_layer is reached or downscale process
    while layer != max_layer:
        layer += 1

        layer_image = pyramid_reduce(prev_layer_image, downscale, sigma, order,
                                     mode, cval)

        prev_shape = np.asarray(current_shape)
        prev_layer_image = layer_image
        current_shape = np.asarray(layer_image.shape)

        if np.all(current_shape == prev_shape):
            break

        yield layer_image

def _prepare_grayscale_input_nD(image):
    image = np.squeeze(image)

    return image.astype(float)


class ORB():
    
    def __init__(self, downscale=1.2, n_scales=8,
                 n_keypoints=100, fast_n=9, fast_threshold=0.2,
                 harris_k=0.04):
        self.downscale = downscale
        self.n_scales = n_scales
        self.n_keypoints = n_keypoints
        self.fast_n = fast_n
        self.fast_threshold = fast_threshold
        self.harris_k = harris_k

        self.keypoints = None
        self.scales = None
        self.orientations = None
        self.descriptors = None

    def _build_pyramid(self, image):
        image = _prepare_grayscale_input_nD(image)
        return list(pyramid_gaussian(image, self.n_scales - 1,self.downscale))


    def _detect_octave(self, octave_image):
        dtype = octave_image.dtype
        # Extract keypoints for current octave
        fast = Fast(self.fast_threshold,octave_image)

        fast.detect()

        keypoints = fast.keypoints
        if len(keypoints) == 0:
            return (np.zeros((0, 2), dtype=dtype),
                    np.zeros((0, ), dtype=dtype),)
        orientations = fast.theta

        return keypoints, orientations

    def detect(self, image):
        """Detect oriented FAST keypoints along with the corresponding scale.
        Parameters
        """

        pyramid = self._build_pyramid(image)

        keypoints_list = []
        orientations_list = []
        scales_list = []

        for octave in range(len(pyramid)):

            octave_image = np.ascontiguousarray(pyramid[octave])

            keypoints, orientations = self._detect_octave(
                octave_image)

            keypoints_list.append(keypoints * self.downscale ** octave)
            orientations_list.append(orientations)
            scales_list.append(np.full(
                keypoints.shape[0], self.downscale ** octave,
                dtype=octave_image.dtype))

        keypoints = np.vstack(keypoints_list)
        orientations = np.hstack(orientations_list)
        scales = np.hstack(scales_list)


        self.keypoints = keypoints
        self.scales = scales
        self.orientations = orientations
  
    def _extract_octave(self, octave_image, keypoints, orientations):

        keypoints = np.array(keypoints, dtype=np.intp, order='C',
                             copy=False)
        orientations = np.array(orientations, order='C',
                                copy=False)

        descriptors = _orb_loop(octave_image, keypoints, orientations)

        return descriptors

    def extract(self, image, keypoints, scales, orientations):
        """Extract rBRIEF binary descriptors for given keypoints in image.
        """

        pyramid = self._build_pyramid(image)

        descriptors_list = []
        mask_list = []

        # Determine octaves from scales
        octaves = (np.log(scales) / np.log(self.downscale)).astype(np.intp)

        for octave in range(len(pyramid)):

            # Mask for all keypoints in current octave
            octave_mask = octaves == octave

            if np.sum(octave_mask) > 0:

                octave_image = np.ascontiguousarray(pyramid[octave])

                octave_keypoints = keypoints[octave_mask]
                octave_keypoints /= self.downscale ** octave
                octave_orientations = orientations[octave_mask]

                descriptors= self._extract_octave(octave_image,
                                                         octave_keypoints,
                                                         octave_orientations)

                descriptors_list.append(descriptors)


        self.descriptors = np.vstack(descriptors_list).view(np.bool)
        self.mask_ = np.hstack(mask_list)

    def detect_and_extract(self, image):
        """Detect oriented FAST keypoints and extract rBRIEF descriptors.
        """

        pyramid = self._build_pyramid(image)

        keypoints_list = []
        orientations_list = []
        descriptors_list = []
        scaled_keypoints = []

        for octave in range(len(pyramid)):

            octave_image = np.ascontiguousarray(pyramid[octave])

            keypoints, orientations = self._detect_octave(
                octave_image)

            if len(keypoints) == 0:
                keypoints_list.append(keypoints)
                descriptors_list.append(np.zeros((0, 256), dtype=np.bool))
                continue

            descriptors= self._extract_octave(octave_image, keypoints,
                                                     orientations)
            
            for x in keypoints:
                    scaled_keypoints.append((int(x[0]* self.downscale ** octave),int(x[1]* self.downscale ** octave)))
            keypoints_list.append(keypoints)
            orientations_list.append(orientations)
            descriptors_list.append(descriptors)

        keypoints = np.vstack(scaled_keypoints)
        orientations = np.hstack(orientations_list)
        descriptors = np.vstack(descriptors_list).view(np.bool)

        self.keypoints = keypoints
        self.orientations = orientations
        self.descriptors = descriptors

