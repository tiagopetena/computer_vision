import numpy as np
import cv2
from math import pi,e, sin, cos, radians

from matplotlib import pyplot as plt
from convolution import convolve

sobel_x = np.array(
    [-1,0,1,
     -2,0,2,
     -1,0,1]).reshape(3,3) 
sobel_y = np.array(
    [ 1, 2, 1,
      0, 0, 0,
     -1,-2,-1]).reshape(3,3) 

def get_gaussian_kernel(sigma, size, show=False):
    """Returns a gaussian kernel at a scale of [sigma].

    Args:
        sigma (int): Gaussian scale. 
        show (bool): show plot of the kernel. Defaults to False.

    Returns:
        2d np.array: returns a gaussian kernel.
    """

    assert size%2 != 0, "sigma must be an odd number"

    kernel = np.zeros((size, size)) 
    limit = (size)//2
    
    for x in range(-limit, limit+1):
        for y in range(-limit, limit+1): 
            kernel[x+limit,y+limit] = (1/(2*pi*sigma**2))*e**(-1*(x**2+y**2)/(2*sigma**2))

    if show==True:
        print(kernel)
        plt.imshow(kernel, interpolation='none')
        plt.show()

    return kernel


def partial_derivative(M, theta, show=False):
    """ Apply a partial derivative over M at the direction of 
    a vector at theta degrees. This is done using sobel filters
    convolutions.

    Args:
        M (2d np.array): matrix to apply derivative.
        theta (float): partial derivative direction(in degrees).
        show (bool, optional): show plot of the kernel.. Defaults to False.

    Returns:
        2d np.array: 1st derivative of M
    """

    theta_rad = radians(theta) 

    dMdx = convolve(M, sobel_x) 
    dMdy = convolve(M, sobel_y) 

    dM = sin(theta_rad)*dMdy + cos(theta_rad)*dMdx

    if show == True:
        print(dM)
        plt.imshow(dM, interpolation='none')
        plt.show()

    return dM


def partial_2nd_derivative(M, theta):
    return partial_derivative(partial_derivative(M, theta), theta)


def crop(X, crop_size=2):
    return X[crop_size:X.shape[0]-crop_size, crop_size:X.shape[1]-crop_size]

def get_center_surrounded(gauss, sigma, ratio):

    center_surrounded = gauss - get_gaussian_kernel(ratio*sigma, gauss.shape[0]) 
 
    return center_surrounded

def get_filter_bank(scales,kernel_size, show=False):
    """[summary]

    Args:
        scales (Int list): list containing the scales(sigma) in the filter bank
        kernel_size ([type]): size of the filter(kernel) kernel_size*kernel_size. must be an odd number.
        show (bool, optional): [description]. Defaults to False.

    Returns:
        [type]: A list containing dictionaries of the filters in each scale.
    """


    filter_bank = []

    for sigma in scales:  

        gauss = get_gaussian_kernel(sigma, size=kernel_size+4)

        # odd/sobel filters
        odd_0 = {"filter":crop(partial_derivative(gauss, 0)), "orientation":0}
        odd_30 = {"filter":crop(partial_derivative(gauss, 30)), "orientation":30}
        odd_60 = {"filter":crop(partial_derivative(gauss, 60)), "orientation":60}
        odd_90 = {"filter":crop(partial_derivative(gauss, 90)), "orientation":90}
        odd_120 = {"filter":crop(partial_derivative(gauss, 120)), "orientation":120}
        odd_150 = {"filter":crop(partial_derivative(gauss, 150)), "orientation":150}

        odd_filters  = [odd_0, odd_30, odd_60, odd_90, odd_120, odd_150]

        # even/laplacian filters
        even_0 = {"filter":crop(partial_2nd_derivative(gauss, 0)), "orientation":0}
        even_30 = {"filter":crop(partial_2nd_derivative(gauss, 30)), "orientation":30}
        even_60 = {"filter":crop(partial_2nd_derivative(gauss, 60)), "orientation":60}
        even_90 = {"filter":crop(partial_2nd_derivative(gauss, 90)), "orientation":90}
        even_120 = {"filter":crop(partial_2nd_derivative(gauss, 120)), "orientation":120}
        even_150 = {"filter":crop(partial_2nd_derivative(gauss, 150)), "orientation":150}
        
        even_filters = [even_0, even_30, even_60, even_90, even_120, even_150]
 
        center_surrounded = crop(get_center_surrounded(gauss, sigma, ratio=2))

        this_scale = {"scale": sigma,
                      "odd":odd_filters,
                      "even":even_filters,
                      "center_surrounded":center_surrounded}

        filter_bank.append(this_scale)

        if show == True:
            x1 = np.concatenate((odd_0['filter'], odd_30['filter'], odd_60['filter']), axis=1)
            x2 = np.concatenate((odd_90['filter'], odd_120['filter'], odd_150['filter']), axis=1) 
            x3 = np.concatenate((even_0['filter'], even_30['filter'], even_60['filter']), axis=1)
            x4 = np.concatenate((even_90['filter'], even_120['filter'], even_150['filter']), axis=1) 
            x5 = np.concatenate((center_surrounded, np.zeros_like(center_surrounded), np.zeros_like(center_surrounded)), axis=1) 
            plot = np.concatenate((x1,x2,x3,x4,x5), axis=0)

            plt.imshow(plot, interpolation='none')
            plt.show()

    return filter_bank


def oriented_energy(img, odd, even, filter_bank):

    oes = []
 
    for scale in filter_bank:  
        odds = [convolve(img, f['filter'], padding=cv2.BORDER_REFLECT)**2 for f in scale['odd']]
        even = [convolve(img, f['filter'], padding=cv2.BORDER_REFLECT)**2 for f in scale['even']]
        oe = [o + e for (o,e) in zip(odds,even)]
        oes.append(oe)

    return oes

def filter_img(img, filter_bank):

    odds = []
    evens = []
    oe = []
    center_surr = []

    for i, scale in enumerate(filter_bank):
        print(f'[scale {i}]')
        print(f'odd filtering...')
        odds.append(np.array([convolve(img, f['filter'], padding=cv2.BORDER_REFLECT).flatten() for f in scale['odd']]))
        print(f'even filtering...')
        evens.append(np.array([convolve(img, f['filter'], padding=cv2.BORDER_REFLECT).flatten() for f in scale['even']]))
        print(f'oriented energy...')
        oe.append(np.array([np.array([o**2 for o in odd]) + np.array([e**2 for e in even]) for (odd,even) in zip(odds[i],evens[i])]))
        print(f'center surrounded filtering...\n')
        center_surr.append(np.array(convolve(img, scale['center_surrounded'], padding=cv2.BORDER_REFLECT).flatten()))

    return odds, evens, oe, center_surr