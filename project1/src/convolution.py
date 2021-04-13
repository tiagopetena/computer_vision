import cv2
import numpy as np
 
def convolve(image, kernel, padding=cv2.BORDER_CONSTANT): 
    """ image*kernel
    Args:
        image (2D np.array): matrix to be convoluted
        kernel (2D np.array): convolution kernel/filter

    Returns:
        2D np array: convolution matrix resultant
    """
    
    # kernel flip
    kernel_flipped = np.flipud(np.fliplr(kernel))
    

    # get image and kernel dimensions
    image_height, image_width = image.shape[:2] 
    kernel_height, kernel_width = kernel_flipped.shape[:2]

    output = np.zeros((image_height, image_width))
    pad = kernel_width // 2 

    # padding
    image_padded = cv2.copyMakeBorder(image, pad, pad, pad, pad, padding) 
     
    for i in range(image_height):
        for j in range(image_width):   
            output[i,j] = (kernel_flipped * image_padded[i: i+kernel_height, j: j+kernel_width]).sum() 
    return output
