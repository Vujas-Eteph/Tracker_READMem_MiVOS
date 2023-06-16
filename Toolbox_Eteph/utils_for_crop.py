import numpy as np
from icecream import ic
import cv2


def _compute_new_lengths_for_x_N_y(y:float,x:float,coeff=5):
    # calculate new y and x
    default_area = y*x
    ratio = y/x
    new_area = coeff*default_area
    new_x = np.sqrt(new_area/ratio)
    new_y = ratio*new_x

    return new_y, new_x


def _get_central_position(bin_mask:np.array):
    #return y and x coords for the centroid
    mu = cv2.moments(bin_mask, binaryImage = True)

    return (mu['m10'] / (mu['m00'] + 1e-5), mu['m01'] / (mu['m00'] + 1e-5))[::-1]


if '__main__' == __name__:
    y = 10
    x = 5
    y_n, x_n = _compute_new_lengths_for_x_N_y(y,x)
    ic(y_n)
    ic(x_n)
    ic(y_n*x_n)

    bin_mask = np.array([[0,0,0,0,0],
                         [0,0,1,1,0],
                         [0,0,1,1,0],
                         [1,0,1,0,0],
                         [0,1,1,0,0]])

    mu = _get_central_position(bin_mask)
    ic(mu)


