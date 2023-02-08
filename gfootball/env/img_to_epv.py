import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import scipy.cluster.vq as scv
from PIL import Image
import cv2
from gfootball.env import Metrica_PitchControl as mpc

def colormap2arr(arr, cmap):
    # http://stackoverflow.com/questions/3720840/how-to-reverse-color-map-image-to-scalar-values/3722674#3722674
    gradient = cmap(np.linspace(0.0, 1.0, 100))

    # Reshape arr to something like (240*240, 4), all the 4-tuples in a long list...
    arr2 = arr.reshape((arr.shape[0] * arr.shape[1], arr.shape[2]))

    # Use vector quantization to shift the values in arr2 to the nearest point in
    # the code book (gradient).
    code, dist = scv.vq(arr2, gradient)

    # code is an array of length arr2 (240*240), holding the code book index for
    # each observation. (arr2 are the "observations".)
    # Scale the values so they are from 0 to 1.
    values = code.astype('float') / gradient.shape[0]

    # Reshape values back to (240,240)
    values = values.reshape(arr.shape[0], arr.shape[1])
    values = values[::-1]
    return values

def get_EPV_at_location(x, y, EPV, field_dimen=(96., 64.)):
    # EPV = np.fliplr(EPV)
    # print('----', EPV.shape)
    ny, nx = EPV.shape
    dx = field_dimen[0] / float(nx)
    dy = field_dimen[1] / float(ny)
    ix = (x + field_dimen[0] / 2. - 0.0001) / dx
    iy = (y + field_dimen[1] / 2. - 0.0001) / dy
    # print('---x', x)
    # print('---y', y)
    # print('---ix', ix)
    # print('---iy', iy)
    return EPV[int(iy), int(ix)]


def calculate_epv(PPCF, x, y, EPV):
    # print('---x', int(x)+48)
    # print('---y', int(y)+48)
    Patt_target = PPCF[int(x)+48, int(y)+32]

    # EPV at end location
    EPV_target = get_EPV_at_location(x, y, EPV)

    EEPV_target = Patt_target * EPV_target

    return EEPV_target


def midpoint_double1(f, PPCF, a, b, c, d, nx, ny, EPV):
    hx = (b - a) / float(nx)
    hy = (d - c) / float(ny)
    I = 0
    for i in range(nx):
        for j in range(ny):
            xi = a + hx / 2 + i * hx
            yj = c + hy / 2 + j * hy
            I += hx * hy * f(PPCF, xi, yj, EPV)
            # print(hx * hy * f(PPCF, xi, yj, EPV))
            # print(i)
            # print(j)
    return I


# values = []
# for i in range(3):
#     arr = Image.open('/home/aarongu/Desktop/generation' + str(i) + '.png')
#     arr = np.array(arr) / 255

#     value = colormap2arr(arr, cm.bwr)
#     value = value.reshape(96,64)
#     values.append(value)
#
# EPV = np.loadtxt('/home/aarongu/Downloads/EPV_grid.csv', delimiter=',')
# EPV_Values = []
# for i in range(3):
#     EPV_Value = midpoint_double1(calculate_epv, values[i], -48, 48, -32, 32, 48, 32, EPV)
#     EPV_Values.append(EPV_Value)
# print(EPV_Values)

