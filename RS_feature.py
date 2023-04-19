"""
This module is used to extract RS feature.
The RS feature is extracted from the residual noise by applying a high efficient denoising filter and a weight assignment strategy.
The denoising filter reduces the noise level and preserves the RS patterns,
while the weight assignment strategy enhances the RS feature map by assigning different weights to different frequency subbands.
"""

from math import ceil

import numpy as np
from pywt import dwt2, idwt2
from scipy.signal import correlate2d

na = np.array
ne = np.empty
nz = np.zeros
no = np.ones


# Maximum a posteriori (MAP)
def MAP(coef, NoiseVar):
    coefVar = correlate2d(coef ** 2, no([3, 3]) / (3 * 3), 'same')

    for w in range(5, 10, 2):
        EstVar = correlate2d(coef ** 2, no([w, w]) / (w * w), 'same')
        coefVar = np.minimum(coefVar, EstVar)

    coefVar = coefVar - NoiseVar
    coefVar[coefVar < 0] = 0
    tc = coef * NoiseVar / (coefVar + NoiseVar)
    return tc


# Wavelet filtering to obtain cover estimation
def WaveletFiltering(img, NoiseVar=0.5, L=1):
    img = img.astype(np.float64)
    (M, N) = img.shape
    m = 2 ** L
    minpad = 2

    nr = ceil((M + minpad) / m) * m
    nc = ceil((N + minpad) / m) * m
    pr = ceil((nr - M) / 2)
    pc = ceil((nc - N) / 2)

    img = np.pad(img, (nr - M - 1, nc - N - 1), 'symmetric')

    coeffs = dwt2(img, 'db8', mode='symmetric')
    cA, (cH, cV, cD) = coeffs

    cA = nz(np.shape(cA))
    cH = MAP(cH, NoiseVar)
    cV = MAP(cV, NoiseVar)
    cD = MAP(cD, NoiseVar)

    coeffs2 = cA, (cH, cV, cD)
    img2 = idwt2(coeffs2, 'db8', mode='symmetric')

    C_hat = img2[pr:pr + M, pc:pc + N]

    return C_hat


# Local variance v1
def localvar_1(x, k, X_hat):
    # K must be an odd number, which is set to a constant 3 in our code
    if k % 2 == 0:
        k = k + 1

    dif = x - X_hat
    dif_sq = dif * dif
    kern = no([k, k], dtype=int)

    # Here the correlate2d is used to accelerate the operation.
    return correlate2d(dif_sq, kern, 'same', 'wrap') / (k ** 2 - 1)


# Local variance v2
def localvar_2(x, k, X_hat):
    (m, n) = np.shape(x)
    x_p = np.pad(x, ((1, 1), (1, 1)), 'wrap')
    X_hat = X_hat.reshape(-1, order='F')
    c_group = ne([3, 64 * 64 * 3])

    for i in range(m * n):
        choose = x_p[i % 64:i % 64 + k, i // 64:i // 64 + k] - X_hat[i]
        choose[1, 1] = 0
        c_group[0:k, i * k:i * k + k] = np.square(choose) / 8

    # Here the correlate2d is used to accelerate the operation.
    lv = correlate2d(c_group, no([k, k], dtype=int), 'same', 'wrap')
    return lv[1, 1::3].reshape((m, n), order='F')


# Weighting coefficient v1
def weight_coefficient_1(S, X_hat):
    varS = localvar_1(S, 3, X_hat)
    w1 = 1 / (5 + varS ** 2)
    return w1


# Weighting coefficient v2
def weight_coefficient_2(S, X_hat):
    varS = localvar_2(S, 3, X_hat)
    w2 = 1 / (5 + varS ** 2)
    return w2


# Weighted residual noise v1
def RS_1(S, w, X_hat):
    # The LSB plane flipped image
    Sbar = S + 1 - 2 * (S % 2)

    beta_hat = w * (S - X_hat) * (S - Sbar)
    return beta_hat


# Weighted residual noise v2
def RS_2(S, w, X_hat):
    Sbar = S + 1 - 2 * (S % 2)

    beta_hat = w * X_hat * (S - Sbar)
    return beta_hat


mcf2 = lambda stego, weight, cover: RS_2(stego, weight, cover)
mcf1 = lambda stego, weight, cover: RS_1(stego, weight, cover)


def RS_feature_extraction_1(stego_list, weight_list, cover_list):
    feature = list(map(mcf1, stego_list, weight_list, cover_list))
    return na(sum(feature)).reshape((-1, 1), order='F')


def RS_feature_extraction_2(stego_list, weight_list, cover_list):
    feature = list(map(mcf2, stego_list, weight_list, cover_list))
    return na(sum(feature)).reshape((-1, 1), order='F')
