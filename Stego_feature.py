"""
This module is used to extract Stego feature.
The stego feature is extracted from the original DCT coefficient by exploiting the adjacent correlation property of the DCT domain.
The adjacent correlation reflects the level of inter-pixel dependency in the spatial domain of the stego image.
By measuring the correlation between adjacent DCT coefficients, the stego feature can capture the changes introduced by data hiding.
"""

import numpy as np

na = np.array
ne = np.empty
nz = np.zeros
no = np.ones


# Stego feature extraction
def stego_feature_extraction(stego_list):
    # Here m,n are the length and width of the co-frequency subimages respectively.
    m, n = 64, 64
    feature = nz([m * n, 72], dtype=np.float32)

    for stego in stego_list:
        # Edge processing
        stego = np.pad(stego, ((1, 1), (1, 1)), 'wrap')

        # Computing adjacent DCT difference coefficients
        difference = []
        da = difference.append
        for k in range(8):
            da(nz([m + 2, n + 2], dtype=np.float32))
        difference[0][0:m, 1:n] = stego[0:m, 1:n] - stego[0:m, 0:n - 1]
        difference[1][0:m, 0:n - 1] = stego[0:m, 0:n - 1] - stego[0:m, 1:n]
        difference[2][1:m, 0:n] = stego[1:m, 0:n] - stego[0:m - 1, 0:n]
        difference[3][0:m - 1, 0:n] = stego[0:m - 1, 0:n] - stego[1:m, 0:n]
        difference[4][1:m, 0:n - 1] = stego[1:m, 0:n - 1] - stego[0:m - 1, 1:n]
        difference[5][1:m, 1:n] = stego[1:m, 1:n] - stego[0:m - 1, 0:n - 1]
        difference[6][0:m - 1, 0:n - 1] = stego[0:m - 1, 0:n - 1] - stego[1:m, 1:n]
        difference[7][0:m - 1, 1:n] = stego[0:m - 1, 1:n] - stego[1:m, 0:n - 1]

        # Statistical modeling of adjacent DCT coefficient differences
        for j in range(8):
            difference[j] = difference[j][1:m + 1, 1:n + 1]
            difference[j] = difference[j].reshape(-1, order='F')

            feature[difference[j] == -4, j * 9] += 1
            feature[difference[j] == -3, j * 9 + 1] += 1
            feature[difference[j] == -2, j * 9 + 2] += 1
            feature[difference[j] == -1, j * 9 + 3] += 1
            feature[difference[j] == 0, j * 9 + 4] += 1
            feature[difference[j] == 1, j * 9 + 5] += 1
            feature[difference[j] == 2, j * 9 + 6] += 1
            feature[difference[j] == 3, j * 9 + 7] += 1
            feature[difference[j] == 4, j * 9 + 8] += 1

    for p in range(0, 64 * 64):
        for q in range(8):
            feature[p, (q * 9):(q * 9 + 9)] /= sum(feature[p, (q * 9):(q * 9 + 9)])

    return feature
