"""
This is the main part of the program.

Input : A set of stego JPEG images with the same dimensions and embedded with the same stego-key,
and the corresponding stego-key
Output : A location map that indicates the embedding positions in the input images,
represented by a binary matrix with the same size as the input images

Stego-key is a binary matrix with the same dimensions as the image, which determines whether a given position is an embedding position or not during the embedding process.A value of 0 indicates a non-embedding position (non-payload position),and a value of 1 indicates an embedding position (payload position).
"""

import gc
import time

import joblib
import jpegio as jio
import numpy as np
from sklearn.metrics import precision_score, recall_score, accuracy_score, confusion_matrix

from RS_feature import RS_feature_extraction_1, RS_feature_extraction_2, WaveletFiltering, weight_coefficient_1, \
    weight_coefficient_2
from Stego_feature import stego_feature_extraction
from model import ensemble_classifier

na = np.array
ne = np.empty
nz = np.zeros
no = np.ones


# Generating co-frequency subimages
def cfsi(img):
    (col, row) = img.shape
    subimg = ne(64, dtype=object)
    for m in range(8):
        for n in range(8):
            subimg[m * 8 + n] = img[m:col:8, n:row:8]
    return subimg


# Generating DCT co-frequency subimages
def dct_cfsi(dcts, num):
    (col, row) = dcts[0].shape
    subdct = ne(64, dtype=object)
    for i in range(64):
        subdct[i] = []
        sa = subdct[i].append
        for j in range(num):
            sa(dcts[j][i // 8:col:8, i % 8:row:8])
    return subdct


# Importing images
def readin(path, start, end, trainortest, rate):
    path = (path, trainortest, '_', str(rate), '_75/')
    dcts = list()
    da = dcts.append
    for i in range(start, end + 1):
        print(i)
        seq = (''.join(path), str(i), '.jpg')
        img = jio.read(''.join(seq))
        img = img.coef_arrays[0].astype(np.float32)
        da(img)
    return dcts


# Counting the number of non-zero coefficients at each DCT coefficient position.
def location_statistics(dcts):
    (m, n) = np.shape(dcts[0])
    num = nz([m, n], dtype=np.float64)
    for i in range(len(dcts)):
        index = (dcts[i] != 0).astype(np.float64)
        num[index == 1] += 1
    return num


# Generate labels with stego-key for steganographic embedding paths.
def label_generating(label, num):
    label[num == 0] = 0
    return label


# Effective mean normalization
def effective_mean_normalization(feature, num):
    feature, num = na(feature), na(num)
    for i in range(64):
        num[i] = num[i].reshape(-1, order='F')
        for j in range(64 * 64):
            feature[i][j] /= num[i][j]
        feature[i][np.isinf(feature[i])] = 0
        feature[i][np.isnan(feature[i])] = 0
    return feature


# Weight normalization
def weight_normalization(feature, weight):
    feature, weight = na(feature), na(weight)
    weight_list = []
    wa = weight_list.append
    for i in range(64):
        wa(na(sum(weight[i])).reshape((-1, 1), order='F'))
        feature[i] /= weight_list[i]
        feature[i][np.isinf(feature[i])] = 0
        feature[i][np.isnan(feature[i])] = 0
    return feature


if __name__ == '__main__':
    # Timer start
    time_start = time.time()

    rate = 0.5
    num_train = 100
    num_test = 100

    path = "D:/Stego/Jsteg_"
    # path = "D:/Stego/JstegM_"
    # path = "D:/Stego/F5_"
    path_train = (path, 'train_', str(rate), '_75', '/label_train.csv')
    path_test = (path, 'test_', str(rate), '_75', '/label_test.csv')

    # Read in the DCT coefficients of the image set.
    dct_train = readin(path, 1, num_train, 'train', rate)
    dct_test = readin(path, 1001, 1000 + num_test, 'test', rate)
    # Counting the number of non-zero coefficients at each DCT coefficient position.
    num_train_s = location_statistics(dct_train)
    num_test_s = location_statistics(dct_test)
    subnum_train = cfsi(num_train_s)
    subnum_test = cfsi(num_test_s)

    # RS feature extraction
    # Generating DCT co-frequency subimages
    subdct_train = dct_cfsi(dct_train, num_train)
    subdct_test = dct_cfsi(dct_test, num_test)

    # Estimation of cover images
    msc = lambda subdct: list(map(WaveletFiltering, subdct))
    subcover_train = list(map(msc, subdct_train))
    subcover_test = list(map(msc, subdct_test))
    # Calculation of weights by local variance v1 (l1)
    msw1 = lambda subdct, subcover: list(map(weight_coefficient_1, subdct, subcover))
    subweight_train_l1 = list(map(msw1, subdct_train, subcover_train))
    subweight_test_l1 = list(map(msw1, subdct_test, subcover_test))
    # Calculation of weights by local variance v2 (l2)
    msw2 = lambda subdct, subcover: list(map(weight_coefficient_2, subdct, subcover))
    subweight_train_l2 = list(map(msw2, subdct_train, subcover_train))
    subweight_test_l2 = list(map(msw2, subdct_test, subcover_test))

    # Weighted residual noise v1 (RS1)
    subfeature_train_l1_rs1 = list(map(RS_feature_extraction_1, subdct_train, subweight_train_l1, subcover_train))
    subfeature_test_l1_rs1 = list(map(RS_feature_extraction_1, subdct_test, subweight_test_l1, subcover_test))
    subfeature_train_l2_rs1 = list(map(RS_feature_extraction_1, subdct_train, subweight_train_l2, subcover_train))
    subfeature_test_l2_rs1 = list(map(RS_feature_extraction_1, subdct_test, subweight_test_l2, subcover_test))
    # Weighted residual noise v2 (RS2)
    subfeature_train_l1_rs2 = list(map(RS_feature_extraction_2, subdct_train, subweight_train_l1, subcover_train))
    subfeature_test_l1_rs2 = list(map(RS_feature_extraction_2, subdct_test, subweight_test_l1, subcover_test))
    subfeature_train_l2_rs2 = list(map(RS_feature_extraction_2, subdct_train, subweight_train_l2, subcover_train))
    subfeature_test_l2_rs2 = list(map(RS_feature_extraction_2, subdct_test, subweight_test_l2, subcover_test))

    # RS1
    # Effective mean normalization-l1&l2
    subfeature_train_1 = effective_mean_normalization(subfeature_train_l1_rs1, subnum_train)
    subfeature_test_1 = effective_mean_normalization(subfeature_test_l1_rs1, subnum_test)
    subfeature_train_2 = effective_mean_normalization(subfeature_train_l2_rs1, subnum_train)
    subfeature_test_2 = effective_mean_normalization(subfeature_test_l2_rs1, subnum_test)
    # Weight normalization-l1&l2
    subfeature_train_3 = weight_normalization(subfeature_train_l1_rs1, subweight_train_l1)
    subfeature_test_3 = weight_normalization(subfeature_test_l1_rs1, subweight_test_l1)
    subfeature_train_4 = weight_normalization(subfeature_train_l2_rs1, subweight_train_l2)
    subfeature_test_4 = weight_normalization(subfeature_test_l2_rs1, subweight_test_l2)

    # RS2
    # Effective mean normalization-l1&l2
    subfeature_train_5 = effective_mean_normalization(subfeature_train_l1_rs2, subnum_train)
    subfeature_test_5 = effective_mean_normalization(subfeature_test_l1_rs2, subnum_test)
    subfeature_train_6 = effective_mean_normalization(subfeature_train_l2_rs2, subnum_train)
    subfeature_test_6 = effective_mean_normalization(subfeature_test_l2_rs2, subnum_test)
    # Weight normalization-l1&l2
    subfeature_train_7 = weight_normalization(subfeature_train_l1_rs2, subweight_train_l1)
    subfeature_test_7 = weight_normalization(subfeature_test_l1_rs2, subweight_test_l1)
    subfeature_train_8 = weight_normalization(subfeature_train_l2_rs2, subweight_train_l2)
    subfeature_test_8 = weight_normalization(subfeature_test_l2_rs2, subweight_test_l2)

    # Combine all 8 residuals to obtain RS features.
    mhs = lambda feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8: np.hstack(
        (feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8))
    RSfeature_train = list(
        map(mhs, subfeature_train_1, subfeature_train_2, subfeature_train_3, subfeature_train_4, subfeature_train_5,
            subfeature_train_6, subfeature_train_7, subfeature_train_8))
    RSfeature_test = list(
        map(mhs, subfeature_test_1, subfeature_test_2, subfeature_test_3, subfeature_test_4, subfeature_test_5,
            subfeature_test_6, subfeature_test_7, subfeature_test_8))

    # Stego feature extraction
    # Extracting the stego features of each subband
    stegofeature_train = list(map(stego_feature_extraction, subdct_train))
    stegofeature_test = list(map(stego_feature_extraction, subdct_test))
    # Normalize the stego features for all subbands
    stegofeature_train = effective_mean_normalization(stegofeature_train, subnum_train)
    stegofeature_test = effective_mean_normalization(stegofeature_test, subnum_test)

    # Read in the stego-key.
    stego_key_train = np.loadtxt(''.join(path_train), dtype=np.float32, delimiter=',')
    stego_key_test = np.loadtxt(''.join(path_test), dtype=np.float32, delimiter=',')
    # Labels generating
    label_train = label_generating(stego_key_train.reshape([512, 512], order='F'), num_train_s)
    label_test = label_generating(stego_key_test.reshape([512, 512], order='F'), num_test_s)
    # Preprocessing of labels
    sublabel_train = cfsi(label_train.reshape([512, 512], order='F'))
    sublabel_test = cfsi(label_test.reshape([512, 512], order='F'))

    # Release the memory if necessary.
    del dct_train, dct_test, subdct_train, subdct_test, stego_key_train, stego_key_test, subfeature_train_1, subfeature_train_2, subfeature_train_3, subfeature_train_4, subfeature_train_5,
    subfeature_train_6, subfeature_train_7, subfeature_train_8, subfeature_test_1, subfeature_test_2, subfeature_test_3, subfeature_test_4, subfeature_test_5,
    subfeature_test_6, subfeature_test_7, subfeature_test_8
    gc.collect()

    # Three evaluation metrics: accuracy, precision, and recall
    accuracy_train = nz([8, 8], dtype=np.float32)
    accuracy_test = nz([8, 8], dtype=np.float32)
    precision_train = nz([8, 8], dtype=np.float32)
    precision_test = nz([8, 8], dtype=np.float32)
    recall_train = nz([8, 8], dtype=np.float32)
    recall_test = nz([8, 8], dtype=np.float32)
    # Confusion Matrix
    TP_train = nz([8, 8], dtype=np.float32)
    TP_test = nz([8, 8], dtype=np.float32)
    FN_train = nz([8, 8], dtype=np.float32)
    FN_test = nz([8, 8], dtype=np.float32)
    FP_train = nz([8, 8], dtype=np.float32)
    FP_test = nz([8, 8], dtype=np.float32)
    TN_train = nz([8, 8], dtype=np.float32)
    TN_test = nz([8, 8], dtype=np.float32)

    for i in range(1, 64):
        print(i)
        # Fusion future
        fusion_feature_train = np.hstack((stegofeature_train[i], RSfeature_train[i]))
        fusion_feature_test = np.hstack((stegofeature_test[i], RSfeature_test[i]))
        # Constructing the classifier in subband i
        classifier = ensemble_classifier
        # Train & Predict
        classifier.fit(fusion_feature_train, sublabel_train[i].reshape(-1, order='F'))
        pred_train = classifier.predict(fusion_feature_train)
        pred_test = classifier.predict(fusion_feature_test)
        # Calculation of three evaluation metrics in subband i
        accuracy_train[i // 8, i % 8] = accuracy_score(sublabel_train[i].reshape(-1, order='F'), pred_train)
        accuracy_test[i // 8, i % 8] = accuracy_score(sublabel_test[i].reshape(-1, order='F'), pred_test)
        precision_train[i // 8, i % 8] = precision_score(sublabel_train[i].reshape(-1, order='F'), pred_train,
                                                         average='binary')
        precision_test[i // 8, i % 8] = precision_score(sublabel_test[i].reshape(-1, order='F'), pred_test,
                                                        average='binary')
        recall_train[i // 8, i % 8] = recall_score(sublabel_train[i].reshape(-1, order='F'), pred_train,
                                                   average='binary')
        recall_test[i // 8, i % 8] = recall_score(sublabel_test[i].reshape(-1, order='F'), pred_test, average='binary')
        # Confusion Matrix
        CM_train = confusion_matrix(sublabel_train[i].reshape(-1, order='F'), pred_train, labels=[1, 0])
        CM_test = confusion_matrix(sublabel_test[i].reshape(-1, order='F'), pred_test, labels=[1, 0])
        TP_train[i // 8, i % 8] = CM_train[0, 0]
        TP_test[i // 8, i % 8] = CM_test[0, 0]
        FN_train[i // 8, i % 8] = CM_train[0, 1]
        FN_test[i // 8, i % 8] = CM_test[0, 1]
        FP_train[i // 8, i % 8] = CM_train[1, 0]
        FP_test[i // 8, i % 8] = CM_test[1, 0]
        TN_train[i // 8, i % 8] = CM_train[1, 1]
        TN_test[i // 8, i % 8] = CM_test[1, 1]

        print(str(i), "Training accuracy: ", accuracy_train[i // 8, i % 8])
        print(str(i), "Test accuracy: ", accuracy_test[i // 8, i % 8])

        # Save the trained classifier in subband i.
        joblib.dump(classifier, ''.join(('ensemble_classifier_in_subband_', str(i), '_', str(num_train))))

    # Evaluating the performance of our method on the entire image scale using three metrics.
    print("Overall training accuracy: ", (np.sum(TP_train) + np.sum(TN_train)) / (
            np.sum(TP_train) + np.sum(FN_train) + np.sum(FP_train) + np.sum(TN_train)))
    print("Overall test accuracy: ",
          (np.sum(TP_test) + np.sum(TN_test)) / (np.sum(TP_test) + np.sum(FN_test) + np.sum(FP_test) + np.sum(TN_test)))
    print("Overall training precision: ", np.sum(TP_train) / (np.sum(TP_train) + np.sum(FP_train)))
    print("Overall test precision: ", np.sum(TP_test) / (np.sum(TP_test) + np.sum(FP_test)))
    print("Overall training recall: ", np.sum(TP_train) / (np.sum(TP_train) + np.sum(FN_train)))
    print("Overall test recall: ", np.sum(TP_test) / (np.sum(TP_test) + np.sum(FN_test)))

    # Timer end
    time_end = time.time()
    print('totally cost', time_end - time_start)
