import math
import os
import cv2
import tensorflow as tf
from utils import filters
from utils.helpers import over_sample_rows
from utils.cast import tf_to_ndarray, cast_like_matlab_uint8_2d
from utils.helpers import *
from utils.mse import mse


def synthesis_filter_bank2d_ghm_mult(x):

    h = int(x.shape[1])//2
    w = int(x.shape[2])//2

    w_mat = filters.ghm_w_mat(h, w)
    w_mat_tf = tf.constant(w_mat, dtype=tf.float32)
    w_mat_tf = tf.transpose(w_mat_tf)
    w_mat_tf = tf.expand_dims(w_mat_tf, axis=-1)
    w_mat_tf = tf.expand_dims(w_mat_tf, axis=0)
    w_mat_tf = tf.repeat(w_mat_tf, 3, axis=-1)

    ll = tf.split(tf.split(x, 2, axis=1)[0], 2, axis=2)[0]
    lh = tf.split(tf.split(x, 2, axis=1)[0], 2, axis=2)[1]
    hl = tf.split(tf.split(x, 2, axis=1)[1], 2, axis=2)[0]
    hh = tf.split(tf.split(x, 2, axis=1)[1], 2, axis=2)[1]

    ll = up_sample_4_1(ll)
    lh = up_sample_4_1(lh)
    hl = up_sample_4_1(hl)
    hh = up_sample_4_1(hh)

    recon_1 = tf.concat([tf.concat([ll, lh], axis=2), tf.concat([hl, hh], axis=2)], axis=1)
    recon_1_tr = tf.transpose(recon_1, perm=[0, 2, 1, 3])

    perm_cols = permute_rows_4_2(recon_1_tr)
    # cros_w_x = tf.matmul(w_mat_tf, perm_cols[:, ..., 0])
    cros_w_x = tf.einsum('bijw,bjkw->bikw', w_mat_tf, perm_cols)
    # cros_w_x = tf.expand_dims(cros_w_x, axis=-1)

    cros_w_x_ds = cros_w_x[:, 0::2, :, :]
    cros_w_x_ds_tr = tf.transpose(cros_w_x_ds, perm=[0, 2, 1, 3])
    perm_rows = permute_rows_4_2(cros_w_x_ds_tr)

    # cross_w_perm_rows = tf.matmul(w_mat_tf, perm_rows[:, ..., 0])
    cross_w_perm_rows = tf.einsum('bijw,bjkw->bikw', w_mat_tf, perm_rows)
    # cross_w_perm_rows = tf.expand_dims(cross_w_perm_rows, axis=-1)

    res = cross_w_perm_rows[:, 0::2, :, :]
    return res


def analysis_filter_bank2d_ghm_mult(x):
    # parameters
    conv_type = 'same'
    h = int(x.shape[1])
    w = int(x.shape[2])

    w_mat = filters.ghm_w_mat(h, w)
    w_mat_tf = tf.constant(w_mat, dtype=tf.float32)
    w_mat_tf = tf.expand_dims(w_mat_tf, axis=-1)
    w_mat_tf = tf.expand_dims(w_mat_tf, axis=0)
    w_mat_tf = tf.repeat(w_mat_tf, 3, axis=-1)

    x_os = over_sample_rows(x)

    cros_w_x = tf.einsum('bijw,bjkw->bikw', w_mat_tf, x_os)
    #cros_w_x = tf.expand_dims(cros_w_x, axis=-1)


    perm_rows = permute_rows_2_1(cros_w_x)
    perm_rows_tr = tf.transpose(perm_rows, perm=[0, 2, 1, 3])
    perm_rows_os = over_sample_rows(perm_rows_tr)

    # z_w_x = tf.matmul(w_mat_tf, perm_rows_os[:, ..., 0])
    z_w_x = tf.einsum('bijw,bjkw->bikw', w_mat_tf, perm_rows_os)
    # z_w_x = tf.expand_dims(z_w_x, axis=-1)
    perm_cols = permute_rows_2_1(z_w_x)
    perm_cols = tf.transpose(perm_cols, perm=[0, 2, 1, 3])

    ll = tf.split(tf.split(perm_cols, 2, axis=1)[0], 2, axis=2)[0]
    lh = tf.split(tf.split(perm_cols, 2, axis=1)[0], 2, axis=2)[1]
    hl = tf.split(tf.split(perm_cols, 2, axis=1)[1], 2, axis=2)[0]
    hh = tf.split(tf.split(perm_cols, 2, axis=1)[1], 2, axis=2)[1]

    ll = split_to_ll_lh_hl_hh(ll)
    lh = split_to_ll_lh_hl_hh(lh)
    hl = split_to_ll_lh_hl_hh(hl)
    hh = split_to_ll_lh_hl_hh(hh)

    res = tf.concat([tf.concat([ll, lh], axis=2), tf.concat([hl, hh], axis=2)], axis=1)
    return res


def analysis_filter_bank2d_ghm(x):
    # parameters
    conv_type = 'same'
    h = int(x.shape[1])
    w = int(x.shape[2])

    ghm_fir = filters.ghm()
    lp1, lp2, hp1, hp2 = construct_tf_filter(ghm_fir[0], ghm_fir[1], ghm_fir[2], ghm_fir[3])
    filt_len = int(lp1.shape[1])
    x_os = over_sample_rows(x)
    x_pad = tf.pad(x_os,[[0, 0], [filt_len, filt_len], [0, 0], [0, 0]], mode='CONSTANT',constant_values=0)

    lp1_ds = fir_down_sample(x_pad, lp1, filt_len-2)
    lp1_ds1 = lp1_ds[:, 0:lp1_ds.shape[1]-5:2, :, :]

    lp2_ds = fir_down_sample(x_pad, lp2, filt_len-2)
    lp2_ds1 = lp2_ds[:, 2:lp2_ds.shape[1]-3:2, :, :]

    hp1_ds = fir_down_sample(x_pad, hp1, filt_len-2)
    hp1_ds1 = hp1_ds[:, 0:lp1_ds.shape[1]-5:2, :, :]

    hp2_ds = fir_down_sample(x_pad, hp2, filt_len-2)
    hp2_ds1 = hp2_ds[:, 2:lp2_ds.shape[1]-3:2, :, :]*(-1)

    lp1_ds1_tr = tf.transpose(lp1_ds1, perm=[0,2,1,3])
    lp2_ds1_tr = tf.transpose(lp2_ds1, perm=[0,2,1,3])
    hp1_ds1_tr = tf.transpose(hp1_ds1, perm=[0,2,1,3])
    hp2_ds1_tr = tf.transpose(hp2_ds1, perm=[0,2,1,3])

    lp1_ds1_os = over_sample_rows(lp1_ds1_tr)
    lp2_ds1_os = over_sample_rows(lp2_ds1_tr)
    hp1_ds1_os = over_sample_rows(hp1_ds1_tr)
    hp2_ds1_os = over_sample_rows(hp2_ds1_tr)

    lp1_ds1_os_pad = tf.pad(lp1_ds1_os,[[0, 0], [filt_len, filt_len], [0, 0], [0, 0]], mode='CONSTANT',constant_values=0)
    lp2_ds1_os_pad = tf.pad(lp2_ds1_os,[[0, 0], [filt_len, filt_len], [0, 0], [0, 0]], mode='CONSTANT',constant_values=0)
    hp1_ds1_os_pad = tf.pad(hp1_ds1_os,[[0, 0], [filt_len, filt_len], [0, 0], [0, 0]], mode='CONSTANT',constant_values=0)
    hp2_ds1_os_pad = tf.pad(hp2_ds1_os,[[0, 0], [filt_len, filt_len], [0, 0], [0, 0]], mode='CONSTANT',constant_values=0)

    lp1_lp1_ds = fir_down_sample(lp1_ds1_os_pad, lp1, start=filt_len-2, step=4)
    lp1_hp1_ds = fir_down_sample(lp1_ds1_os_pad, hp1, start=filt_len-2, step=4)
    hp1_lp1_ds = fir_down_sample(hp1_ds1_os_pad, lp1, start=filt_len-2, step=4)
    hp1_hp1_ds = fir_down_sample(hp1_ds1_os_pad, hp1, start=filt_len-2, step=4)

    lp1_lp1_tr = tf.transpose(lp1_lp1_ds[:,:-3,:,:], perm=[0,2,1,3])
    lp1_hp1_tr = tf.transpose(lp1_hp1_ds[:,:-3,:,:], perm=[0,2,1,3])
    hp1_lp1_tr = tf.transpose(hp1_lp1_ds[:,:-3,:,:], perm=[0,2,1,3])
    hp1_hp1_tr = tf.transpose(hp1_hp1_ds[:,:-3,:,:], perm=[0,2,1,3])

    lp1_lp2_ds = fir_down_sample(lp1_ds1_os_pad, lp2, start=filt_len-2, step=4)
    lp1_hp2_ds = fir_down_sample(lp1_ds1_os_pad, hp2, start=filt_len-2, step=4)
    hp1_lp2_ds = fir_down_sample(hp1_ds1_os_pad, lp2, start=filt_len-2, step=4)
    hp1_hp2_ds = fir_down_sample(hp1_ds1_os_pad, hp2, start=filt_len-2, step=4)

    lp1_lp2_tr = tf.transpose(lp1_lp2_ds[:,:-3,:,:], perm=[0,2,1,3])
    lp1_hp2_tr = tf.transpose(lp1_hp2_ds[:,:-3,:,:], perm=[0,2,1,3])
    hp1_lp2_tr = tf.transpose(hp1_lp2_ds[:,:-3,:,:], perm=[0,2,1,3])
    hp1_hp2_tr = tf.transpose(hp1_hp2_ds[:,:-3,:,:], perm=[0,2,1,3])

    lp2_lp1_ds = fir_down_sample(lp2_ds1_os_pad, lp1, start=filt_len-2, step=4)
    lp2_hp1_ds = fir_down_sample(lp2_ds1_os_pad, hp1, start=filt_len-2, step=4)
    hp2_lp1_ds = fir_down_sample(hp2_ds1_os_pad, lp1, start=filt_len-2, step=4)
    hp2_hp1_ds = fir_down_sample(hp2_ds1_os_pad, hp1, start=filt_len-2, step=4)

    lp2_lp1_tr = tf.transpose(lp2_lp1_ds[:,:-3,:,:], perm=[0,2,1,3])
    lp2_hp1_tr = tf.transpose(lp2_hp1_ds[:,:-3,:,:], perm=[0,2,1,3])
    hp2_lp1_tr = tf.transpose(hp2_lp1_ds[:,:-3,:,:], perm=[0,2,1,3])
    hp2_hp1_tr = tf.transpose(hp2_hp1_ds[:,:-3,:,:], perm=[0,2,1,3])

    lp2_lp2_ds = fir_down_sample(lp2_ds1_os_pad, lp2, start=filt_len-2, step=4)
    lp2_hp2_ds = fir_down_sample(lp2_ds1_os_pad, hp2, start=filt_len-2, step=4)
    hp2_lp2_ds = fir_down_sample(hp2_ds1_os_pad, lp2, start=filt_len-2, step=4)
    hp2_hp2_ds = fir_down_sample(hp2_ds1_os_pad, hp2, start=filt_len-2, step=4)

    lp2_lp2_tr = tf.transpose(lp2_lp2_ds[:,:-3,:,:], perm=[0,2,1,3])
    lp2_hp2_tr = tf.transpose(lp2_hp2_ds[:,:-3,:,:], perm=[0,2,1,3])
    hp2_lp2_tr = tf.transpose(hp2_lp2_ds[:,:-3,:,:], perm=[0,2,1,3])
    hp2_hp2_tr = tf.transpose(hp2_hp2_ds[:,:-3,:,:], perm=[0,2,1,3])

    res = [[lp1_lp1_tr, lp1_hp1_tr,hp1_lp1_tr, hp1_hp1_tr],
           [lp1_lp2_tr, lp1_hp2_tr,hp1_lp2_tr, hp1_hp2_tr],
           [lp2_lp1_tr, lp2_hp1_tr,hp2_lp1_tr, hp2_hp1_tr],
           [lp2_lp2_tr, lp2_hp2_tr,hp2_lp2_tr, hp2_hp2_tr],
    ]
    return res


img_grey = cv2.imread("../input/Lenna_orig.png", 1)

x_f32 = tf.cast(img_grey, dtype=tf.float32)
w, h, c = img_grey.shape
# x_f32 = tf.expand_dims(x_f32, axis=-1)
x_f32 = tf.expand_dims(x_f32, axis=0)


decomp = analysis_filter_bank2d_ghm_mult(x_f32)
recon = synthesis_filter_bank2d_ghm_mult(decomp)
recon_img = tf_to_ndarray(recon)


print(mse(img_grey, recon_img))

# ll = tf.concat([tf.concat([res[0][0], res[1][0]], axis=2), tf.concat([res[2][0], res[3][0]], axis=2)], axis=1)
# lh = tf.concat([tf.concat([res[0][1], res[1][1]], axis=2), tf.concat([res[2][1], res[3][1]], axis=2)], axis=1)
# hl = tf.concat([tf.concat([res[0][2], res[1][2]], axis=2), tf.concat([res[2][2], res[3][2]], axis=2)], axis=1)
# hh = tf.concat([tf.concat([res[0][3], res[1][3]], axis=2), tf.concat([res[2][3], res[3][3]], axis=2)], axis=1)
#
# res = tf.concat([tf.concat([ll, lh], axis=2), tf.concat([hl, hh], axis=2)], axis=1)
#
# # cv2.imshow("test", cast_like_matlab_uint8_2d(tf_to_ndarray(res)).astype('uint8'))
# # cv2.waitKey(0)
# synthesis_filter_bank2d_ghm(res)


print("hey")