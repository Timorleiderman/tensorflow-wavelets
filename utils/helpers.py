
import math
import tensorflow as tf


def roll_pad(data, pad_len):

    # circular shift
    # This procedure (periodic extension) can create
    # undesirable artifacts at the beginning and end
    # of the subband signals, however, it is the most
    # convenient solution.
    # When the analysis and synthesis filters are exactly symmetric,
    # a different procedure (symmetric extension) can be used,
    # that avoids the artifacts associated with periodic extension

    data_roll = tf.roll(data, shift=-pad_len, axis=1)
    # zero padding

    data_roll_pad = tf.pad(data_roll,
                           [[0, 0], [pad_len, pad_len], [0, 0], [0, 0]],
                           mode='CONSTANT',
                           constant_values=0)

    return data_roll_pad


def fir_down_sample(data, fir):
    # input tensors rank 4

    data_tr = tf.transpose(data, perm=[0, 2, 1, 3])
    conv = tf.nn.conv2d(
        data_tr, fir, padding='SAME', strides=[1, 1, 1, 1],
    )
    conv_tr = tf.transpose(conv, perm=[0, 2, 1, 3])

    # down sample
    lo_conv_ds = conv_tr[:, 0:conv_tr.shape[1]:2, :, :]
    return lo_conv_ds


def circular_shift_fix_crop(data, shift_fix, crop):

    circular_shift_fix = tf.math.add(data[:, 0:shift_fix, :, :],
                                     data[:, -shift_fix:, :, :])

    fix = tf.concat([circular_shift_fix, data[:, shift_fix:, :, :]], axis=1)

    if crop == 0:
        res = fix
    else:
        res = fix[:, 0:crop, :, :]

    return res


def list_to_tf(data):
    list_len = len(data)
    data_tf = tf.constant(data)
    data_tf = tf.reshape(data_tf, (1, list_len, 1, 1))
    return data_tf


def duel_filter_tf(duelfilt):
    filt_len = len(duelfilt[0][0])

    tree1_lp_tf = list_to_tf(duelfilt[0][0])
    tree1_hp_tf = list_to_tf(duelfilt[0][1])
    tree2_lp_tf = list_to_tf(duelfilt[1][0])
    tree2_hp_tf = list_to_tf(duelfilt[1][1])

    tree1 = tf.stack((tree1_lp_tf, tree1_hp_tf), axis=0)
    tree2 = tf.stack((tree2_lp_tf, tree2_hp_tf), axis=0)
    duelfilt_tf = tf.stack((tree1, tree2), axis=0)
    return duelfilt_tf


def construct_tf_filter(lod_row, hid_row, lod_col, hid_col):

    filt_len = len(lod_row)

    lod_row_tf = tf.constant(lod_row[::-1])
    lod_row_tf = tf.reshape(lod_row_tf, (1, filt_len, 1, 1))

    hid_row_tf = tf.constant(hid_row[::-1])
    hid_row_tf = tf.reshape(hid_row_tf, (1, filt_len, 1, 1))

    lod_col_tf = tf.constant(lod_col[::-1])
    lod_col_tf = tf.reshape(lod_col_tf, (1, filt_len, 1, 1))

    hid_col_tf = tf.constant(hid_col[::-1])
    hid_col_tf = tf.reshape(hid_col_tf, (1, filt_len, 1, 1))

    return lod_row_tf, hid_row_tf, lod_col_tf, hid_col_tf


def add_sub(a, b):
    add = (a + b) / math.sqrt(2)
    sub = (a - b) / math.sqrt(2)
    return add, sub


def up_sample_fir(x, fir):
    # create zero like tensor
    x = tf.transpose(x, perm=[0, 2, 1, 3])
    zero_tensor = tf.zeros_like(x)
    # stack both tensors
    stack_rows = tf.stack([x, zero_tensor], axis=3)
    # reshape for zero insertion between the rows
    stack_rows = tf.reshape(stack_rows, shape=[-1, x.shape[1], x.shape[2]*2, x.shape[3]])

    conv = tf.nn.conv2d(
        stack_rows, fir, padding='SAME', strides=[1, 1, 1, 1],
    )
    res = tf.transpose(conv, perm=[0, 2, 1, 3])
    return res


def analysis_filter_bank2d(x, lod_row, hid_row, lod_col, hid_col):
    # parameters
    h = int(x.shape[1])
    w = int(x.shape[2])
    filt_len = int(lod_row.shape[1])
    x_roll_padd = roll_pad(x, filt_len//2)

    lo_conv_ds = fir_down_sample(x_roll_padd, lod_row)
    hi_conv_ds = fir_down_sample(x_roll_padd, hid_row)

    # # crop to needed dims
    lo = circular_shift_fix_crop(lo_conv_ds, filt_len//2, h//2)
    hi = circular_shift_fix_crop(hi_conv_ds, filt_len//2, h//2)

    # next is the columns filtering
    lo_tr = tf.transpose(lo, perm=[0, 2, 1, 3])
    hi_tr = tf.transpose(hi, perm=[0, 2, 1, 3])

    lo_tr_roll_padd = roll_pad(lo_tr, filt_len//2)
    hi_tr_roll_padd = roll_pad(hi_tr, filt_len//2)

    lo_lo_conv_ds = fir_down_sample(lo_tr_roll_padd, lod_col)
    lo_hi_conv_ds = fir_down_sample(lo_tr_roll_padd, hid_col)
    hi_lo_conv_ds = fir_down_sample(hi_tr_roll_padd, lod_col)
    hi_hi_conv_ds = fir_down_sample(hi_tr_roll_padd, hid_col)

    lo_lo = circular_shift_fix_crop(lo_lo_conv_ds, filt_len//2, w//2)
    lo_hi = circular_shift_fix_crop(lo_hi_conv_ds, filt_len//2, w//2)
    hi_lo = circular_shift_fix_crop(hi_lo_conv_ds, filt_len//2, w//2)
    hi_hi = circular_shift_fix_crop(hi_hi_conv_ds, filt_len//2, w//2)

    lo_lo = tf.transpose(lo_lo, perm=[0, 2, 1, 3])
    lo_hi = tf.transpose(lo_hi, perm=[0, 2, 1, 3])
    hi_lo = tf.transpose(hi_lo, perm=[0, 2, 1, 3])
    hi_hi = tf.transpose(hi_hi, perm=[0, 2, 1, 3])

    return [lo_lo, [lo_hi, hi_lo, hi_hi]]
