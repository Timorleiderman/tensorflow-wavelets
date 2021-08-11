import math

import numpy as np


def FSfarras():

    a_lo_hi_rows_cols = list()
    s_lo_hi_rows_cols = list()

    a_lo_hi_rows_cols.append([
        [0,
         -0.0883883476483200,
         0.0883883476483200,
         0.695879989034000,
         0.695879989034000,
         0.0883883476483200,
         -0.0883883476483200,
         0.0112267921525400,
         0.0112267921525400,
         0],
        [0,
         -0.0112267921525400,
         0.0112267921525400,
         0.0883883476483200,
         0.0883883476483200,
         -0.695879989034000,
         0.695879989034000,
         -0.0883883476483200,
         -0.0883883476483200,
         0]
    ])

    a_lo_hi_rows_cols.append([
        [0.0112267921525400,
         0.0112267921525400,
         -0.0883883476483200,
         0.0883883476483200,
         0.695879989034000,
         0.695879989034000,
         0.0883883476483200,
         -0.0883883476483200,
         0,
         0],
        [0,
         0,
         -0.0883883476483200,
         -0.0883883476483200,
         0.695879989034000,
         -0.695879989034000,
         0.0883883476483200,
         0.0883883476483200,
         0.0112267921525400,
         -0.0112267921525400]

    ])
    s_lo_hi_rows_cols.append(list())
    s_lo_hi_rows_cols.append(list())

    s_lo_hi_rows_cols[0].append(a_lo_hi_rows_cols[0][0][::-1])
    s_lo_hi_rows_cols[0].append(a_lo_hi_rows_cols[0][1][::-1])
    s_lo_hi_rows_cols[1].append(a_lo_hi_rows_cols[1][0][::-1])
    s_lo_hi_rows_cols[1].append(a_lo_hi_rows_cols[1][1][::-1])

    return a_lo_hi_rows_cols, s_lo_hi_rows_cols


def duelfilt():
    a_lo_hi_rows_cols = list()
    s_lo_hi_rows_cols = list()
    a_lo_hi_rows_cols.append([
        [0.0351638400000000,
         0,
         -0.0883294200000000,
         0.233890320000000,
         0.760272370000000,
         0.587518300000000,
         0,
         -0.114301840000000,
         0,
         0],
        [0,
         0,
         -0.114301840000000,
         0,
         0.587518300000000,
         -0.760272370000000,
         0.233890320000000,
         0.0883294200000000,
         0,
         -0.0351638400000000]
    ])
    a_lo_hi_rows_cols.append([
        [0,
         0,
         -0.114301840000000,
         0,
         0.587518300000000,
         0.760272370000000,
         0.233890320000000,
         -0.0883294200000000,
         0,
         0.0351638400000000,
         ],
        [-0.0351638400000000,
         0,
         0.0883294200000000,
         0.233890320000000,
         -0.760272370000000,
         0.587518300000000,
         0,
         -0.114301840000000,
         0,
         0]
    ])

    s_lo_hi_rows_cols.append(list())
    s_lo_hi_rows_cols.append(list())

    s_lo_hi_rows_cols[0].append(a_lo_hi_rows_cols[0][0][::-1])
    s_lo_hi_rows_cols[0].append(a_lo_hi_rows_cols[0][1][::-1])
    s_lo_hi_rows_cols[1].append(a_lo_hi_rows_cols[1][0][::-1])
    s_lo_hi_rows_cols[1].append(a_lo_hi_rows_cols[1][1][::-1])

    return a_lo_hi_rows_cols, s_lo_hi_rows_cols


def ghm():
    # initialize the coefficients
    h0 = [[3/(5*math.sqrt(2)), 4/5], [-1/20, -3/(10*math.sqrt(2))]]

    h1 = [[3/(5*math.sqrt(2)), 0], [9/20,  1/math.sqrt(2)]]

    h2 = [[0, 0], [9/20, -3/(10*math.sqrt(2))]]

    h3 = [[0, 0], [-1/20, 0]]

    g0 = [[-1/20, -3/(10*math.sqrt(2))], [1/(10*math.sqrt(2)), 3/10]]

    g1 = [[9/20, -1/math.sqrt(2)], [-9/(10*math.sqrt(2)), 0]]

    g2 = [[9/20, -3/(10*math.sqrt(2))], [9/(10*math.sqrt(2)), -3/10]]

    g3 = [[-1/20, 0], [-1/(10*math.sqrt(2)), 0]]

    h_filter = [x+y+z+w for x,y,z,w in zip(h0, h1, h2, h3)]
    g_filter = [x+y+z+w for x,y,z,w in zip(g0, g1, g2, g3)]

    w = h_filter + g_filter
    hight, width = 512, 512
    w_mat = np.zeros((2*hight-4, 2*width))

    last_filter1 = [x+y for x,y in zip(h2, h3)]
    last_filter2 = [x+y for x,y in zip(g2, g3)]
    last_filter3 = [x+y for x,y in zip(h0, h1)]
    last_filter4 = [x+y for x,y in zip(g0, g1)]

    last_fil12 = last_filter1 + last_filter2
    last_fil34 = last_filter3 + last_filter4
    zeros_between = np.zeros((4, 2*hight-8))
    lat_box = np.concatenate([last_fil12, zeros_between, last_fil34], axis=1)
    for i in range((hight//2)-1):
        w_mat[4*i:4*(i+1), 4*(i+1)-4:4*(i+1)+4] = w

    w_mat = np.concatenate([w_mat, lat_box], axis=0)

    return [h_filter[0], h_filter[1], g_filter[0], g_filter[1]]


