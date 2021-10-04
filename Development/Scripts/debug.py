
from utils.cast import tf_to_ndarray
from utils.write_raw import tensor_to_write_raw


def debug_raw(w):

    lo11 = tf_to_ndarray(w[2][0][0])
    lo12 = tf_to_ndarray(w[2][0][1])
    lo21 = tf_to_ndarray(w[2][1][0])
    lo22 = tf_to_ndarray(w[2][1][1])

    ch_t1111 = tf_to_ndarray(w[0][0][0][0])
    cv_t1112 = tf_to_ndarray(w[0][0][0][1])
    cd_t1113 = tf_to_ndarray(w[0][0][0][2])

    ch_t1121 = tf_to_ndarray(w[0][0][1][0])
    cv_t1122 = tf_to_ndarray(w[0][0][1][1])
    cd_t1123 = tf_to_ndarray(w[0][0][1][2])

    ch_t1211 = tf_to_ndarray(w[0][1][0][0])
    cv_t1212 = tf_to_ndarray(w[0][1][0][1])
    cd_t1213 = tf_to_ndarray(w[0][1][0][2])

    ch_t1221 = tf_to_ndarray(w[0][1][1][0])
    cv_t1222 = tf_to_ndarray(w[0][1][1][1])
    cd_t1223 = tf_to_ndarray(w[0][1][1][2])

    ch_t2111 = tf_to_ndarray(w[1][0][0][0])
    cv_t2112 = tf_to_ndarray(w[1][0][0][1])
    cd_t2113 = tf_to_ndarray(w[1][0][0][2])

    ch_t2121 = tf_to_ndarray(w[1][0][1][0])
    cv_t2122 = tf_to_ndarray(w[1][0][1][1])
    cd_t2123 = tf_to_ndarray(w[1][0][1][2])

    ch_t2211 = tf_to_ndarray(w[1][1][0][0])
    cv_t2212 = tf_to_ndarray(w[1][1][0][1])
    cd_t2213 = tf_to_ndarray(w[1][1][0][2])

    ch_t2221 = tf_to_ndarray(w[1][1][1][0])
    cv_t2222 = tf_to_ndarray(w[1][1][1][1])
    cd_t2223 = tf_to_ndarray(w[1][1][1][2])

    tensor_to_write_raw(r"G:\My Drive\Colab Notebooks\MWCNN\output\lo11_python.hex", lo11)
    tensor_to_write_raw(r"G:\My Drive\Colab Notebooks\MWCNN\output\lo12_python.hex", lo12)
    tensor_to_write_raw(r"G:\My Drive\Colab Notebooks\MWCNN\output\lo21_python.hex", lo21)
    tensor_to_write_raw(r"G:\My Drive\Colab Notebooks\MWCNN\output\lo22_python.hex", lo22)

    tensor_to_write_raw(r"G:\My Drive\Colab Notebooks\MWCNN\output\ch_t1111_python.hex", ch_t1111)
    tensor_to_write_raw(r"G:\My Drive\Colab Notebooks\MWCNN\output\cv_t1112_python.hex", cv_t1112)
    tensor_to_write_raw(r"G:\My Drive\Colab Notebooks\MWCNN\output\cd_t1113_python.hex", cd_t1113)

    tensor_to_write_raw(r"G:\My Drive\Colab Notebooks\MWCNN\output\ch_t1121_python.hex", ch_t1121)
    tensor_to_write_raw(r"G:\My Drive\Colab Notebooks\MWCNN\output\cv_t1122_python.hex", cv_t1122)
    tensor_to_write_raw(r"G:\My Drive\Colab Notebooks\MWCNN\output\cd_t1123_python.hex", cd_t1123)

    tensor_to_write_raw(r"G:\My Drive\Colab Notebooks\MWCNN\output\ch_t1211_python.hex", ch_t1211)
    tensor_to_write_raw(r"G:\My Drive\Colab Notebooks\MWCNN\output\cv_t1212_python.hex", cv_t1212)
    tensor_to_write_raw(r"G:\My Drive\Colab Notebooks\MWCNN\output\cd_t1213_python.hex", cd_t1213)

    tensor_to_write_raw(r"G:\My Drive\Colab Notebooks\MWCNN\output\ch_t1221_python.hex", ch_t1221)
    tensor_to_write_raw(r"G:\My Drive\Colab Notebooks\MWCNN\output\cv_t1222_python.hex", cv_t1222)
    tensor_to_write_raw(r"G:\My Drive\Colab Notebooks\MWCNN\output\cd_t1223_python.hex", cd_t1223)

    tensor_to_write_raw(r"G:\My Drive\Colab Notebooks\MWCNN\output\ch_t2111_python.hex", ch_t2111)
    tensor_to_write_raw(r"G:\My Drive\Colab Notebooks\MWCNN\output\cv_t2112_python.hex", cv_t2112)
    tensor_to_write_raw(r"G:\My Drive\Colab Notebooks\MWCNN\output\cd_t2113_python.hex", cd_t2113)

    tensor_to_write_raw(r"G:\My Drive\Colab Notebooks\MWCNN\output\ch_t2121_python.hex", ch_t2121)
    tensor_to_write_raw(r"G:\My Drive\Colab Notebooks\MWCNN\output\cv_t2122_python.hex", cv_t2122)
    tensor_to_write_raw(r"G:\My Drive\Colab Notebooks\MWCNN\output\cd_t2123_python.hex", cd_t2123)

    tensor_to_write_raw(r"G:\My Drive\Colab Notebooks\MWCNN\output\ch_t2211_python.hex", ch_t2211)
    tensor_to_write_raw(r"G:\My Drive\Colab Notebooks\MWCNN\output\cv_t2212_python.hex", cv_t2212)
    tensor_to_write_raw(r"G:\My Drive\Colab Notebooks\MWCNN\output\cd_t2213_python.hex", cd_t2213)

    tensor_to_write_raw(r"G:\My Drive\Colab Notebooks\MWCNN\output\ch_t2221_python.hex", ch_t2221)
    tensor_to_write_raw(r"G:\My Drive\Colab Notebooks\MWCNN\output\cv_t2222_python.hex", cv_t2222)
    tensor_to_write_raw(r"G:\My Drive\Colab Notebooks\MWCNN\output\cd_t2223_python.hex", cd_t2223)
