import os
import errno
import numpy as np
import tensorflow as tf
from tensorflow.python.ops import linalg_ops
from tfoptests.persistor import TensorFlowPersistor, BASE_DIR

np.random.seed(seed=713)
tf.set_random_seed(1)


def _GetNormOpTest(dtype_, shape_, ord_, axis_, keep_dims_, use_static_shape_, save_dir_):
    def _CompareNorm(matrix):
        # tf_matrix = tf.Variable(matrix,name="input")
        tf.reset_default_graph()
        in_node = tf.placeholder("float", matrix.shape, name="input")
        in0 = tf.Variable(tf.random_normal(matrix.shape), name="in0", dtype=tf.float32)
        tf_matrix = in_node + in0
        tf_norm = linalg_ops.norm(
            tf_matrix, ord=ord_, axis=axis_, keep_dims=keep_dims_, name="norm_op")
        out_node = tf.identity(tf_norm, name="output")
        init = tf.global_variables_initializer()
        sess = tf.Session()
        sess.run(init)
        tfp = TensorFlowPersistor(save_dir=save_dir_)
        tfp.set_placeholders([in_node]) \
            .set_training_sess(sess) \
            .set_output_tensors([out_node]) \
            .set_test_data({"input": matrix}) \
            .build_save_frozen_graph()

    def Test(save_dir_i_):
        is_matrix_norm = (isinstance(axis_, tuple) or
                          isinstance(axis_, list)) and len(axis_) == 2
        is_fancy_p_norm = np.isreal(ord_) and np.floor(ord_) != ord_
        if ((not is_matrix_norm and ord_ == "fro") or
                (is_matrix_norm and is_fancy_p_norm)):
            print("Not supported by neither numpy.linalg.norm nor tf.norm")
            print("==========================================================")
            return
        if is_matrix_norm and ord_ == 2:
            print("Not supported by tf.norm")
            print("==========================================================")
            return
        matrix = np.random.randn(*shape_).astype(dtype_)
        test_info = BASE_DIR + "/" + save_dir_i_ + "/test.info"
        print("writing to...")
        print(test_info)
        if not os.path.exists(os.path.dirname(test_info)):
            try:
                os.makedirs(os.path.dirname(test_info))
            except OSError as exc:  # Guard against race condition
                if exc.errno != errno.EEXIST:
                    raise
        print("matrix dims")
        print(matrix.shape)
        with open(test_info, "w") as f:
            f.write("ord ")
            f.write(str(ord_))
            f.write("\naxis ")
            f.write(str(axis_))
            f.write("\nkeep_dims ")
            f.write(str(keep_dims_))
            f.write("\ninput matrix shape ")
            f.write(str(matrix.shape))
            if dtype_ in (np.complex64, np.complex128):
                matrix += 1j * np.random.randn(*shape_).astype(dtype_)
            _CompareNorm(matrix)
            print("==========================================================")

    return Test


def test_norm():
    use_static_shape = False
    dtype = np.float32
    test_num = 0
    # for rows in 2, 5:
    for rows in [2]:
        # for cols in 2, 5:
        for cols in [5]:
            # for batch in [], [2], [2, 3]:
            for batch in [[]]:
                shape = batch + [rows, cols]
                for ord in "euclidean", "fro", 0.5, 1, 2, np.inf:
                    # for axis in [None, (-2, -1), (-1, -2), -len(shape), 0, len(shape) - 1]:
                    for axis in [None, (-2, -1)]:
                        # for keep_dims in False, True:
                        for keep_dims in [False]:
                            name = "%s_ord_%s_axis_%s_%s" % (
                                "_".join(map(str, shape)), ord, axis,
                                keep_dims)
                            if name not in ["2_2_ord_0.5_axis_None_False",
                                            "2_2_ord_0.5_axis_None_True"]:
                                print(name)
                                save_dir_i = "norm_tests" + "/" + "norm_" + str(test_num)
                                test_func = _GetNormOpTest(dtype, shape, ord, axis, keep_dims,
                                                           use_static_shape, save_dir_i)
                                test_func(save_dir_i)
                                test_num += 1


if __name__ == '__main__':
    test_norm()
