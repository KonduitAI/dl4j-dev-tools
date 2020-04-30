import tensorflow as tf
import numpy as np


class OpCreator:
    def __init__(self, op):
        self.op = op
        self.node_num = 0

    def setVars(self, vars):
        self.vars = vars

    def setPlaceholders(self, placeholders):
        self.placeholders = placeholders

    def execute(self, some_op):
        self.node_num += 1
        method_name = 'execute_' + some_op
        try:
            method = getattr(self, method_name)
        except AttributeError:
            print(method_name, "not found - no method has been defined for this")
        else:
            return method()

    def execute_hsv_to_rgb(self):
        return [tf.image.hsv_to_rgb(self.vars[0])]

    def execute_rgb_to_hsv(self):
        return [tf.image.rgb_to_hsv(self.vars[0])]

    def execute_yiq_to_rgb(self):
        return [tf.image.yiq_to_rgb(self.vars[0])]

    def execute_rgb_to_yiq(self):
        return [tf.image.rgb_to_yiq(self.vars[0])]

    def execute_yuv_to_rgb(self):
        return [tf.image.yuv_to_rgb(self.vars[0])]

    def execute_rgb_to_yuv(self):
        return [tf.image.rgb_to_yuv(self.vars[0])]

    def execute_rgb_to_grayscale(self):
        return [tf.image.rgb_to_grayscale(self.vars[0])]

    def execute_adjust_saturation(self):
        return [tf.image.adjust_saturation(self.vars[0], self.op["factor"])]

    def execute_reduce_sum(self):
        return [tf.reduce_sum(input_tensor=self.vars[0], axis=self.op["axis"], keepdims=self.op["keepdims"])]

    def execute_segment_max(self):
        return [tf.math.segment_max(data=self.vars[0], segment_ids=self.vars[1])]

    def execute_segment_min(self):
        return [tf.math.segment_min(data=self.vars[0], segment_ids=self.vars[1])]

    def execute_segment_mean(self):
        return [tf.math.segment_mean(data=self.vars[0], segment_ids=self.vars[1])]

    def execute_segment_prod(self):
        return [tf.math.segment_prod(data=self.vars[0], segment_ids=self.vars[1])]

    def execute_segment_sum(self):
        return [tf.math.segment_sum(data=self.vars[0], segment_ids=self.vars[1])]

    def execute_space_to_batch(self):
        return [tf.space_to_batch(input=self.vars[0], paddings=self.vars[1], block_shape=2)]

    def execute_space_to_depth(self):
        return [tf.compat.v1.space_to_depth(input=self.vars[0], block_size=2, data_format=self.op["data_format"])]

    def execute_batch_to_space(self):
        return [tf.batch_to_space(input=self.vars[0], crops=self.vars[1], block_shape=2)]

    def execute_depth_to_space(self):
        return [tf.compat.v1.depth_to_space(input=self.vars[0], block_size=2, data_format=self.op["data_format"])]

    def execute_size(self):
        temp = tf.add(self.vars[0], 1.0)
        return [tf.add(tf.size(input=temp), 1)]

    def execute_shape(self):
        temp = tf.add(self.vars[0], 1.0)
        return [tf.add(tf.shape(input=temp), 1)]

    def execute_shapen(self):
        out = tf.shape_n(input=self.vars)
        #Concat multiple outputs to avoid graph saving issue
        return [tf.concat(out, axis=0)]

    def execute_matrix_inverse(self):
        return [tf.linalg.inv(input=self.vars[0])]

    def execute_pad(self):
        if(len(self.vars) > 2):
            return [tf.pad(tensor=self.vars[0], paddings=self.vars[1], constant_values=self.vars[2], mode = self.op["mode"])]
        else:
            return [tf.pad(tensor=self.vars[0], paddings=self.vars[1], mode=self.op["mode"])]

    def execute_unique(self):
        #Hack for multi-output saving issue: concat
        temp = tf.unique(self.vars[0])
        toConcat = []
        toConcat.append(temp[0])
        toConcat.append(tf.cast(temp[1], dtype=tf.float32))
        return [tf.concat(toConcat, axis=0)]

    def execute_unique_with_counts(self):
        temp = tf.unique_with_counts(self.vars[0])
        toConcat = []
        toConcat.append(temp[0])
        toConcat.append(tf.cast(temp[1], dtype=tf.float32))
        toConcat.append(tf.cast(temp[2], dtype=tf.float32))
        return [tf.concat(toConcat,axis=0)]

    def execute_topk(self):
        temp = tf.nn.top_k(input=self.vars[0], k=self.op["k"], sorted=self.op["sorted"])
        toConcat = []
        toConcat.append(temp[0])
        toConcat.append(tf.cast(temp[1], dtype=tf.float32))
        #Concat multiple outputs to avoid graph saving issue. Note that values and indices have same shape
        return [tf.concat(toConcat, axis=0)]

    def execute_in_top_k(self):
        return [tf.nn.in_top_k(predictions=self.vars[0], targets=self.vars[1], k=self.op["k"])]

    def execute_matrix_determinant(self):
        return [tf.linalg.det(input=self.vars[0])]

    def execute_matrix_set_diag(self):
        return [tf.linalg.set_diag(input=self.vars[0], diagonal=self.vars[1])]

    def execute_identity(self):
        return [tf.identity(self.vars[0])]

    def execute_identity_n(self):
        return tf.identity_n(self.vars)

    def execute_zeta(self):
        x = tf.add(self.vars[0], 1.0)    #x values must be > 1
        return [tf.math.zeta(x=x, q=self.vars[1])]

    def execute_confusion_matrix(self):
        weights = None
        if(len(self.vars) > 2):
            weights = self.vars[2]
        return [tf.math.confusion_matrix(labels=self.vars[0], predictions=self.vars[1], num_classes=self.op["num_classes"], weights=weights)]

    def execute_stack(self):
        return [tf.stack(values=self.vars, axis=self.op["axis"])]

    def execute_parallel_stack(self):
        return [tf.parallel_stack(values=self.vars)]

    def execute_accumulate_n(self):
        return [tf.math.accumulate_n(self.vars)]

    def execute_angle(self):
        return [tf.add(tf.math.angle(self.vars[0]), 1.0)]

    def execute_approximate_equal(self):
        return [tf.approximate_equal(self.vars[0], self.vars[1])]

    def execute_matmul(self):
        ta = self.op.get("transpose_a", False)
        tb = self.op.get("transpose_b", False)
        print(self.op)
        print("ta = ",ta)
        print("tb = ",tb)
        return [tf.matmul(self.vars[0], self.vars[1], transpose_a=ta, transpose_b=tb, name = "matmul-" + str(self.node_num))]

    def execute_matrix_diag_part(self):
        return [tf.linalg.diag_part(self.vars[0])]

    def execute_svd(self):
        shapes = self.op["varShapes"]
        if(shapes[len(shapes)-1] != shapes[len(shapes)-2]):
            raise ValueError("Only square inputs currently supported due to multiple outputs issue")

        # Add identity so
        input = self.vars[0]
        shape = self.op["varShapes"][0]
        rank = len(shape)
        # if(input.rank() == 2):
        #     #OK as is
        # el
        if(rank == 2):
            input = tf.add(input, tf.linalg.eye(num_rows=shape[0], num_columns=shape[1]))
        if(rank == 3):
            input = tf.add(input, tf.linalg.eye(num_rows=shape[1], num_columns=shape[2], batch_shape=[shape[0]]))
        if(rank == 4):
            input = tf.add(input, tf.linalg.eye(num_rows=shape[2], num_columns=shape[3], batch_shape=[shape[0], shape[1]]))

        svd = tf.linalg.svd(tensor=input, full_matrices=self.op["full_matrices"], compute_uv=self.op["compute_uv"])

        # Note that SVD has multiple solutions, that differ only by sign
        if(isinstance(svd, list) or isinstance(svd, tuple)):
            return [tf.math.abs(svd[0]), tf.math.abs(svd[1]), tf.math.abs(svd[2])]
        else:
            return [tf.math.abs(svd)]


    def execute_mean_squared_error(self):
        weights = 1.0
        if(len(self.vars) > 2):
            weights = self.vars[2]

        return [tf.compat.v1.losses.mean_squared_error(labels=self.vars[0], predictions=self.vars[1], weights=weights)]

    def execute_absolute_difference(self):
        weights = 1.0
        if(len(self.vars) > 2):
            weights = self.vars[2]

        return [tf.compat.v1.losses.absolute_difference(labels=self.vars[0], predictions=self.vars[1], weights=weights)]

    def execute_cosine_distance(self):
        weights = 1.0
        if(len(self.vars) > 2):
            weights = self.vars[2]
        r = self.op.get("reduction", tf.compat.v1.losses.Reduction.SUM_BY_NONZERO_WEIGHTS)

        return [tf.compat.v1.losses.cosine_distance(labels=self.vars[0], predictions=self.vars[1], weights=weights, axis=self.op["axis"], reduction=r)]

    def execute_hinge_loss(self):
        weights = 1.0
        if(len(self.vars) > 2):
            weights = self.vars[2]
        r = self.op.get("reduction", tf.compat.v1.losses.Reduction.SUM_BY_NONZERO_WEIGHTS)

        return [tf.compat.v1.losses.hinge_loss(labels=self.vars[0], logits=self.vars[1], weights=weights, reduction=r)]

    def execute_huber_loss(self):
        weights = 1.0
        if(len(self.vars) > 2):
            weights = self.vars[2]
        r = self.op.get("reduction", tf.compat.v1.losses.Reduction.SUM_BY_NONZERO_WEIGHTS)
        delta = self.op.get("delta", 1.0)

        return [tf.compat.v1.losses.huber_loss(labels=self.vars[0], predictions=self.vars[1], weights=weights, reduction=r, delta=delta)]

    def execute_log_loss(self):
        weights = 1.0
        if(len(self.vars) > 2):
            weights = self.vars[2]
        r = self.op.get("reduction", tf.compat.v1.losses.Reduction.SUM_BY_NONZERO_WEIGHTS)
        eps = self.op.get("epsilon", 1e-7)

        return [tf.compat.v1.losses.log_loss(labels=self.vars[0], predictions=self.vars[1], weights=weights, reduction=r, epsilon=eps)]

    def execute_sigmoid_cross_entropy(self):
        weights = 1.0
        if(len(self.vars) > 2):
            weights = self.vars[2]
        ls = self.op.get("label_smoothing",0)
        r = self.op.get("reduction", tf.compat.v1.losses.Reduction.SUM_BY_NONZERO_WEIGHTS)

        return [tf.compat.v1.losses.sigmoid_cross_entropy(multi_class_labels=self.vars[0], logits=self.vars[1], weights=weights, label_smoothing=ls, reduction=r)]

    def execute_softmax_cross_entropy(self):
        weights = 1.0
        if(len(self.vars) > 2):
            weights = self.vars[2]
        ls = self.op.get("label_smoothing",0)
        r = self.op.get("reduction", tf.compat.v1.losses.Reduction.SUM_BY_NONZERO_WEIGHTS)

        return [tf.compat.v1.losses.softmax_cross_entropy(onehot_labels=self.vars[0], logits=self.vars[1], weights=weights, label_smoothing=ls, reduction=r)]

    def execute_sparse_softmax_cross_entropy(self):
        weights = 1.0
        if(len(self.vars) > 2):
            weights = self.vars[2]
        r = self.op.get("reduction", tf.compat.v1.losses.Reduction.SUM_BY_NONZERO_WEIGHTS)

        return [tf.compat.v1.losses.sparse_softmax_cross_entropy(labels=self.vars[0], logits=self.vars[1], weights=weights, reduction=r)]

    def execute_l2_loss(self):
        return [tf.nn.l2_loss(self.vars[0])]

    def execute_nn_cnn1d(self):
        return [tf.nn.conv1d(input=self.vars[0], filters=self.vars[1], stride=self.op["stride"], padding=self.op["padding"], data_format=self.op["data_format"])]

    def execute_layers_cnn1d(self):
        kr = self.op.get("kernel_regularizer",None)
        br = self.op.get("bias_regularizer",None)
        ar = self.op.get("activity_regularizer",None)
        kc = self.op.get("kernel_constraint",None)
        bc = self.op.get("bias_constraint",None)
        print("kernel constraint: ", kc)
        print("bias constraint: ", bc)
        return [tf.compat.v1.layers.conv1d(inputs=self.vars[0], filters=self.op["filters"], kernel_size=self.op["kernel_size"], strides=self.op["strides"],
                                 padding=self.op["padding"], data_format=self.op["data_format"], dilation_rate=self.op["dilation_rate"],
                                 kernel_regularizer=kr, bias_regularizer=br, activity_regularizer=ar, kernel_constraint=kc, bias_constraint=bc)]

    def execute_max_pooling1d(self):
        return [tf.compat.v1.layers.max_pooling1d(inputs=self.vars[0], pool_size=self.op["pooling_size"], strides=self.op["stride"], padding=self.op["padding"], data_format=self.op["data_format"])]

    def execute_avg_pooling1d(self):
        return [tf.compat.v1.layers.average_pooling1d(inputs=self.vars[0], pool_size=self.op["pooling_size"], strides=self.op["stride"], padding=self.op["padding"], data_format=self.op["data_format"])]

    def execute_max_pool_with_argmax(self):
        return [tf.nn.max_pool_with_argmax(input = self.vars[0], ksize = self.op["ksizes"], strides = self.op["strides"],\
                                           padding = self.op["padding"],  data_format = self.op["data_format"], output_dtype = self.op["output_dtype"])]

    def execute_dense(self):
        kr = self.op.get("kernel_regularizer",None)
        br = self.op.get("bias_regularizer",None)
        return [tf.compat.v1.layers.dense(inputs=self.vars[0], units=self.op["units"], activation=self.op["activation"], use_bias=self.op["use_bias"], kernel_regularizer=kr, bias_regularizer=br)]

    def execute_flatten(self):
        return [tf.compat.v1.layers.flatten(inputs=self.vars[0])]

    def execute_nn_conv2d(self):
        return [tf.nn.conv2d(input=self.vars[0], filters=self.vars[1], strides=self.op["strides"], padding=self.op["padding"],
                             data_format=self.op["data_format"], dilations=self.op.get("dilations", [1,1,1,1]))]

    def execute_layers_conv2d(self):
        return [tf.compat.v1.layers.conv2d(inputs=self.vars[0], filters=self.op["filters"], kernel_size=self.op["kernel_size"], strides=self.op["strides"],
                                 padding=self.op["padding"], data_format=self.op["data_format"], dilation_rate=self.op["dilation_rate"],
                                 activation=self.op.get("activation",None), kernel_regularizer=self.op.get("kernel_regularizer",None),
                                 bias_regularizer=self.op.get("bias_regularizer",None), activity_regularizer=self.op.get("activity_regularizer",None),
                                 kernel_constraint=self.op.get("kernel_constraint",None), bias_constraint=self.op.get("bias_constraint",None))]

    def execute_layers_sepconv1d(self):
        return [tf.compat.v1.layers.separable_conv1d(inputs=self.vars[0], filters=self.op["filters"], kernel_size=self.op["kernel_size"], strides=self.op["strides"],
                                           padding=self.op["padding"], data_format=self.op["data_format"], dilation_rate=self.op["dilation_rate"],
                                           depth_multiplier=self.op["depth_multiplier"],
                                           activation=self.op.get("activation",None), depthwise_regularizer=self.op.get("kernel_regularizer",None),
                                           bias_regularizer=self.op.get("bias_regularizer",None), activity_regularizer=self.op.get("activity_regularizer",None),
                                           depthwise_constraint=self.op.get("kernel_constraint",None), bias_constraint=self.op.get("bias_constraint",None))]

    def execute_layers_sepconv2d(self):
        return [tf.compat.v1.layers.separable_conv1d(inputs=self.vars[0], filters=self.op["filters"], kernel_size=self.op["kernel_size"], strides=self.op["strides"],
                                           padding=self.op["padding"], data_format=self.op["data_format"], dilation_rate=self.op["dilation_rate"],
                                           depth_multiplier=self.op["depth_multiplier"],
                                           activation=self.op.get("activation",None), depthwise_regularizer=self.op.get("kernel_regularizer",None),
                                           bias_regularizer=self.op.get("bias_regularizer",None), activity_regularizer=self.op.get("activity_regularizer",None),
                                           depthwise_constraint=self.op.get("kernel_constraint",None), bias_constraint=self.op.get("bias_constraint",None))]

    def execute_layers_conv2d_transpose(self):
        return [tf.compat.v1.layers.conv2d_transpose(inputs=self.vars[0], filters=self.op["filters"], kernel_size=self.op["kernel_size"], strides=self.op["strides"],
                                 padding=self.op["padding"], data_format=self.op["data_format"],
                                 activation=self.op.get("activation",None), kernel_regularizer=self.op.get("kernel_regularizer",None),
                                 bias_regularizer=self.op.get("bias_regularizer",None), activity_regularizer=self.op.get("activity_regularizer",None),
                                 kernel_constraint=self.op.get("kernel_constraint",None), bias_constraint=self.op.get("bias_constraint",None))]


    def execute_layers_conv3d(self):
        return [tf.compat.v1.layers.conv3d(inputs=self.vars[0], filters=self.op["filters"], kernel_size=self.op["kernel_size"], strides=self.op["strides"],
                                 padding=self.op["padding"], data_format=self.op["data_format"], dilation_rate=self.op["dilation_rate"],
                                 activation=self.op.get("activation",None), kernel_regularizer=self.op.get("kernel_regularizer",None),
                                 bias_regularizer=self.op.get("bias_regularizer",None), activity_regularizer=self.op.get("activity_regularizer",None),
                                 kernel_constraint=self.op.get("kernel_constraint",None), bias_constraint=self.op.get("bias_constraint",None))]

    def execute_max_pooling3d(self):
        return [tf.compat.v1.layers.max_pooling3d(inputs=self.vars[0], pool_size=self.op["pooling_size"], strides=self.op["stride"], padding=self.op["padding"], data_format=self.op["data_format"])]

    def execute_avg_pooling3d(self):
        return [tf.compat.v1.layers.average_pooling3d(inputs=self.vars[0], pool_size=self.op["pooling_size"], strides=self.op["stride"], padding=self.op["padding"], data_format=self.op["data_format"])]

    def execute_conv3d_transpose_layers(self):
        return [tf.compat.v1.layers.conv3d_transpose(inputs=self.vars[0], filters=self.op["filters"], kernel_size=self.op["kernel_size"], strides=self.op["strides"],
                                           padding=self.op["padding"], data_format=self.op["data_format"],
                                           activation=self.op.get("activation",None), kernel_regularizer=self.op.get("kernel_regularizer",None),
                                           bias_regularizer=self.op.get("bias_regularizer",None), activity_regularizer=self.op.get("activity_regularizer",None),
                                           kernel_constraint=self.op.get("kernel_constraint",None), bias_constraint=self.op.get("bias_constraint",None))]

    def execute_conv3d_transpose_nn(self):
        # "TypeError: conv3d_transpose() got an unexpected keyword argument 'dilations'" :/
        # return [tf.nn.conv3d_transpose(value=self.vars[0], filter=self.vars[1], strides=self.op["strides"],padding=self.op["padding"], data_format=self.op["data_format"], dilations=self.op["dilations"])]
        return [tf.nn.conv3d_transpose(input=self.vars[0], filters=self.vars[1], output_shape=self.op["output_shape"], strides=self.op["strides"],padding=self.op["padding"], data_format=self.op["data_format"])]

    def execute_batchnorm(self):
        return [tf.compat.v1.layers.batch_normalization(inputs=self.vars[0], axis=self.op["axis"], momentum=self.op.get("momentum",0.99), epsilon=self.op.get("epsilon",0.001),
                                              center=self.op.get("center",True), scale=self.op.get("scale",True), fused=self.op["fused"])]

    def execute_embedding_lookup(self):
        nParamArrs = len(self.vars)-1
        params = []
        for i in range(nParamArrs):
            params.append(self.vars[i])
        print("vars: ", self.vars)
        print("ids: ", self.vars[nParamArrs])
        return [tf.nn.embedding_lookup(params=params, ids=self.vars[nParamArrs], partition_strategy=self.op["partition_strategy"], max_norm=self.op["max_norm"])]

    def execute_l2_normalize(self):
        return [tf.nn.l2_normalize(x=self.vars[0], axis=self.op["axis"], epsilon=self.op["epsilon"])]

    def execute_lrn(self):
        return [tf.nn.lrn(input=self.vars[0], depth_radius=self.op["depth_radius"], bias=self.op["bias"], alpha=self.op["alpha"], beta=self.op["beta"])]

    def execute_layers_dropout(self):
        return [tf.compat.v1.layers.dropout(inputs=self.vars[0], rate=self.op["rate"], noise_shape=self.op.get("noise_shape",None), training=self.op["training"])]

    def execute_contrib_nn_alpha_dropout(self):
        return [tf.contrib.nn.alpha_dropout(x=self.vars[0], keep_prob=self.op["keep_prob"], noise_shape=self.op.get("noise_shape",None))]

    def execute_meshgrid(self):
        meshgrid = tf.meshgrid(self.vars, indexing=self.op["indexing"])
        return [tf.stack(meshgrid, axis=0)] #Workaround for multi-output issue

    def execute_eye(self):
        batch_shape = None
        if(len(self.vars) > 0):
            batch_shape = tf.cast(self.vars[0],dtype=tf.int32)
        return [tf.eye(num_rows=self.op["num_rows"], num_columns=self.op["num_columns"], batch_shape=batch_shape)]

    def execute_log_determinant(self):
        #Attempting to ensure the input sub-matrices are hermitian positive definite matrix... this doesn't guarantee it??
        inArr = self.vars[0]
        if(len(self.op["varShapes"][0]) == 2):
            inArr = inArr + tf.eye(num_rows=self.op["varShapes"][0][0], num_columns=self.op["varShapes"][0][1])
        elif(len(self.op["varShapes"][0]) == 3):
            inArr = inArr + tf.eye(num_rows=self.op["varShapes"][0][1], num_columns=self.op["varShapes"][0][2], batch_shape=[self.op["varShapes"][0][0]])
        elif(len(self.op["varShapes"][0]) == 4):
            inArr = inArr + tf.eye(num_rows=self.op["varShapes"][0][2], num_columns=self.op["varShapes"][0][3], batch_shape=[self.op["varShapes"][0][0], self.op["varShapes"][0][1]])
        else:
            raise ValueError("Only rank 2-4 implemented")

        return [tf.linalg.logdet(inArr)]

    def execute_slog_determinant(self):
        #Attempting to ensure the input sub-matrices are hermitian positive definite matrix... this doesn't guarantee it??
        inArr = self.vars[0]
        if(len(self.op["varShapes"][0]) == 2):
            inArr = inArr + tf.eye(num_rows=self.op["varShapes"][0][0], num_columns=self.op["varShapes"][0][1])
        elif(len(self.op["varShapes"][0]) == 3):
            inArr = inArr + tf.eye(num_rows=self.op["varShapes"][0][1], num_columns=self.op["varShapes"][0][2], batch_shape=[self.op["varShapes"][0][0]])
        elif(len(self.op["varShapes"][0]) == 4):
            inArr = inArr + tf.eye(num_rows=self.op["varShapes"][0][2], num_columns=self.op["varShapes"][0][3], batch_shape=[self.op["varShapes"][0][0], self.op["varShapes"][0][1]])
        else:
            raise ValueError("Only rank 2-4 implemented")

        return tf.linalg.slogdet(inArr)


    def execute_sequence_mask(self):
        maxLen = None
        if(len(self.vars) > 1):
            maxLen = self.vars[1]
        return [tf.sequence_mask(lengths=self.vars[0], maxlen=maxLen)]

    def execute_rint(self):
        return [tf.math.rint(self.vars[0])]

    def execute_histogram_fixed_width(self):
        return [tf.histogram_fixed_width(values=self.vars[0], value_range=self.vars[1], nbins=self.op["nbins"])]

    def execute_bincount(self):
        w = None
        if(len(self.vars) > 1):
            w = self.vars[1]
        return [tf.math.bincount(arr=self.vars[0], weights=w, minlength=self.op["minlength"], maxlength=self.op["maxlength"])]

    def execute_scatter_nd(self):
        return [tf.scatter_nd(indices=self.vars[0], updates=self.vars[1], shape=self.op["shape"])]

    def execute_scatter_nd_add(self):
        # Create an intermediate variable - otherwise the scatter op will modify the variable content in-place
        # and hence we'll save the input post-modification, rather than pre-modification
        intermediate = tf.Variable(tf.zeros(self.op["varShapes"][0]), dtype=tf.float32)
        intermediate = tf.compat.v1.assign(intermediate, self.vars[0])
        return [tf.compat.v1.scatter_nd_add(ref=intermediate, indices=self.vars[1], updates=self.vars[2], use_locking=self.op["use_locking"])]

    def execute_scatter_nd_sub(self):
        intermediate = tf.Variable(tf.zeros(self.op["varShapes"][0]), dtype=tf.float32)
        intermediate = tf.compat.v1.assign(intermediate, self.vars[0])
        return [tf.compat.v1.scatter_nd_sub(ref=intermediate, indices=self.vars[1], updates=self.vars[2], use_locking=self.op["use_locking"])]

    def execute_scatter_nd_update(self):
        intermediate = tf.Variable(tf.zeros(self.op["varShapes"][0]), dtype=tf.float32)
        intermediate = tf.compat.v1.assign(intermediate, self.vars[0])
        return [tf.compat.v1.scatter_nd_update(ref=intermediate, indices=self.vars[1], updates=self.vars[2], use_locking=self.op["use_locking"])]

    def execute_scatter_add(self):
        return [tf.compat.v1.scatter_add(ref=self.vars[0], indices=self.vars[1], updates=self.vars[2])]

    def execute_scatter_div(self):
        return [tf.compat.v1.scatter_div(ref=self.vars[0], indices=self.vars[1], updates=self.vars[2])]

    def execute_scatter_max(self):
        return [tf.compat.v1.scatter_max(ref=self.vars[0], indices=self.vars[1], updates=self.vars[2])]

    def execute_scatter_min(self):
        return [tf.compat.v1.scatter_min(ref=self.vars[0], indices=self.vars[1], updates=self.vars[2])]

    def execute_scatter_mul(self):
        return [tf.compat.v1.scatter_mul(ref=self.vars[0], indices=self.vars[1], updates=self.vars[2])]

    def execute_scatter_sub(self):
        return [tf.compat.v1.scatter_sub(ref=self.vars[0], indices=self.vars[1], updates=self.vars[2])]

    def execute_scatter_update(self):
        return [tf.compat.v1.scatter_update(ref=self.vars[0], indices=self.vars[1], updates=self.vars[2])]

    def execute_sufficient_statistics(self):
        temp = tf.add(self.vars[0], 1.0)
        return tf.nn.sufficient_statistics(x=self.vars[0], axes=self.op["axes"], shift=self.op["shift"], keepdims=self.op["keep_dims"])

    def execute_split(self):
        num_or_size_splits=self.op.get("num_or_size_split", None)
        return tf.split(value=self.vars[0], num_or_size_splits=num_or_size_splits, axis=self.op["axis"])

    def execute_reduce_logsumexp(self):
        return [tf.reduce_logsumexp(input_tensor=self.vars[0], axis=self.op["axis"], keepdims=self.op["keepdims"])]

    def execute_nth_element(self):
        return [tf.contrib.nn.nth_element(input=self.vars[0], n=self.vars[1], reverse=self.op["reverse"])]

    def execute_reduce_any(self):
        return [tf.reduce_any(input_tensor=self.vars[0], axis=self.op["axis"], keepdims=self.op["keepdims"])]

    def execute_reduce_all(self):
        return [tf.reduce_all(input_tensor=self.vars[0], axis=self.op["axis"], keepdims=self.op["keepdims"])]

    def execute_reduce_max(self):
        return [tf.reduce_max(input_tensor=self.vars[0], axis=self.op["axis"], keepdims=self.op["keepdims"])]

    def execute_reduce_min(self):
        return [tf.reduce_min(input_tensor=self.vars[0], axis=self.op["axis"], keepdims=self.op["keepdims"])]

    def execute_reduce_mean(self):
        return [tf.reduce_mean(input_tensor=self.vars[0], axis=self.op["axis"], keepdims=self.op["keepdims"])]

    def execute_reduce_prod(self):
        return [tf.reduce_prod(input_tensor=self.vars[0], axis=self.op["axis"], keepdims=self.op["keepdims"])]

    def execute_boolean_mask(self):
        return [tf.boolean_mask(tensor=self.vars[0], mask=self.vars[1])]

    def execute_where(self):
        c = self.vars[0]
        x = None
        y = None
        if(len(self.vars) > 1):
            x = self.vars[1]
            y = self.vars[2]
        else:
            tf.Variable(tf.add(self.vars[0], 0.0))
        # print("x: ",x)
        # print("y: ",y)
        # print("cond: ",c)
        return [tf.compat.v1.where(condition=c, x=x, y=y)]

    def execute_broadcast_dynamic_shape(self):
        return [tf.broadcast_dynamic_shape(self.vars[0], self.vars[1])]

    def execute_broadcast_to(self):
        return [tf.broadcast_to(input=self.vars[0], shape=self.vars[1])]

    def execute_unsorted_segment_max(self):
        return [tf.math.unsorted_segment_max(data=self.vars[0], segment_ids=self.vars[1], num_segments=self.op["num_segments"])]

    def execute_unsorted_segment_min(self):
        return [tf.math.unsorted_segment_min(data=self.vars[0], segment_ids=self.vars[1], num_segments=self.op["num_segments"])]

    def execute_unsorted_segment_mean(self):
        return [tf.math.unsorted_segment_mean(data=self.vars[0], segment_ids=self.vars[1], num_segments=self.op["num_segments"])]

    def execute_unsorted_segment_prod(self):
        return [tf.math.unsorted_segment_prod(data=self.vars[0], segment_ids=self.vars[1], num_segments=self.op["num_segments"])]

    def execute_unsorted_segment_sqrt_n(self):
        return [tf.math.unsorted_segment_sqrt_n(data=self.vars[0], segment_ids=self.vars[1], num_segments=self.op["num_segments"])]

    def execute_unsorted_segment_sum(self):
        return [tf.math.unsorted_segment_sum(data=self.vars[0], segment_ids=self.vars[1], num_segments=self.op["num_segments"])]

    def execute_truncatemod(self):
        return [tf.truncatemod(x=self.vars[0], y=self.vars[1])]

    def execute_tensordot(self):
        return [tf.tensordot(a=self.vars[0], b=self.vars[1], axes=self.op["axes"])]

    def execute_assert_equal(self):
        with tf.control_dependencies([tf.compat.v1.assert_equal(x=self.vars[0], y=self.vars[1])]):
            out = tf.add(self.vars[0], self.vars[1])
        return [out]

    def execute_assert_greater(self):
        with tf.control_dependencies([tf.compat.v1.assert_greater(x=self.vars[0], y=self.vars[1])]):
            out = tf.add(self.vars[0], self.vars[1])
        return [out]

    def execute_assert_greater_equal(self):
        with tf.control_dependencies([tf.compat.v1.assert_greater_equal(x=self.vars[0], y=self.vars[1])]):
            out = tf.add(self.vars[0], self.vars[1])
        return [out]

    def execute_assert_less(self):
        with tf.control_dependencies([tf.compat.v1.assert_less(x=self.vars[0], y=self.vars[1])]):
            out = tf.add(self.vars[0], self.vars[1])
        return [out]

    def execute_assert_less_equal(self):
        with tf.control_dependencies([tf.compat.v1.assert_less_equal(x=self.vars[0], y=self.vars[1])]):
            out = tf.add(self.vars[0], self.vars[1])
        return [out]

    def execute_assert_none_equal(self):
        with tf.control_dependencies([tf.compat.v1.assert_none_equal(x=self.vars[0], y=self.vars[1])]):
            out = tf.add(self.vars[0], self.vars[1])
        return [out]

    def execute_assert_integer(self):
        with tf.control_dependencies([tf.compat.v1.assert_integer(x=self.vars[0])]):
            out = tf.add(self.vars[0], 1)
        return [out]

    def execute_assert_negative(self):
        with tf.control_dependencies([tf.compat.v1.assert_negative(x=self.vars[0])]):
            out = tf.add(self.vars[0], 1)
        return [out]

    def execute_assert_positive(self):
        with tf.control_dependencies([tf.compat.v1.assert_positive(x=self.vars[0])]):
            out = tf.add(self.vars[0], 1)
        return [out]

    def execute_assert_rank(self):
        with tf.control_dependencies([tf.compat.v1.assert_rank(x=self.vars[0], rank=self.vars[1])]):
            out = tf.add(self.vars[0], tf.cast(self.vars[1], self.vars[0].dtype))
        return [out]

    def execute_assert_rank_at_least(self):
        with tf.control_dependencies([tf.compat.v1.assert_rank_at_least(x=self.vars[0], rank=self.vars[1])]):
            out = tf.add(self.vars[0], tf.cast(self.vars[1], self.vars[0].dtype))
        return [out]

    def execute_assert_type(self):
        with tf.control_dependencies([tf.compat.v1.assert_type(tensor=self.vars[0], tf_type=self.op["tf_type"])]):
            out = tf.add(self.vars[0], 1)
        return [out]

    def execute_cond(self):
        def ifTrue():
            return tf.linspace(start=1.0, stop=5.0, num=5)
        def ifFalse():
            return tf.ones(shape=[5], dtype=tf.float32)
        return [tf.cond(pred=self.vars[0], true_fn=ifTrue, false_fn=ifFalse)]

    def execute_case(self):
        input = self.vars[0]
        a = (input <= 1, lambda: input * 1)
        b = (input <= 2, lambda: input * 2)
        c = (input <= 3, lambda: input * 3)
        default = lambda: input * 4
        pairs = [a,b,c]
        return [tf.case(pairs, default)]

    def execute_while1(self):
        # Simple counter loop, there condition is less than self.vars[1]
        def condition(i, j):
            return i < j
        def body(i, j):
            return i+1, j
        loop = tf.while_loop(cond=condition, body=body, loop_vars=(0.0, self.vars[0]))
        return loop

    def execute_while2(self):
        # Loop: keep dividing self.vars[1] by 2 until sum(self.vars[1]) < sum(self.vars[0])
        def condition(x, y):
            return tf.reduce_sum(input_tensor=y) < tf.reduce_sum(input_tensor=x)
        def body(x, y):
            return x, y/2
        loop = tf.while_loop(cond=condition, body=body, loop_vars=(self.vars[0], self.vars[1]))
        return loop

    def execute_sum_dynamic_axis(self):
        if(self.op["axistype"] == "argmin"):
            axis = tf.math.argmin(input=tf.shape(input=self.vars[0]))
        else:
            axis = tf.math.argmax(input=tf.shape(input=self.vars[0]))
        return [tf.reduce_sum(input_tensor=self.vars[0], axis=axis, keepdims=self.op["keepdims"])]

    def execute_tensorarray_getset(self):
        infershape = True
        if("infer_shape" in self.op):
            infershape = self.op["infer_shape"]
        ta = tf.TensorArray(dtype=self.op["dtype"], size=self.op["size"], dynamic_size=self.op["dynamic_size"], tensor_array_name=self.op["tensor_array_name"], element_shape=self.op["element_shape"], infer_shape=infershape)
        n = len(self.vars)
        for i in range(n):
            #Note: on each write, need to use the new/returned TensorArray for all subsequent ops
            ta = ta.write(i, self.vars[i])

        out = []
        for i in range(n):
            out.append(ta.read(i))
        return out

    def execute_tensorarray_size(self):
        infershape = True
        if("infer_shape" in self.op):
            infershape = self.op["infer_shape"]
        ta = tf.TensorArray(dtype=self.op["dtype"], size=self.op["size"], dynamic_size=self.op["dynamic_size"], tensor_array_name=self.op["tensor_array_name"], element_shape=self.op["element_shape"], infer_shape=infershape)
        n = len(self.vars)
        for i in range(n):
            #Note: on each write, need to use the new/returned TensorArray for all subsequent ops
            ta = ta.write(i, self.vars[i])

        return [ta.size()]

    def execute_tensorarray_concat(self):
        infershape = True
        if("infer_shape" in self.op):
            infershape = self.op["infer_shape"]
        ta = tf.TensorArray(dtype=self.op["dtype"], size=self.op["size"], dynamic_size=self.op["dynamic_size"], tensor_array_name=self.op["tensor_array_name"], element_shape=self.op["element_shape"], infer_shape=infershape)
        n = len(self.vars)
        for i in range(n):
            #Note: on each write, need to use the new/returned TensorArray for all subsequent ops
            ta = ta.write(i, self.vars[i])

        return [ta.concat()]

    def execute_tensorarray_stack(self):
        infershape = True
        if("infer_shape" in self.op):
            infershape = self.op["infer_shape"]
        ta = tf.TensorArray(dtype=self.op["dtype"], size=self.op["size"], dynamic_size=self.op["dynamic_size"], tensor_array_name=self.op["tensor_array_name"], element_shape=self.op["element_shape"], infer_shape=infershape)
        n = len(self.vars)
        for i in range(n):
            #Note: on each write, need to use the new/returned TensorArray for all subsequent ops
            ta = ta.write(i, self.vars[i])

        return [ta.stack()]

    def execute_tensorarray_unstack(self):
        #Unstack: create empty tensor array, stack the test array inputs, then unstack them to the TensorArray
        # (then pull them out for testing...)

        infershape = True
        if("infer_shape" in self.op):
            infershape = self.op["infer_shape"]
        ta = tf.TensorArray(dtype=self.op["dtype"], size=self.op["size"], dynamic_size=self.op["dynamic_size"], tensor_array_name=self.op["tensor_array_name"], element_shape=self.op["element_shape"], infer_shape=infershape)
        n = len(self.vars)

        stack = tf.stack(self.vars, axis=0)

        ta = ta.unstack(stack)  #Stack to increase rank by 1 before TensorArray unstack

        n = len(self.vars)
        out = []
        for i in range(n):
            #Note: on each write, need to use the new/returned TensorArray for all subsequent ops
            out.append(ta.read(i))

        return out

    def execute_tensorarray_identity(self):
        infershape = True
        if("infer_shape" in self.op):
            infershape = self.op["infer_shape"]
        ta = tf.TensorArray(dtype=self.op["dtype"], size=self.op["size"], dynamic_size=self.op["dynamic_size"], tensor_array_name=self.op["tensor_array_name"], element_shape=self.op["element_shape"], infer_shape=infershape)
        n = len(self.vars)
        for i in range(n):
            #Note: on each write, need to use the new/returned TensorArray for all subsequent ops
            ta = ta.write(i, self.vars[i])

        ta2 = ta.identity()
        out = []
        for i in range(n):
            out.append(ta2.read(i))
        return out

    def execute_tensorarray_split(self):
        infershape = True
        if("infer_shape" in self.op):
            infershape = self.op["infer_shape"]
        ta = tf.TensorArray(dtype=self.op["dtype"], size=self.op["size"], dynamic_size=self.op["dynamic_size"], tensor_array_name=self.op["tensor_array_name"], element_shape=self.op["element_shape"], infer_shape=infershape)

        ta = ta.split(value=self.vars[0], lengths=self.vars[1])

        n = self.op["varShapes"][1][0]
        out = []
        for i in range(n):
            out.append(ta.read(i))
        return out

    def execute_tensorarray_close(self):
        infershape = True
        if("infer_shape" in self.op):
            infershape = self.op["infer_shape"]
        ta = tf.TensorArray(dtype=self.op["dtype"], size=self.op["size"], dynamic_size=self.op["dynamic_size"], tensor_array_name=self.op["tensor_array_name"], element_shape=self.op["element_shape"], infer_shape=infershape)
        n = len(self.vars)
        for i in range(n):
            #Note: on each write, need to use the new/returned TensorArray for all subsequent ops
            ta = ta.write(i, self.vars[i])

        out = []
        for i in range(n):
            out.append(ta.read(i))

        ta = ta.close()     #Needs to be consumed...
        with tf.control_dependencies([ta]):
            out.append(tf.Variable(tf.ones(shape=[2,2], dtype=tf.float32)))
        return out

    def execute_extractImagePatches(self):
        out = [tf.image.extract_patches(images=self.vars[0], sizes=self.op["ksizes"], strides=self.op["strides"], rates=self.op["rates"], padding=self.op["padding"])]
        return out

    def execute_stopGradient(self):
        temp = tf.tanh(self.vars[0])
        out = [tf.stop_gradient(temp)]
        return out

    def execute_lstmcell(self):
        lstm = tf.compat.v1.nn.rnn_cell.LSTMCell(num_units=self.op["num_units"], use_peepholes=self.op["use_peepholes"], cell_clip=self.op["cell_clip"],
                                       proj_clip=self.op["proj_clip"], forget_bias=self.op["forget_bias"], activation=self.op["activation"])

        initState = None
        if(len(self.vars) > 1):
            initState = [self.vars[1], self.vars[2]]
            if(self.op["static"] == False):
                initState = tf.compat.v1.nn.rnn_cell.LSTMStateTuple(initState[0], initState[1])

        if(self.op["static"] == True):
            x = tf.unstack(self.vars[0], num=self.op["timeSteps"], axis=1)
            outputs, states = tf.compat.v1.nn.static_rnn(lstm, inputs=x, initial_state=initState, dtype=self.op["dtype"])
        else:
            outputs, states = tf.compat.v1.nn.dynamic_rnn(lstm, inputs=self.vars[0], initial_state=initState, dtype=self.op["dtype"], time_major=self.op["time_major"])

        concatOutputs = tf.concat(outputs, axis=0)
        concatStates = tf.concat(states, axis=0)
        return [concatOutputs, concatStates]

    def execute_basicrnncell(self):
        rnn = tf.compat.v1.nn.rnn_cell.BasicRNNCell(num_units=self.op["num_units"], activation=self.op["activation"])

        initState = None
        if(len(self.vars) > 1):
            initState = self.vars[1];

        if(self.op["static"] == True):
            x = tf.unstack(self.vars[0], num=self.op["timeSteps"], axis=1)
            outputs, states = tf.compat.v1.nn.static_rnn(rnn, inputs=x, initial_state=initState, dtype=self.op["dtype"])
        else:
            outputs, states = tf.compat.v1.nn.dynamic_rnn(rnn, inputs=self.vars[0], initial_state=initState, dtype=self.op["dtype"], time_major=self.op["time_major"])

        concatOutputs = tf.concat(outputs, axis=0)
        concatStates = tf.concat(states, axis=0)
        return [concatOutputs, concatStates]

    def execute_basiclstmcell(self):
        rnn = tf.compat.v1.nn.rnn_cell.BasicLSTMCell(num_units=self.op["num_units"], activation=self.op["activation"], forget_bias=self.op["forget_bias"])

        initState = None
        if(len(self.vars) > 1):
            initState = [self.vars[1], self.vars[2]]
            if(self.op["static"] == False):
                initState = tf.compat.v1.nn.rnn_cell.LSTMStateTuple(initState[0], initState[1])

        if(self.op["static"] == True):
            x = tf.unstack(self.vars[0], num=self.op["timeSteps"], axis=1)
            outputs, states = tf.compat.v1.nn.static_rnn(rnn, inputs=x, initial_state=initState, dtype=self.op["dtype"])
        else:
            outputs, states = tf.compat.v1.nn.dynamic_rnn(rnn, inputs=self.vars[0], initial_state=initState, dtype=self.op["dtype"], time_major=self.op["time_major"])

        concatOutputs = tf.concat(outputs, axis=0)
        concatStates = tf.concat(states, axis=0)
        return [concatOutputs, concatStates]

    def execute_grucell(self):
        rnn = tf.compat.v1.nn.rnn_cell.GRUCell(num_units=self.op["num_units"], activation=self.op["activation"])

        initState = None
        if(len(self.vars) > 1):
            initState = self.vars[1]

        if(self.op["static"] == True):
            x = tf.unstack(self.vars[0], num=self.op["timeSteps"], axis=1)
            outputs, states = tf.compat.v1.nn.static_rnn(rnn, inputs=x, initial_state=initState, dtype=self.op["dtype"])
        else:
            outputs, states = tf.compat.v1.nn.dynamic_rnn(rnn, inputs=self.vars[0], initial_state=initState, dtype=self.op["dtype"], time_major=self.op["time_major"])

        concatOutputs = tf.concat(outputs, axis=0)
        concatStates = tf.concat(states, axis=0)
        return [concatOutputs, concatStates]

    def execute_grublockcellv2(self):
        rnn = tf.contrib.rnn.GRUBlockCellV2(num_units=self.op["num_units"])

        initState = None
        if(len(self.vars) > 1):
            initState = self.vars[1]

        if(self.op["static"] == True):
            x = tf.unstack(self.vars[0], num=self.op["timeSteps"], axis=1)
            outputs, states = tf.compat.v1.nn.static_rnn(rnn, inputs=x, initial_state=initState, dtype=self.op["dtype"])
        else:
            outputs, states = tf.compat.v1.nn.dynamic_rnn(rnn, inputs=self.vars[0], initial_state=initState, dtype=self.op["dtype"], time_major=self.op["time_major"])

        concatOutputs = tf.concat(outputs, axis=0)
        concatStates = tf.concat(states, axis=0)
        return [concatOutputs, concatStates]

    def execute_lstmblockcell(self):
        rnn = tf.contrib.rnn.LSTMBlockCell(num_units=self.op["num_units"], forget_bias=self.op["forget_bias"])

        initState = None
        if(len(self.vars) > 1):
            initState = [self.vars[1], self.vars[2]]
            if(self.op["static"] == False):
                initState = tf.compat.v1.nn.rnn_cell.LSTMStateTuple(initState[0], initState[1])

        if(self.op["static"] == True):
            x = tf.unstack(self.vars[0], num=self.op["timeSteps"], axis=1)
            outputs, states = tf.compat.v1.nn.static_rnn(rnn, inputs=x, initial_state=initState, dtype=self.op["dtype"])
        else:
            outputs, states = tf.compat.v1.nn.dynamic_rnn(rnn, inputs=self.vars[0], initial_state=initState, dtype=self.op["dtype"], time_major=self.op["time_major"])

        concatOutputs = tf.concat(outputs, axis=0)
        concatStates = tf.concat(states, axis=0)
        return [concatOutputs, concatStates]

    def execute_srucell(self):
        rnn = tf.contrib.rnn.SRUCell(num_units=self.op["num_units"], activation=self.op["activation"])

        initState = None
        if(len(self.vars) > 1):
            initState = self.vars[1]

        if(self.op["static"] == True):
            x = tf.unstack(self.vars[0], num=self.op["timeSteps"], axis=1)
            outputs, states = tf.compat.v1.nn.static_rnn(rnn, inputs=x, initial_state=initState, dtype=self.op["dtype"])
        else:
            outputs, states = tf.compat.v1.nn.dynamic_rnn(rnn, inputs=self.vars[0], initial_state=initState, dtype=self.op["dtype"], time_major=self.op["time_major"])

        concatOutputs = tf.concat(outputs, axis=0)
        concatStates = tf.concat(states, axis=0)
        return [concatOutputs, concatStates]

    def execute_lstmblockfusedcell(self):
        initState = None
        if(len(self.vars) > 1):
            initState = [self.vars[1], self.vars[2]]
            if(self.op["static"] == False):
                initState = tf.compat.v1.nn.rnn_cell.LSTMStateTuple(initState[0], initState[1])

        rnn = tf.contrib.rnn.LSTMBlockFusedCell(num_units=self.op["num_units"], forget_bias=self.op["forget_bias"], cell_clip=self.op["cell_clip"], use_peephole=self.op["use_peephole"])
        outputs, states = rnn(inputs=self.vars[0], initial_state=initState, dtype=self.op["dtype"])

        concatStates = tf.concat(states, axis=0)
        # print("NAME: ", outputs.name)
        return [outputs, concatStates]

    def execute_bidirectional_basicrnncell(self):
        rnn1 = tf.compat.v1.nn.rnn_cell.BasicRNNCell(num_units=self.op["num_units"], activation=self.op["activation"])
        rnn2 = tf.compat.v1.nn.rnn_cell.BasicRNNCell(num_units=self.op["num_units"], activation=self.op["activation"])

        initState1 = None
        initState2 = None
        if(len(self.vars) > 1):
            initState1 = self.vars[1]
            initState2 = self.vars[2]

        if(self.op["static"] == True):
            x = tf.unstack(self.vars[0], num=self.op["timeSteps"], axis=1)
            outputs, statesFwd, statesBwd = tf.compat.v1.nn.static_bidirectional_rnn(cell_fw=rnn1, cell_bw=rnn2, inputs=x, initial_state_fw=initState1, initial_state_bw=initState2, dtype=self.op["dtype"])
            concatOutputs = tf.concat(outputs, axis=0)
            concatStatesFwd = tf.concat(statesFwd, axis=0)
            concatStatesBwd = tf.concat(statesBwd, axis=0)
            return [concatOutputs, concatStatesFwd, concatStatesBwd]
        else:
            outputs, states = tf.compat.v1.nn.bidirectional_dynamic_rnn(cell_fw=rnn1, cell_bw=rnn2, inputs=self.vars[0], initial_state_fw=initState1, initial_state_bw=initState2, dtype=self.op["dtype"], time_major=self.op["time_major"])
            concatOutputs = tf.concat(outputs, axis=0)
            concatStates = tf.concat(states, axis=0)
            return [concatOutputs, concatStates]

    def execute_timereversed_lstmblockfusedcell(self):
        initState = None
        if(len(self.vars) > 1):
            initState = [self.vars[1], self.vars[2]]
            if(self.op["static"] == False):
                initState = tf.compat.v1.nn.rnn_cell.LSTMStateTuple(initState[0], initState[1])

        rnn = tf.contrib.rnn.LSTMBlockFusedCell(num_units=self.op["num_units"], forget_bias=self.op["forget_bias"], cell_clip=self.op["cell_clip"], use_peephole=self.op["use_peephole"])
        rnn = tf.contrib.rnn.TimeReversedFusedRNN(rnn)
        outputs, states = rnn(inputs=self.vars[0], initial_state=initState, dtype=self.op["dtype"])

        concatStates = tf.concat(states, axis=0)
        return [outputs, concatStates]


    def execute_fused_adaptor_basicrnncell(self):
        initState = None
        if(len(self.vars) > 1):
            initState = self.vars[1]

        rnn = tf.compat.v1.nn.rnn_cell.BasicRNNCell(num_units=self.op["num_units"], activation=self.op["activation"])
        rnn = tf.contrib.rnn.FusedRNNCellAdaptor(cell=rnn, use_dynamic_rnn=self.op["use_dynamic_rnn"])
        outputs, states = rnn(inputs=self.vars[0], initial_state=initState, dtype=self.op["dtype"])

        concatStates = tf.concat(states, axis=0)
        return [outputs, concatStates]

    def execute_stack_bidir_basicrnncell(self):
        rnn1 = []
        rnn2 = []
        for i in range(self.op["size"]):
            rnn1.append(tf.compat.v1.nn.rnn_cell.BasicRNNCell(num_units=self.op["num_units"], activation=self.op["activation"]))
            rnn2.append(tf.compat.v1.nn.rnn_cell.BasicRNNCell(num_units=self.op["num_units"], activation=self.op["activation"]))

        initState1 = None
        initState2 = None
        if(len(self.vars) > 1):
            initState1 = []
            initState2 = []
            for i in range(self.op["size"]):
                initState1.append(self.vars[1 + 2*i])
                initState2.append(self.vars[2 + 2*i])

        if(self.op["static"] == True):
            x = tf.unstack(self.vars[0], num=self.op["timeSteps"], axis=1)
            outputs, statesFwd, statesBwd = tf.contrib.rnn.stack_bidirectional_rnn(cells_fw=rnn1, cells_bw=rnn2, inputs=x, initial_states_fw=initState1, initial_states_bw=initState2, dtype=self.op["dtype"])
        else:
            outputs, statesFwd, statesBwd = tf.contrib.rnn.stack_bidirectional_dynamic_rnn(cells_fw=rnn1, cells_bw=rnn2, inputs=self.vars[0], initial_states_fw=initState1, initial_states_bw=initState2, dtype=self.op["dtype"], time_major=self.op["time_major"])

        concatOutputs = tf.concat(outputs, axis=0)
        concatStatesFwd = tf.concat(statesFwd, axis=0)
        concatStatesBwd = tf.concat(statesBwd, axis=0)
        return [concatOutputs, concatStatesFwd, concatStatesBwd]

    def execute_cast(self):
        out = [tf.cast(self.vars[0], dtype=self.op["dtype"])]
        return out

    def execute_reshape(self):
        return [tf.reshape(self.vars[0], shape=self.op["shape"])]

    def execute_arg_max(self):
        return [tf.argmax(input=self.vars[0], axis=self.op["dimension"])]

    def execute_arg_min(self):
        return [tf.argmin(input=self.vars[0], axis=self.op["dimension"])]

    def execute_assign(self):
        return [tf.compat.v1.assign(ref=self.vars[0], value=self.vars[1])]

    def execute_concat(self):
        return [tf.concat(values=self.vars, axis=self.op["axis"])]

    def execute_expand_dims(self):
        return [tf.expand_dims(input=self.vars[0], axis=self.op["axis"])]

    def execute_fill(self):
        return [tf.fill(dims=self.vars[0], value=self.vars[1])]

    def execute_gather(self):
        return [tf.gather(params=self.vars[0], indices=self.vars[1], axis=self.op["axis"])]

    def execute_ones(self):
        return [tf.ones(shape=self.vars[0], dtype=self.op["dtype"])]

    def execute_ones_like(self):
        return [tf.ones_like(input=self.vars[0])]

    def execute_zeros(self):
        return [tf.zeros(shape=self.vars[0], dtype=self.op["dtype"])]

    def execute_zeros_like(self):
        return [tf.zeros_like(input=self.vars[0], dtype=self.op["dtype"])]

    def execute_range(self):
        return [tf.range(start=self.vars[0], limit=self.vars[1], delta=self.vars[2])]

    def execute_rank(self):
        return [tf.rank(self.vars[0])]

    def execute_realdiv(self):
        return [tf.realdiv(self.vars[0], self.vars[1])]

    def execute_reverse(self):
        return [tf.reverse(tensor=self.vars[0], axis=self.op["axis"])]

    def execute_slice(self):
        return [tf.slice(self.vars[0], begin=self.vars[1], size=self.vars[1])]

    def execute_squeeze(self):
        print("Squeeze shape: ", self.vars[0])
        return [tf.squeeze(input=self.vars[0], axis=self.op["axis"])]

    def execute_strided_slice(self):
        return [tf.strided_slice(self.vars[0], begin=self.op["begin"], end=self.op["end"], strides=self.op["strides"], begin_mask=self.op["begin_mask"], end_mask=self.op["end_mask"])]

    def execute_transpose(self):
        return [tf.transpose(a=self.vars[0], perm=self.op["perm"])]

    def execute_unstack(self):
        self.vars[0] = tf.reshape(self.vars[0], self.op["varShapes"][0])
        return tf.unstack(value=self.vars[0], num=self.op["num"], axis=self.op["axis"])

    def execute_abs(self):
        return [tf.math.abs(self.vars[0])]

    def execute_add(self):
        return [tf.math.add(self.vars[0], self.vars[1])]

    def execute_add_scalar(self):
        return [self.vars[0] + self.op["scalar"]]

    def execute_sub(self):
        return [tf.math.subtract(self.vars[0], self.vars[1])]

    def execute_mul(self):
        return [tf.math.multiply(self.vars[0], self.vars[1])]

    def execute_div(self):
        return [tf.math.divide(self.vars[0], self.vars[1])]

    def execute_div_no_nan(self):
        return [tf.math.divide_no_nan(self.vars[0], self.vars[1])]

    def execute_add_n(self):
        return [tf.math.add_n(self.vars)]

    def execute_cos(self):
        return [tf.math.cos(self.vars[0])]

    def execute_sin(self):
        return [tf.math.sin(self.vars[0])]

    def execute_tan(self):
        return [tf.math.tan(self.vars[0])]

    def execute_cosh(self):
        return [tf.math.cosh(self.vars[0])]

    def execute_acos(self):
        return [tf.math.acos(self.vars[0])]

    def execute_acosh(self):
        return [tf.math.acosh(self.vars[0])]

    def execute_asin(self):
        return [tf.math.asin(self.vars[0])]

    def execute_asinh(self):
        return [tf.math.asinh(self.vars[0])]

    def execute_atan(self):
        return [tf.math.atan(self.vars[0])]

    def execute_atanh(self):
        return [tf.math.atanh(self.vars[0])]

    def execute_ceil(self):
        return [tf.math.ceil(self.vars[0])]

    def execute_count_nonzero(self):
        return [tf.math.count_nonzero(self.vars[0], axis=self.op["axis"])]

    def execute_count_zero(self):
        return [tf.math.count_zero(self.vars[0], axis=self.op["axis"], keep_dims=self.op["keep_dims"])]

    def execute_cumprod(self):
        return [tf.math.cumprod(self.vars[0], axis=self.op["axis"])]

    def execute_cumsum(self):
        return [tf.math.cumsum(self.vars[0], axis=self.op["axis"])]

    def execute_equal(self):
        return [tf.math.equal(self.vars[0], self.vars[1])]

    def execute_exp(self):
        return [tf.math.exp(self.vars[0])]

    def execute_floor(self):
        return [tf.math.floor(self.vars[0])]

    def execute_floordiv(self):
        return [tf.math.floordiv(self.vars[0], self.vars[1])]

    def execute_log(self):
        return [tf.math.log(self.vars[0])]

    def execute_log_sigmoid(self):
        return [tf.math.log_sigmoid(self.vars[0])]

    def execute_sigmoid(self):
        return [tf.math.sigmoid(self.vars[0])]

    def execute_negative(self):
        return [tf.math.negative(self.vars[0])]

    def execute_reciprocal(self):
        return [tf.math.reciprocal(self.vars[0])]

    def execute_sign(self):
        return [tf.math.sign(self.vars[0])]

    def execute_softplus(self):
        return [tf.math.softplus(self.vars[0])]

    def execute_sqrt(self):
        return [tf.math.sqrt(self.vars[0])]

    def execute_square(self):
        return [tf.math.square(self.vars[0])]

    def execute_rsqrt(self):
        return [tf.math.rsqrt(self.vars[0])]

    def execute_greater(self):
        return [tf.math.greater(self.vars[0], self.vars[1])]

    def execute_greater_equal(self):
        return [tf.math.greater_equal(self.vars[0], self.vars[1])]

    def execute_less(self):
        return [tf.math.less(self.vars[0], self.vars[1])]

    def execute_less_equal(self):
        return [tf.math.less_equal(self.vars[0], self.vars[1])]

    def execute_not_equal(self):
        return [tf.math.not_equal(self.vars[0], self.vars[1])]

    def execute_truediv(self):
        return [tf.math.truediv(self.vars[0], self.vars[1])]

    def execute_zero_fraction(self):
        return [tf.math.zero_fraction(self.vars[0])]

    def execute_round(self):
        return [tf.math.round(self.vars[0])]

    def execute_maximum(self):
        return [tf.math.maximum(self.vars[0], self.vars[1])]

    def execute_minimum(self):
        return [tf.math.minimum(self.vars[0], self.vars[1])]

    def execute_pow(self):
        return [tf.math.pow(self.vars[0], self.vars[1])]

    def execute_logical_and(self):
        return [tf.math.logical_and(self.vars[0], self.vars[1])]

    def execute_logical_or(self):
        return [tf.math.logical_or(self.vars[0], self.vars[1])]

    def execute_logical_xor(self):
        return [tf.math.logical_xor(self.vars[0], self.vars[1])]

    def execute_logical_not(self):
        return [tf.math.logical_not(self.vars[0])]

    def execute_diag_part(self):
        return [tf.linalg.tensor_diag_part(self.vars[0])]

    def execute_fake_quant_with_min_max_vars(self):
        return [tf.quantization.fake_quant_with_min_max_vars(inputs=self.vars[0], min=self.vars[1], max=self.vars[2], num_bits=self.op["num_bits"], narrow_range=self.op["narrow_range"])]

    def execute_fake_quant_with_min_max_args(self):
        return [tf.quantization.fake_quant_with_min_max_args(inputs=self.vars[0], min=self.op["min"], max=self.op["max"], num_bits=self.op["num_bits"], narrow_range=self.op["narrow_range"])]

    def execute_fake_quant_with_min_max_vars_per_channel(self):
        return [tf.quantization.fake_quant_with_min_max_vars_per_channel(inputs=self.vars[0], min=self.vars[1], max=self.vars[2], num_bits=self.op["num_bits"], narrow_range=self.op["narrow_range"])]

    def execute_check_numerics(self):
        return [tf.debugging.check_numerics(tensor=self.vars[0], message=self.op["message"])]

    def execute_adjust_contrast(self):
        return [tf.image.adjust_contrast(self.vars[0],self.op["contrast_factor"])]

    def execute_adjust_contrast_v2(self):
        return [tf.compat.v2.image.adjust_contrast(self.vars[0],self.vars[1])]

    def execute_adjust_hue(self):
        return [tf.image.adjust_hue(self.vars[0],self.op["delta"])]

    def execute_strings_split(self):
        print("strings.split input: ", self.vars[0])
        out = tf.strings.split(self.vars[0], sep=self.op["split"]).to_sparse()
        print("strings.split output: ", out)
        return [out]

    def execute_bitcast(self):
        return [tf.bitcast(self.vars[0], self.op["output"])]

    def execute_bitcast_float64(self):
        z = tf.bitcast(self.vars[0], tf.float64)
        return [tf.bitcast(z, self.op["output"])]

    def execute_bitwise_and(self):
        return [tf.bitwise.bitwise_and(self.vars[0], self.vars[1])]

    def execute_bitwise_or(self):
        return [tf.bitwise.bitwise_or(self.vars[0], self.vars[1])]

    def execute_bitwise_xor(self):
        return [tf.bitwise.bitwise_xor(self.vars[0], self.vars[1])]

    def execute_left_shift(self):
        return [tf.bitwise.left_shift(self.vars[0], self.vars[1])]

    def execute_right_shift(self):
        return [tf.bitwise.right_shift(self.vars[0], self.vars[1])]

    def execute_crop_and_resize(self):
        return [tf.image.crop_and_resize(image = self.vars[0], boxes = self.vars[1], crop_size = self.vars[2], box_indices = self.vars[3],\
                                         method = self.op["method"], extrapolation_value = self.op["ext_value"])]

    def execute_random_crop(self):
        return [tf.image.random_crop(self.vars[0], self.vars[1])]

    def execute_draw_bounding_boxes(self):
        return [tf.image.draw_bounding_boxes(images = self.vars[0], boxes = self.vars[1], colors = self.vars[2])]

    def execute_resize_bilinear(self):
        return [tf.image.resize(images = self.vars[0], size = self.vars[1], \
                method=tf.image.ResizeMethod.BILINEAR, half_pixel_centers = self.op["half_pixel_centers"])]

    def execute_resize_nearest_neighbor(self):
        return [tf.image.resize(images = self.vars[0], size = self.vars[1], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)]

    def execute_resize_bicubic(self):
        return [tf.image.resize(images=self.vars[0], size=self.vars[1], \
                                        method=tf.image.ResizeMethod.BICUBIC, \
                                        half_pixel_centers=self.op["half_pixel_centers"])]

    def execute_resize_area(self):
        return [tf.image.resize(images=self.vars[0], size=self.vars[1], \
                                        method=tf.image.ResizeMethod.AREA)]

    def execute_non_max_suppression(self):
        iou_threshold = 0.5
        score_threshold = 0.5
        if("iou_threshold" in self.op):
            iou_threshold = self.op["iou_threshold"]
        if("score_threshold" in self.op):
            score_threshold = self.op["score_threshold"]

        return [tf.image.non_max_suppression(boxes =
                                             self.vars[0], scores = self.vars[1], max_output_size = self.vars[2],
                                             iou_threshold=iou_threshold, score_threshold=score_threshold)]

    def execute_non_max_suppression_v2(self):
        return [tf.image.non_max_suppression(self.vars[0], self.vars[1], self.vars[2])]

    def execute_dropout(self):
        return [tf.nn.dropout(x = self.vars[0], rate = self.op["rate"], seed = self.op.get("seed", None), noise_shape = self.op.get("noise_shape", None))]

    def execute_is_non_decreasing(self):
        return [tf.math.is_non_decreasing(x = self.vars[0])]

    def execute_is_strictly_increasing(self):
        return [tf.math.is_strictly_increasing(x = self.vars[0])]

    def execute_leaky_relu(self):
        return [tf.nn.leaky_relu(features = self.vars[0], alpha=self.op["alpha"])]

    def execute_log_softmax(self):
        return [tf.nn.log_softmax(logits = self.vars[0], axis = self.op["axis"])]

    def execute_max(self):
        return [tf.math.maximum(self.vars[0], self.vars[1])]

    def execute_min(self):
        return [tf.math.minimum(self.vars[0], self.vars[1])]

    def execute_mod(self):
        return [tf.math.mod(self.vars[0], self.vars[1])]

    def execute_mul(self):
        return [tf.math.mul(self.vars[0], self.vars[1])]

    def execute_betainc(self):
        return [tf.math.betainc(self.vars[0], self.vars[1], self.vars[2])]

    def execute_fused_batch_norm(self):
        return tf.compat.v1.nn.fused_batch_norm(x= self.vars[0], scale = self.vars[1], offset = self.vars[2], mean = None, variance = None, epsilon = self.op["epsilon"],\
                                data_format = self.op["data_format"], is_training = True)

    def execute_matrix_band_part(self):
        return [tf.linalg.band_part(input = self.vars[0], num_lower = self.op["num_lower"], num_upper = self.op["num_upper"])]

    def execute_toggle_bits(self):
        return [tf.bitwise.invert(self.vars[0])]

    def execute_polygamma(self):
        return [tf.math.polygamma(self.vars[0], self.vars[1])]

    def execute_lgamma(self):
        return [tf.math.lgamma(self.vars[0])]

    def execute_roll(self):
        return [tf.roll(self.vars[0], self.op["shift"], self.op["axis"])]

    def execute_lu(self):
        output_idx_type = tf.int32
        if ("output_idx_type" in self.op):
            output_idx_type = self.op["output_idx_type"]
        return tf.linalg.lu(self.vars[0], output_idx_type)

    def execute_triangular_solve(self):
        return [tf.linalg.triangular_solve(self.vars[0], self.vars[1], self.op["lower"], self.op["adjoint"])]

    def execute_linear_solve(self):
        return [tf.linalg.solve(self.vars[0], self.vars[1], self.op["adjoint"])]

    def execute_lstsq(self):
        return [tf.linalg.lstsq(self.vars[0], self.vars[1], l2_regularizer = self.op["l2_regularizer"], fast = self.op["fast"])]

    def execute_compare_and_bitpack(self):
        return [tf.raw_ops.CompareAndBitpack(input=self.vars[0], threshold = self.op["threshold"])]

    def execute_Conv3DBackpropInputV2(self):
        return [tf.raw_ops.Conv3DBackpropInputV2(input_sizes=self.vars[0],filter=self.vars[1], out_backprop=self.vars[2], strides=self.op["strides"], padding=self.op["padding"], dilations=self.op["dilations"])]

    def execute_empty(self):
        return [tf.raw_ops.Empty(shape = self.vars[0], dtype = self.op["dtype"])]

    def execute_deep_copy(self):
        return [tf.raw_ops.DeepCopy(x = self.vars[0])]

    def execute_ones_like(self):
        return [tf.ones_like(tensor = self.vars[0])]

    def execute_random_gamma(self):
         return [tf.random.gamma(
                     shape = self.vars[0] ,
                     alpha = self.op["alpha"],
                     dtype = self.op["dtype"],
                     seed = self.op["seed"],
                 )]


    def execute_random_poisson(self):
         return [tf.random.poisson(shape = self.vars[0], lam = self.op["lam"], dtype=self.op["dtype"])]

    def execute_random_poisson_v2(self):
         return [tf.raw_ops.RandomPoissonV2(shape = self.vars[0], rate = self.op["rate"], dtype=self.op["dtype"])]

    def execute_random_shuffle(self):
         return [tf.random.shuffle(
                     value = self.vars[0],
                     seed=None,
                 )]

    def execute_random_normal(self):
         return [tf.random.normal(
                     shape = self.vars[0],
                     mean=self.op["mean"],
                     stddev=self.op["stddev"],
                     dtype=self.op["dtype"],
                     seed=self.op["seed"],
                 )]

    def execute_random_uniform(self):
         return [tf.random.uniform(
                     shape = self.vars[0],
                     minval=self.op["minval"],
                     maxval=self.op["maxval"],
                     dtype=self.op["dtype"],
                     seed=self.op["seed"],
                 )]

    def execute_unsorted_segment_mean(self):
         return [tf.math.unsorted_segment_mean(
                     data = self.vars[0],
                     segment_ids = self.op["segment_ids"],
                     num_segments = self.op["num_segments"],
                )]
    def execute_unsorted_segment_sqrt_n(self):
         return [tf.math.unsorted_segment_sqrt_n(
                     data = self.vars[0],
                     segment_ids = self.op["segment_ids"],
                     num_segments = self.op["num_segments"],
                )]

    def execute_div(self):
         return [tf.raw_ops.Div(
                     x = self.vars[0],
                     y = self.vars[1]
                 )]
