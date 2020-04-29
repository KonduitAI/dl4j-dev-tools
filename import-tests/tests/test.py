import tensorflow as tf

# RECONSTRUCTION OF REQUESTED OP, THIS FILE WILL BE REMOVED BEFORE MERGE
# https://gist.github.com/AlexDBlack/89c57e7916a7c22a1d92db650625c0bf#file-gistfile1-txt-L553-L600
# Known issue ops that are excluded in the top TFGraphTestAllsameDiff
# Bincount
# ConfusionMatrix
# FusedBatchNormV2/V3
# LogMatrixDeterminant
# MatrixBandPart

# Issued ops
# Conv2DTranspose - already in TFGraphTestAllsameDiff
# Conv3DBackpropInput - Op Conv3DBackpropInput is not available in GraphDef version 134. It has been removed in version 10. Use Conv3DBackpropInputV2.
# Copy  - AttributeError: module 'tensorflow._api.v1.raw_ops' has no attribute 'Copy'
# CopyHost - AttributeError: module 'tensorflow._api.v1.raw_ops' has no attribute 'CopyHost'
# Mish - addons https://www.tensorflow.org/addons/api_docs/python/tfa/activations/mish
# which is not compatible with 1.15
#  TF 1.15

# Conv3DBackpropInputV2
input = [1, 3, 3, 3, 2]
filter = tf.random.uniform(shape = [1, 1, 1, 2, 1], dtype=tf.dtypes.float32)
out_backprop = tf.random.uniform(shape = [1, 3, 3, 3, 1], dtype=tf.dtypes.float32)
tf.compat.v1.raw_ops.Conv3DBackpropInputV2(input_sizes=input,filter=filter, out_backprop=out_backprop, strides=[1, 1, 1, 1, 1], padding="SAME", dilations=[1, 1, 1, 1, 1],
                                              name=None)

# CompareAndBitpack
CaB_input = tf.random.uniform(shape = [1,2,3,4,5,6,7,8], dtype=tf.dtypes.float32)
out = tf.raw_ops.CompareAndBitpack(input = CaB_input, threshold =6)

# DeepCopy
tf.raw_ops.DeepCopy(x = tf.random.uniform(shape = [1,2,3,4,5,6,7,8], dtype=tf.dtypes.float32))

# Dropout
tf.nn.dropout(
     x=tf.random.uniform(shape = [1,2,3,4], dtype=tf.dtypes.float32),
     rate=0.2
 )

# Div
tf.raw_ops.Div(
   x=tf.random.uniform(shape = [1,2,3], dtype=tf.dtypes.float32),
   y=tf.random.uniform(shape = [1,2,3], dtype=tf.dtypes.float32)
)

# Empty
tf.raw_ops.Empty(shape = [1,2,3], dtype=tf.dtypes.float32)


# IsNonDecreasing
tf.math.is_non_decreasing(
      tf.random.uniform(shape = [1,2,3], dtype=tf.dtypes.float32),
      name=None
  )

# LeakyRelu
tf.nn.leaky_relu(
     tf.random.uniform(shape = [1,2,3], dtype=tf.dtypes.float32),
     alpha=0.2,
     name=None
 )

# Lgamma
tf.math.lgamma(
    tf.random.uniform(shape = [1,2,3], dtype=tf.dtypes.float32),
    name=None
)

# MaxPoolWithArgMax
tf.nn.max_pool_with_argmax(
    input=tf.random.uniform(shape = [1,16,16,1], dtype=tf.dtypes.float32),
    ksize = [1,1,1,1],
    strides = [1,1,1,1],
    padding="SAME",
    data_format='NHWC',

)

# NOTE this is from addons https://www.tensorflow.org/addons/api_docs/python/tfa/activations/mish
# which is not compatible with 1.15
# from tensorflow_addons.activations.mish import mish
# tfa.activations.mish(tf.random.uniform(shape = [1,2,3], dtype=tf.dtypes.float32))

# MOD
tf.raw_ops.Mod(
x=tf.random.uniform(shape = [1,2,3]), y=tf.random.uniform(shape = [1,2,3]))