
import tensorflow as tf
import numpy as np

from tensorflow.python.client import timeline

def simple_profiling():

    tf.reset_default_graph()
    input = tf.placeholder(dtype=tf.float32, shape=[None, 784], name="in")
    w = tf.Variable(np.random.uniform(size=[784,10]), dtype=tf.float32)
    b = tf.Variable(np.random.uniform(size=[1,10]), dtype=tf.float32)
    z = tf.matmul(input, w) + b
    softmax = tf.nn.softmax(z)
    # softmax = tf.nn.softmax(softmax)
    # softmax = tf.nn.softmax(softmax)
    sess = tf.Session()
    inArr = np.random.uniform(size=[1,784]).astype('f')

    init = tf.global_variables_initializer()
    sess.run(init)
    # out = sess.run([softmax], feed_dict={"in:0": inArr})
    # print(out)

    #Warmup
    for i in range(5):
        sess.run([softmax], feed_dict={"in:0": inArr})

    # add additional options to trace the session execution
    options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
    for i in range(5):
        #Unfortunately, no way to combine multiple runs into one file :(
        # https://github.com/tensorflow/tensorflow/issues/3489
        run_metadata = tf.RunMetadata()
        sess.run([softmax], feed_dict={"in:0": inArr}, options=options, run_metadata=run_metadata)

        # Create the Timeline object, and write it to a json file
        fetched_timeline = timeline.Timeline(run_metadata.step_stats)
        chrome_trace = fetched_timeline.generate_chrome_trace_format()
        with open('/ImportTests/profiles/profile' + str(i) + '.json', 'w') as f:
            f.write(chrome_trace)





if __name__ == '__main__':
    simple_profiling()