import tensorflow as tf

hello = tf.constant('bla')

sess = tf.Session()
print(sess.run(hello))

tf.test.is_gpu_available(
	cuda_only=False,
	min_cuda_compute_capability=None
)
