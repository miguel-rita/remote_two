import numpy as np
import pandas as pd
import tensorflow as tf
import tqdm

def main():

    # Placeholder for raw time series data
    mnist_imgs = tf.placeholder(dtype=tf.uint8, shape=[None, None, None])

    # Get image dimensions assuming dim 0 to be batch dim
    img_h = tf.shape(mnist_imgs)[1]
    img_w = tf.shape(mnist_imgs)[2]

    #processed_mnist_imgs = grid_op(img_h, img_w)
    vertex_tex_data = generate_geometry_texture_data_op(mnist_imgs)
    proj_vertex_tex_data = perspective_op(vertex_tex_data, render_params)

    # Get raw mnist images
    single_img = np.reshape(np.arange(0,6), [2,3])
    dummy_img = np.tile(np.expand_dims(single_img, axis=0), [2, 1, 1])

    # Run session
    with tf.Session() as sess:

        #sess = tf_debug.LocalCLIDebugWrapperSession(sess)

        print(sess.run(proj_vertex_tex_data, feed_dict={mnist_imgs: dummy_img, render_params: np.array([1,0.5])}))
        sess.close()

main()