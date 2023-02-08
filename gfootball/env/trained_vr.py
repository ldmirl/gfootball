'''
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import warnings
warnings.filterwarnings('ignore',category=FutureWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)
'''

import tensorflow.compat.v1 as tf
import numpy as np
from tensorflow.python.keras.layers import Conv2D, Conv2DTranspose, Input, Flatten, Dense, Lambda, Reshape, \
    TimeDistributed, MaxPooling2D, LSTM
from tensorflow.python.keras.models import Model
import gc

'''
tf.logging.set_verbosity(tf.logging.ERROR)
'''



def vrnn_prediction(input):
    tf.reset_default_graph()

    tf.disable_eager_execution()

    batch_size = 1
    in_timesteps = range(0, 3)
    # out_timesteps = range(1, 6)
    lstm_units = 1024
    feature_vector = 1024
    latent_dim = 256

    # placeholders to hold each frame
    x_ = tf.placeholder("float", shape=(None, len(in_timesteps), 64, 64, 3))
    # y_ = tf.placeholder("float", shape=(None, len(out_timesteps), 64, 64, 3))
    l_ = tf.placeholder("float", shape=(None, 1))

    # encoder
    encoder_conv1_w = tf.get_variable("encoder_conv1_w", shape=[7, 7, 3, 16])
    encoder_conv2_w = tf.get_variable("encoder_conv2_w", shape=[5, 5, 16, 32])
    encoder_conv3_w = tf.get_variable("encoder_conv3_w", shape=[5, 5, 32, 48])
    encoder_conv4_w = tf.get_variable("encoder_conv4_w", shape=[3, 3, 48, 64])

    encoder_conv1_b = tf.get_variable("encoder_conv1_b", shape=[16])
    encoder_conv2_b = tf.get_variable("encoder_conv2_b", shape=[32])
    encoder_conv3_b = tf.get_variable("encoder_conv3_b", shape=[48])
    encoder_conv4_b = tf.get_variable("encoder_conv4_b", shape=[64])

    def cross_entropy_2(y_prediction, y):
        prediction_loss = y * tf.log(1e-7 + y_prediction) + (1 - y) * tf.log(1e-7 + 1 - y_prediction)
        return tf.reduce_sum(prediction_loss, axis=[1, 2, 3]) / 64 / 64 / 3

    def encoder(x):
        out = tf.nn.conv2d(input=x, filter=encoder_conv1_w, strides=[1, 2, 2, 1], padding='SAME') + encoder_conv1_b
        out = tf.nn.relu(out)
        out = tf.nn.conv2d(input=out, filter=encoder_conv2_w, strides=[1, 2, 2, 1], padding='SAME') + encoder_conv2_b
        out = tf.nn.relu(out)
        out = tf.nn.conv2d(input=out, filter=encoder_conv3_w, strides=[1, 2, 2, 1], padding='SAME') + encoder_conv3_b
        out = tf.nn.relu(out)
        out = tf.nn.conv2d(input=out, filter=encoder_conv4_w, strides=[1, 2, 2, 1], padding='SAME') + encoder_conv4_b
        out = tf.nn.relu(out)
        out = tf.reshape(out, shape=[-1, 4 * 4 * 64])
        return out

    # decoder
    decoder_conv1_w = tf.get_variable("decoder_conv1_w", shape=[3, 3, 48, 64])
    decoder_conv2_w = tf.get_variable("decoder_conv2_w", shape=[5, 5, 32, 48])
    decoder_conv3_w = tf.get_variable("decoder_conv3_w", shape=[5, 5, 16, 32])
    decoder_conv4_w = tf.get_variable("decoder_conv4_w", shape=[7, 7, 3, 16])

    decoder_conv1_b = tf.get_variable("decoder_conv1_b", shape=[48])
    decoder_conv2_b = tf.get_variable("decoder_conv2_b", shape=[32])
    decoder_conv3_b = tf.get_variable("decoder_conv3_b", shape=[16])
    decoder_conv4_b = tf.get_variable("decoder_conv4_b", shape=[3])

    def decoder(x):
        out = tf.reshape(x, shape=[-1, 4, 4, 64])
        out = tf.nn.conv2d_transpose(out, filter=decoder_conv1_w, strides=[1, 2, 2, 1],
                                     output_shape=[batch_size, 8, 8, 48],
                                     padding='SAME') + decoder_conv1_b
        out = tf.nn.relu(out)
        out = tf.nn.conv2d_transpose(out, filter=decoder_conv2_w, strides=[1, 2, 2, 1],
                                     output_shape=[batch_size, 16, 16, 32], padding='SAME') + decoder_conv2_b
        out = tf.nn.relu(out)
        out = tf.nn.conv2d_transpose(out, filter=decoder_conv3_w, strides=[1, 2, 2, 1],
                                     output_shape=[batch_size, 32, 32, 16], padding='SAME') + decoder_conv3_b
        out = tf.nn.relu(out)
        out = tf.nn.conv2d_transpose(out, filter=decoder_conv4_w, strides=[1, 2, 2, 1],
                                     output_shape=[batch_size, 64, 64, 3], padding='SAME') + decoder_conv4_b
        out = tf.nn.sigmoid(out)
        return out

    # f_posterior
    f_posterior_fc1_w = tf.get_variable("phi_enc_fc1_w", shape=[feature_vector + lstm_units, latent_dim])
    f_posterior_fc2_w = tf.get_variable("phi_enc_fc2_w", shape=[latent_dim, latent_dim])

    f_posterior_fc1_b = tf.get_variable("phi_enc_fc1_b", shape=[latent_dim])
    f_posterior_fc2_b = tf.get_variable("phi_enc_fc2_b", shape=[latent_dim])

    f_posterior_mu_w = tf.get_variable("phi_enc_mu_w", shape=[latent_dim, latent_dim])
    f_posterior_mu_b = tf.get_variable("phi_enc_mu_b", shape=[latent_dim])

    f_posterior_sigma_w = tf.get_variable("phi_enc_sigma_w", shape=[latent_dim, latent_dim])
    f_posterior_sigma_b = tf.get_variable("phi_enc_sigma_b", shape=[latent_dim])

    def f_posterior(out):
        out = tf.matmul(out, f_posterior_fc1_w) + f_posterior_fc1_b
        out = tf.nn.relu(out)
        out = tf.matmul(out, f_posterior_fc2_w) + f_posterior_fc2_b
        out = tf.nn.relu(out)

        out_mu = tf.matmul(out, f_posterior_mu_w) + f_posterior_mu_b
        out_std = tf.nn.softplus(tf.matmul(out, f_posterior_sigma_w) + f_posterior_sigma_b) + 1e-5

        return out_mu, out_std

    # f_decoder
    f_decoder_fc1_w = tf.get_variable("phi_dec_fc1_w", shape=[latent_dim + lstm_units, feature_vector])
    f_decoder_fc2_w = tf.get_variable("phi_dec_fc2_w", shape=[feature_vector, feature_vector])

    f_decoder_fc1_b = tf.get_variable("phi_dec_fc1_b", shape=[feature_vector])
    f_decoder_fc2_b = tf.get_variable("phi_dec_fc2_b", shape=[feature_vector])

    def f_decoder(out):
        out = tf.matmul(out, f_decoder_fc1_w) + f_decoder_fc1_b
        out = tf.nn.relu(out)
        out = tf.matmul(out, f_decoder_fc2_w) + f_decoder_fc2_b
        out = tf.nn.relu(out)

        return out

    # f_z
    f_z_fc1_w = tf.get_variable("phi_z_fc1_w", shape=[latent_dim, latent_dim])
    f_z_fc2_w = tf.get_variable("phi_z_fc2_w", shape=[latent_dim, latent_dim])

    f_z_fc1_b = tf.get_variable("phi_z_fc1_b", shape=[latent_dim])
    f_z_fc2_b = tf.get_variable("phi_z_fc2_b", shape=[latent_dim])

    def f_z(out):
        out = tf.matmul(out, f_z_fc1_w) + f_z_fc1_b
        out = tf.nn.relu(out)
        out = tf.matmul(out, f_z_fc2_w) + f_z_fc2_b
        out = tf.nn.relu(out)
        return out

    # f_prior
    f_prior_fc1_w = tf.get_variable("phi_prior_fc1_w", shape=[lstm_units, latent_dim])
    f_prior_fc2_w = tf.get_variable("phi_prior_fc2_w", shape=[latent_dim, latent_dim])

    f_prior_fc1_b = tf.get_variable("phi_prior_fc1_b", shape=[latent_dim])
    f_prior_fc2_b = tf.get_variable("phi_prior_fc2_b", shape=[latent_dim])

    f_prior_mu_w = tf.get_variable("phi_prior_mu_w", shape=[latent_dim, latent_dim])
    f_prior_std_w = tf.get_variable("phi_prior_std_w", shape=[latent_dim, latent_dim])

    f_prior_mu_b = tf.get_variable("phi_prior_mu_b", shape=[latent_dim])
    f_prior_std_b = tf.get_variable("phi_prior_std_b", shape=[latent_dim])

    def f_prior(out):
        out = tf.matmul(out, f_prior_fc1_w) + f_prior_fc1_b
        out = tf.nn.relu(out)
        out = tf.matmul(out, f_prior_fc2_w) + f_prior_fc2_b
        out = tf.nn.relu(out)

        out_mu = tf.matmul(out, f_prior_mu_w) + f_prior_mu_b
        out_std = tf.nn.softplus(tf.matmul(out, f_prior_std_w) + f_prior_std_b) + 1e-5

        return out_mu, out_std

    def tf_kl_gaussgauss(mu_1, sigma_1, mu_2, sigma_2):
        return tf.reduce_sum(
            tf.log(sigma_2) - tf.log(sigma_1) + (sigma_1 ** 2 + (mu_1 - mu_2) ** 2) / (2 * ((sigma_2) ** 2)) - 0.5,
            axis=1)

    def cross_entropy(y_prediction, y):
        prediction_loss = y * tf.log(1e-7 + y_prediction) + (1 - y) * tf.log(1e-7 + 1 - y_prediction)
        return -tf.reduce_sum(prediction_loss, axis=[1, 2, 3])

    def batch_data(source, target, label, batch_size):
        # Shuffle data
        shuffle_indices = np.random.permutation(np.arange(len(target)))
        source = source[shuffle_indices]
        target = target[shuffle_indices]
        label = label[shuffle_indices]

        for batch_i in range(0, len(source) // batch_size):
            start_i = batch_i * batch_size
            source_batch = source[start_i:start_i + batch_size]
            target_batch = target[start_i:start_i + batch_size]
            label_batch = label[start_i:start_i + batch_size]

            yield np.array(source_batch), np.array(target_batch), np.array(label_batch)

    def cond_to_init_state_dense_1(cond):
        latent_cond = tf.keras.layers.Dense(units=lstm_units)(cond)
        return latent_cond

    def extract_cond_and_h(cond_and_h):
        new_h = tf.keras.layers.Dense(units=lstm_units)(cond_and_h)
        return new_h

    lstm = tf.nn.rnn_cell.LSTMCell(num_units=lstm_units, state_is_tuple=True)
    lstm_state = lstm.zero_state(batch_size, tf.float32)

    pred_list3 = []
    for i in range(0, 5):

        if i == 0:
            lstm_state = tuple(tf.nn.rnn_cell.LSTMStateTuple(lstm_state[0], cond_to_init_state_dense_1(l_)))
        else:
            lstm_state = tuple(tf.nn.rnn_cell.LSTMStateTuple(lstm_state[0], extract_cond_and_h(
                tf.concat([cond_to_init_state_dense_1(l_), lstm_state[1]], axis=1))))

        # Prediction module
        if i <= 2:
            encoder_out_pred3 = encoder(tf.divide(x=x_[:, i, :, :, :], y=255.0))
        else:
            encoder_out_pred3 = encoder(y_hat_pred3)
        # encoder_out_pred3 = encoder(tf.divide(x=x_[:, i, :, :, :], y=255.0))
        # compute prior
        f_prior_out_mu_pred3, f_prior_out_sigma_pred3 = f_prior(lstm_state[1])

        z_pred3 = f_prior_out_mu_pred3 + f_prior_out_sigma_pred3 * tf.random_normal(shape=[256], mean=0.0, stddev=1.0)
        f_z_out_pred3 = f_z(z_pred3)


        # decode [lstm, latent information]
        f_decoder_out_pred3 = f_decoder(tf.concat(values=(lstm_state[1], f_z_out_pred3), axis=1))
        y_hat_pred3 = decoder(f_decoder_out_pred3)

        # append output
        if i>2:
            pred_list3.append(y_hat_pred3)

        lstm_out, lstm_state = lstm(inputs=tf.concat(values=(encoder_out_pred3, f_z_out_pred3), axis=1),
                                    state=lstm_state)


    pred_matrix3 = tf.transpose(tf.stack(pred_list3), [1, 0, 2, 3, 4])
    # reconstruction_loss = cross_entropy_2(y_hat_pred3, tf.divide(x=x_[:, 5, :, :, :], y=255.0))

    sess = tf.Session()
    # sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    saver.restore(sess, "/home/aarongu/Desktop/TrainEpoch-810/epoch-810")

    # input shape as (1,5,64,64,3)
    l = np.zeros((1, 1))

    wtf1 = sess.run(pred_matrix3, feed_dict={x_: input[:, :, :, :, :], l_: l[:, :]})

    wtf2 = sess.run(pred_matrix3, feed_dict={x_: np.concatenate([input[:, 2:, :, :, :], wtf1[:, :, :, :, :]], axis=1), l_: l[:, :]})

    wtf3 = sess.run(pred_matrix3, feed_dict={x_: np.concatenate([wtf1[:, 1:, :, :, :], wtf2[:, :, :, :, :]], axis=1), l_: l[:, :]})

#    wtf4 = sess.run(pred_matrix3, feed_dict={x_: np.concatenate([wtf2[:, 1:, :, :, :], wtf3[:, :, :, :, :]], axis=1), l_: l[:, :]})

#    wtf5 = sess.run(pred_matrix3, feed_dict={x_: np.concatenate([wtf3[:, 1:, :, :, :], wtf4[:, :, :, :, :]], axis=1), l_: l[:, :]})

#    wtf6 = sess.run(pred_matrix3, feed_dict={x_: np.concatenate([wtf4[:, 1:, :, :, :], wtf5[:, :, :, :, :]], axis=1), l_: l[:, :]})

    fuck = np.concatenate([wtf1, wtf2, wtf3], axis=1)
#    fuck = np.concatenate([wtf1, wtf2, wtf3, wtf4, wtf5, wtf6],axis=1)
    
    fuck = fuck[0,:,:,:,:]

    sess.close()

    tf.keras.backend.clear_session()
    gc.collect()
    return fuck
