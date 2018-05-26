#-*- coding: utf-8 -*-
#header comes here
#based on code from ****

from __future__ import division
import os
import time
import tensorflow as tf
import numpy as np
import consts

from ops import *
from utils import *

class baseline(object):
    model_name = "baseline"     # name for checkpoint

    def __init__(self, sess, epoch, batch_size, z_dim, dataset_name, checkpoint_dir, result_dir, log_dir):
        self.sess = sess
        self.dataset_name = dataset_name
        self.checkpoint_dir = checkpoint_dir
        self.result_dir = result_dir
        self.log_dir = log_dir
        self.epoch = epoch
        self.batch_size = batch_size

        if dataset_name == 'mnist' or dataset_name == 'fashion-mnist':
            # parameters
            # self.input_height = 28
            # self.input_width = 28
            # self.output_height = 28
            # self.output_width = 28
            self.embed_size = 512
            self.hidden_size = self.embed_size
            self.dropout_rate = 0.5

            self.z_dim = z_dim         # dimension of noise-vector
            self.c_dim = 1

            # WGAN_GP parameter
            self.lambd = 0.25       # The higher value, the more stable, but the slower convergence
            self.disc_iters = 1     # The number of critic iterations for one-step of generator

            # train
            self.learning_rate = 0.0002
            self.beta1 = 0.5

            # test
            self.sample_num = 64  # number of generated images to be saved

            # load mnist
            # self.data_X, self.data_y = load_mnist(self.dataset_name)

            # get number of batches for a single epoch
            self.num_batches = len(self.data_X) // self.batch_size
        else:
            raise NotImplementedError

    def discriminator(self, x, seq_len, is_training=True, reuse=False):
        # Network Architecture is exactly same as in infoGAN (https://arxiv.org/abs/1606.03657)
        # Architecture : (64)4c2s-(128)4c2s_BL-FC1024_BL-FC1_S
        with tf.variable_scope("discriminator", reuse=reuse):
            # net = lrelu(conv2d(x, 64, 4, 4, 2, 2, name='d_conv1'))
            # net = lrelu(bn(conv2d(net, 128, 4, 4, 2, 2, name='d_conv2'), is_training=is_training, scope='d_bn2'))
            # net = tf.reshape(net, [self.batch_size, -1])
            # net = lrelu(bn(linear(net, 1024, scope='d_fc3'), is_training=is_training, scope='d_bn3'))
            # out_logit = linear(net, 1, scope='d_fc4')
            # out = tf.nn.sigmoid(out_logit)
            #define embedding matrix
            d_embeddings = tf.get_variable(name='d_embeddings', shape=[consts.TAG_NUM, self.embed_size], dtype=tf.float32,
                                         initializer=tf.contrib.layers.xavier_initilizer()) #dims=[tag_num,embed_size]

            # instance GRU and the hidden state vector
            with tf.variable_scope("d_gru", reuse=reuse):
                GRU = tf.contrib.rnn.GRUCell(self.hidden_size)
                h = tf.get_variable(name='h', shape=self.hidden_size, dtype=tf.float32,initializer=np.zeros([self.hidden_size])) #dims=hidden_size

                # get embeddings of the input data
                input_embeddings = tf.matmul(x, d_embeddings, name='input_embeddings')  #dims=[bs,max_len+2,embed_size]

                for i in range(seq_len):
                    if i > 0:
                        tf.get_variable_scope().reuse_variables()
                    o_t, h_t = GRU(input_embeddings[:, i, :], h)
                final_state_drop = tf.nn.dropout(o_t, self.dropout_rate, name='final_state_drop')
                out_logit = linear(final_state_drop, 1, scope='d_fc')
                out = tf.nn.sigmoid(out_logit)

            return out, out_logit, final_state_drop

    def generator(self, z, data, mask, seq_len, is_training=True, reuse=False):
        # Network Architecture is exactly same as in infoGAN (https://arxiv.org/abs/1606.03657)
        # Architecture : FC1024_BR-FC7x7x128_BR-(64)4dc2s_BR-(1)4dc2s_S
        with tf.variable_scope("generator", reuse=reuse):

            data_len = data.shape[1] #max_len+2

            #define embedding matrix
            g_embeddings = tf.get_variable(name='g_embeddings', shape=[consts.TAG_NUM, self.embed_size], dtype=tf.float32,
                                         initializer=tf.contrib.layers.xavier_initilizer()) #dims=[tag_num,embed_size]

            #get embeddings of the real data
            data_embeddings = tf.nn.embedding_lookup(g_embeddings, data, name='data_embeddings') #dims=[bs,max_len+2,embed_size]

            #get one hot vectors of the real data
            data_one_hot = tf.one_hot(data, consts.TAG_NUM, axis=2, dtype=tf.float32, name='data_one_hot') #dims=[bs,max_len+2,tag_num]

            #init placeholders to output probabilities of the network and the final output
            output_prob = tf.constant(value=np.zeros([self.batch_size, data_len, consts.TAG_NUM]), dtype=tf.float32,
                                      name='output_prob') #dims=[bs,max_len+2,tag_num]
            out = tf.constant(value=np.zeros([self.batch_size, data_len, consts.TAG_NUM]), dtype=tf.float32,
                              name='out')  # dims=[bs,max_len+2,tag_num]

            #instance GRU and the hidden state vector
            with tf.variable_scope("g_gru", reuse=reuse):
                GRU = tf.contrib.rnn.GRUCell(self.hidden_size)
                h = tf.get_variable(name='h', shape=self.hidden_size, dtype=tf.float32, initializer=z) #dims=hidden_size

            for i in range(seq_len):
                if i > 0:
                    tf.get_variable_scope().reuse_variables()
                    output_embeddings = tf.matmul(output_prob[:, i-1, :], g_embeddings, name='output_embeddings') #dims=[bs,embed_size]
                x_data = tf.multiply(1-mask[:, i], data_embeddings[:, i, :]) #dims=[bs,embed_size]
                x_gen = tf.multiply(mask[:, i], output_embeddings) #dims=[bs,embed_size]
                x = x_data + x_gen #dims=[bs,embed_size]
                o_t, h_t = GRU(x, h)
                o_drop_t = tf.nn.dropout(o_t, self.dropout_rate, name='o_drop_t')
                g_logits = linear(o_drop_t, consts.TAG_NUM, scope='g_fc')
                if i < data_len - 1: #max_len+1
                    output_prob[:, i+1, :] = tf.nn.softmax(g_logits)

                out_data = tf.multiply(1-mask[:, i], data_one_hot)
                out_gen = tf.multiply(mask[:, i], output_prob)
                out[:, i, :] = out_data + out_gen

            return out

    def build_model(self):
        # some parameters
        image_dims = [self.input_height, self.input_width, self.c_dim]
        bs = self.batch_size

        """ Graph Input """
        # images
        self.inputs = tf.placeholder(tf.float32, [bs] + image_dims, name='real_images')

        # noises
        self.z = tf.placeholder(tf.float32, [bs, self.z_dim], name='z')

        """ Loss Function """

        # output of D for real images
        D_real, D_real_logits, _ = self.discriminator(self.inputs, is_training=True, reuse=False)

        # output of D for fake images
        G = self.generator(self.z, is_training=True, reuse=False)
        D_fake, D_fake_logits, _ = self.discriminator(G, is_training=True, reuse=True)

        # get loss for discriminator
        d_loss_real = - tf.reduce_mean(D_real_logits)
        d_loss_fake = tf.reduce_mean(D_fake_logits)

        self.d_loss = d_loss_real + d_loss_fake

        # get loss for generator
        self.g_loss = - d_loss_fake

        """ Gradient Penalty """
        # This is borrowed from https://github.com/kodalinaveen3/DRAGAN/blob/master/DRAGAN.ipynb
        alpha = tf.random_uniform(shape=self.inputs.get_shape(), minval=0.,maxval=1.)
        differences = G - self.inputs # This is different from MAGAN
        interpolates = self.inputs + (alpha * differences)
        _,D_inter,_=self.discriminator(interpolates, is_training=True, reuse=True)
        gradients = tf.gradients(D_inter, [interpolates])[0]
        slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), reduction_indices=[1]))
        gradient_penalty = tf.reduce_mean((slopes - 1.) ** 2)
        self.d_loss += self.lambd * gradient_penalty

        """ Training """
        # divide trainable variables into a group for D and a group for G
        t_vars = tf.trainable_variables()
        d_vars = [var for var in t_vars if 'd_' in var.name]
        g_vars = [var for var in t_vars if 'g_' in var.name]

        # optimizers
        with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
            self.d_optim = tf.train.AdamOptimizer(self.learning_rate, beta1=self.beta1) \
                      .minimize(self.d_loss, var_list=d_vars)
            self.g_optim = tf.train.AdamOptimizer(self.learning_rate*5, beta1=self.beta1) \
                      .minimize(self.g_loss, var_list=g_vars)

        """" Testing """
        # for test
        self.fake_images = self.generator(self.z, is_training=False, reuse=True)

        """ Summary """
        d_loss_real_sum = tf.summary.scalar("d_loss_real", d_loss_real)
        d_loss_fake_sum = tf.summary.scalar("d_loss_fake", d_loss_fake)
        d_loss_sum = tf.summary.scalar("d_loss", self.d_loss)
        g_loss_sum = tf.summary.scalar("g_loss", self.g_loss)

        # final summary operations
        self.g_sum = tf.summary.merge([d_loss_fake_sum, g_loss_sum])
        self.d_sum = tf.summary.merge([d_loss_real_sum, d_loss_sum])

    def train(self):

        # initialize all variables
        tf.global_variables_initializer().run()

        # graph inputs for visualize training results
        self.sample_z = np.random.uniform(-1, 1, size=(self.batch_size , self.z_dim))

        # saver to save model
        self.saver = tf.train.Saver()

        # summary writer
        self.writer = tf.summary.FileWriter(self.log_dir + '/' + self.model_name, self.sess.graph)

        # restore check-point if it exits
        could_load, checkpoint_counter = self.load(self.checkpoint_dir)
        if could_load:
            start_epoch = (int)(checkpoint_counter / self.num_batches)
            start_batch_id = checkpoint_counter - start_epoch * self.num_batches
            counter = checkpoint_counter
            print(" [*] Load SUCCESS")
        else:
            start_epoch = 0
            start_batch_id = 0
            counter = 1
            print(" [!] Load failed...")

        # loop for epoch
        start_time = time.time()
        for epoch in range(start_epoch, self.epoch):

            # get batch data
            for idx in range(start_batch_id, self.num_batches):
                batch_images = self.data_X[idx*self.batch_size:(idx+1)*self.batch_size]
                batch_z = np.random.uniform(-1, 1, [self.batch_size, self.z_dim]).astype(np.float32)

                # update D network
                _, summary_str, d_loss = self.sess.run([self.d_optim, self.d_sum, self.d_loss],
                                               feed_dict={self.inputs: batch_images, self.z: batch_z})
                self.writer.add_summary(summary_str, counter)

                # update G network
                if (counter-1) % self.disc_iters == 0:
                    batch_z = np.random.uniform(-1, 1, [self.batch_size, self.z_dim]).astype(np.float32)
                    _, summary_str, g_loss = self.sess.run([self.g_optim, self.g_sum, self.g_loss], feed_dict={self.z: batch_z})
                    self.writer.add_summary(summary_str, counter)

                counter += 1

                # display training status
                print("Epoch: [%2d] [%4d/%4d] time: %4.4f, d_loss: %.8f, g_loss: %.8f" \
                      % (epoch, idx, self.num_batches, time.time() - start_time, d_loss, g_loss))

                # save training results for every 300 steps
                if np.mod(counter, 300) == 0:
                    samples = self.sess.run(self.fake_images,
                                            feed_dict={self.z: self.sample_z})
                    tot_num_samples = min(self.sample_num, self.batch_size)
                    manifold_h = int(np.floor(np.sqrt(tot_num_samples)))
                    manifold_w = int(np.floor(np.sqrt(tot_num_samples)))
                    save_images(samples[:manifold_h * manifold_w, :, :, :], [manifold_h, manifold_w],
                                './' + check_folder(self.result_dir + '/' + self.model_dir) + '/' + self.model_name + '_train_{:02d}_{:04d}.png'.format(
                                    epoch, idx))

            # After an epoch, start_batch_id is set to zero
            # non-zero value is only for the first epoch after loading pre-trained model
            start_batch_id = 0

            # save model
            self.save(self.checkpoint_dir, counter)

            # show temporal results
            self.visualize_results(epoch)

        # save model for final step
        self.save(self.checkpoint_dir, counter)

    def visualize_results(self, epoch):
        tot_num_samples = min(self.sample_num, self.batch_size)
        image_frame_dim = int(np.floor(np.sqrt(tot_num_samples)))

        """ random condition, random noise """

        z_sample = np.random.uniform(-1, 1, size=(self.batch_size, self.z_dim))

        samples = self.sess.run(self.fake_images, feed_dict={self.z: z_sample})

        save_images(samples[:image_frame_dim * image_frame_dim, :, :, :], [image_frame_dim, image_frame_dim],
                    check_folder(self.result_dir + '/' + self.model_dir) + '/' + self.model_name + '_epoch%03d' % epoch + '_test_all_classes.png')

    @property
    def model_dir(self):
        return "{}_{}_{}_{}".format(
            self.model_name, self.dataset_name,
            self.batch_size, self.z_dim)

    def save(self, checkpoint_dir, step):
        checkpoint_dir = os.path.join(checkpoint_dir, self.model_dir, self.model_name)

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        self.saver.save(self.sess,os.path.join(checkpoint_dir, self.model_name+'.model'), global_step=step)

    def load(self, checkpoint_dir):
        import re
        print(" [*] Reading checkpoints...")
        checkpoint_dir = os.path.join(checkpoint_dir, self.model_dir, self.model_name)

        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
            counter = int(next(re.finditer("(\d+)(?!.*\d)",ckpt_name)).group(0))
            print(" [*] Success to read {}".format(ckpt_name))
            return True, counter
        else:
            print(" [*] Failed to find a checkpoint")
            return False, 0