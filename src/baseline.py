#-*- coding: utf-8 -*-
#header comes here
#based on code from ****

from __future__ import division
import os
import time
import tensorflow as tf
import numpy as np


#local
import consts
import data
from ops import *
from utils import *
from runtime_process import *

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
        self.debug_mode = False

        if dataset_name == 'sanity':

            self.dataset_h5_path = '../data/sanity_seq-64_dict-ascii_classes-1.h5'
            self.dataset_json_path = '../data/sanity_seq-64_dict-ascii_classes-1.json'

        elif dataset_name == 'short':

            self.dataset_h5_path = '../data/short_seq-64_dict-ascii_classes-1.h5'
            self.dataset_json_path = '../data/short_seq-64_dict-ascii_classes-1.json'

        elif dataset_name == 'news_en_only':

            self.dataset_h5_path = '../data/news_en_only_seq-64_dict-ascii_classes-1.h5'
            self.dataset_json_path = '../data/news_en_only_seq-64_dict-ascii_classes-1.json'

        else:
            raise NotImplementedError

        # arch parameters
        self.embed_size = 512
        self.hidden_size = self.embed_size
        self.dropout_rate_for_train = 0.5
        self.z_dim = self.hidden_size
        self.seq_len = 32

        # WGAN_GP parameter
        self.lambd = 0.25  # The higher value, the more stable, but the slower convergence
        self.disc_iters = 1  # The number of critic iterations for one-step of generator

        # train
        self.learning_rate = 0.0002
        self.beta1 = 0.5

        # test
        self.sample_num = 64  # number of generated sents to be saved

        # instance data handler
        self.data_handler = Runtime_data_handler(h5_path=self.dataset_h5_path,
                                                 seq_len=self.seq_len,
                                                 max_len=self.seq_len,
                                                 teacher_helping_mode='th_extended',
                                                 use_var_len=True,
                                                 batch_size=self.batch_size,
                                                 use_labels=False)

        self.betches_per_iter = 2
        self.iters_per_epoch = self.data_handler.get_num_batches_per_epoch() // self.betches_per_iter

    def discriminator(self, x, max_len, is_training=True, reuse=False):
        # Network Architecture is exactly same as in infoGAN (https://arxiv.org/abs/1606.03657)
        # Architecture : (64)4c2s-(128)4c2s_BL-FC1024_BL-FC1_S
        with tf.variable_scope("discriminator", reuse=reuse):
            #define embedding matrix
            d_embeddings = tf.get_variable(name='d_embeddings', shape=[data.TAG_NUM, self.embed_size], dtype=tf.float32,
                                         initializer=tf.contrib.layers.xavier_initializer()) #dims=[tag_num,embed_size]

            # instance GRU and the hidden state vector
            # with tf.variable_scope("d_gru", reuse=reuse):
            h = tf.constant(np.zeros([self.batch_size, self.hidden_size], dtype=np.float32), name='d_hidden',dtype=tf.float32)  # dims=hidden_size
            o_t = tf.constant(np.zeros([1, self.hidden_size], dtype=np.float32), name='d_o_t',dtype=tf.float32)  # dims=hidden_size
            # get embeddings of the input data
            input_embeddings = tf.reshape(tf.matmul(tf.reshape(x, [-1, data.TAG_NUM]), d_embeddings, name='input_embeddings'),
                                                    [self.batch_size, self.seq_len+2, self.embed_size])  #dims=[bs,max_len+2,embed_size]

            with tf.variable_scope("gru", reuse=reuse):
                for i in range(self.seq_len + 2):
                    if i > 0:
                        tf.get_variable_scope().reuse_variables()
                    h, o_t = tf.cond(pred=tf.less(tf.constant(i),max_len+2),
                                     true_fn= lambda: gru(h,input_embeddings[:,i,:],scope='d_gru') ,
                                     false_fn= lambda: (h, o_t) )
                    # h, o_t = gru(h,input_embeddings[:,i,:],scope='d_gru')
            final_state_drop = tf.nn.dropout(o_t, self.dropout_ph, name='final_state_drop')
            out_logit = linear(final_state_drop, 1, scope='d_fc')
            out = tf.nn.sigmoid(out_logit)

            return out, out_logit, final_state_drop

    def generator(self, z, data_input, mask, max_len, is_training=True, reuse=False):
        # Network Architecture is exactly same as in infoGAN (https://arxiv.org/abs/1606.03657)
        # Architecture : FC1024_BR-FC7x7x128_BR-(64)4dc2s_BR-(1)4dc2s_S
        with tf.variable_scope("generator", reuse=reuse):

            data_len = data_input.shape[1] #max_len+2

            #define embedding matrix
            g_embeddings = tf.get_variable(name='g_embeddings', shape=[data.TAG_NUM, self.embed_size], dtype=tf.float32,
                                         initializer=tf.contrib.layers.xavier_initializer()) #dims=[tag_num,embed_size]

            #get embeddings of the real data
            data_embeddings = tf.nn.embedding_lookup(g_embeddings, data_input, name='data_embeddings') #dims=[bs,max_len+2,embed_size]

            #get one hot vectors of the real data
            data_one_hot = tf.one_hot(data_input, data.TAG_NUM, axis=2, dtype=tf.float32, name='data_one_hot') #dims=[bs,max_len+2,tag_num]

            #init placeholders to output probabilities of the network and the final output
            output_prob = tf.constant(value=np.zeros([self.batch_size, 1, data.TAG_NUM]), dtype=tf.float32,
                                      name='output_prob') #dims=[bs,1,tag_num]
            g_probs = tf.constant(value=np.zeros([self.batch_size, data.TAG_NUM]), dtype=tf.float32,
                                      name='g_probs') #dims=[bs,1,tag_num]

            #instance GRU and the hidden state vector
            # with tf.variable_scope("g_gru", reuse=reuse):
            #     GRU = tf.contrib.rnn.GRUCell(self.hidden_size)
            h = tf.identity(z,name='g_hidden')  # dims=hidden_size; initialized with noise

            for i in range(self.seq_len + 2):
                if i > 0:
                    tf.get_variable_scope().reuse_variables()
                    output_embeddings = tf.matmul(output_prob[:, i-1, :], g_embeddings, name='output_embeddings') #dims=[bs,embed_size]
                else:
                    output_embeddings = tf.zeros([self.batch_size, self.embed_size])
                x_data = tf.multiply(tf.tile(tf.expand_dims(tf.cast(1-mask[:, i], dtype=tf.float32),axis=1),
                                             [1,self.embed_size]), data_embeddings[:, i, :]) #dims=[bs,embed_size]
                x_gen = tf.multiply(tf.tile(tf.expand_dims(tf.cast(mask[:, i], dtype=tf.float32),axis=1),
                                            [1, self.embed_size]), output_embeddings) #dims=[bs,embed_size]
                x = x_data + x_gen #dims=[bs,embed_size]

                g_probs = tf.cond(pred=tf.less(tf.constant(i), max_len + 2),
                                 true_fn=lambda: tf.nn.softmax(linear(tf.nn.dropout((gru(x, h, scope='g_gru'))[1], self.dropout_ph, name='o_drop_t'), data.TAG_NUM, scope='g_fc'),axis=1),
                                 false_fn=lambda: g_probs)

                # h, o_t = gru(x, h, scope='g_gru')
                # o_drop_t = tf.nn.dropout(o_t, self.dropout_ph, name='o_drop_t') # dims=[bs,hidden_size]
                # g_logits = linear(o_drop_t, data.TAG_NUM, scope='g_fc') # dims=[bs,tag_num]
                # g_probs = tf.nn.softmax(g_logits,axis=1) # dims=[bs,tag_num]

                if i < data_len - 1: #max_len+1
                    output_prob = tf.concat([output_prob, tf.expand_dims(g_probs, axis=1)], axis=1)

            out_data = tf.multiply(tf.tile(tf.expand_dims(tf.cast(1-mask, dtype=tf.float32),axis=2),
                                           [1, 1, data.TAG_NUM]), data_one_hot)
            out_gen = tf.multiply(tf.tile(tf.expand_dims(tf.cast(mask, dtype=tf.float32),axis=2),
                                          [1, 1, data.TAG_NUM]), output_prob)
            out = out_data + out_gen

            debug_dict = {'out_data': out_data, 'out_gen': out_gen, 'out': out, 'output_prob': output_prob}

            return out, debug_dict

    def build_model(self):
        # some parameters
        # image_dims = [self.input_height, self.input_width, self.c_dim]
        # bs = self.batch_size

        """ Graph Input """

        # model inputs
        self.max_len = tf.placeholder(tf.int32, name='max_len')
        self.dropout_ph = tf.placeholder(tf.float32, name='dropout_ph')

        # generator inputs
        self.generator_input = tf.placeholder(shape=[self.batch_size, self.seq_len+2], dtype=tf.int32, name='generator_input')
        self.generator_mask = tf.placeholder(shape=[self.batch_size, self.seq_len+2], dtype=tf.int32, name='generator_mask')
        self.z = tf.placeholder(tf.float32, shape=[self.batch_size, self.hidden_size], name='z')
        self.max_len = tf.placeholder(tf.int32, name='max_len')

        # discrimnator input
        self.discrimnator_input = tf.placeholder(shape=[self.batch_size, self.seq_len+2], dtype=tf.int32, name='discrimnator_input')
        self.discrimnator_input_one_hot = tf.one_hot(self.discrimnator_input, dtype=tf.float32, depth=data.TAG_NUM)

        """ Loss Function """
        # output of D for real images
        D_real, D_real_logits, _ = self.discriminator(self.discrimnator_input_one_hot,self.max_len, is_training=True, reuse=False)

        # output of D for fake images
        G, self.g_debug_dict = self.generator(self.z, self.generator_input, self.generator_mask, self.max_len, is_training=True, reuse=False)
        D_fake, D_fake_logits, _ = self.discriminator(G, self.max_len, is_training=True, reuse=True)

        # get loss for discriminator
        d_loss_real = - tf.reduce_mean(D_real_logits)
        d_loss_fake = tf.reduce_mean(D_fake_logits)

        self.d_loss = d_loss_real + d_loss_fake

        # get loss for generator
        self.g_loss = - d_loss_fake

        """ Gradient Penalty """
        # This is borrowed from https://github.com/kodalinaveen3/DRAGAN/blob/master/DRAGAN.ipynb
        alpha = tf.random_uniform(shape=self.discrimnator_input_one_hot.get_shape(), minval=0.,maxval=1.)
        differences = G - self.discrimnator_input_one_hot # This is different from MAGAN
        interpolates = self.discrimnator_input_one_hot + (alpha * differences)
        _,D_inter,_=self.discriminator(interpolates, self.max_len, is_training=True, reuse=True)
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
        self.fake_sents, _ = self.generator(self.z, self.generator_input, self.generator_mask, max_len=self.max_len, is_training=False, reuse=True)

        """ Summary """
        d_loss_real_sum = tf.summary.scalar("d_loss_real", d_loss_real)
        d_loss_fake_sum = tf.summary.scalar("d_loss_fake", d_loss_fake)
        d_loss_sum = tf.summary.scalar("d_loss", self.d_loss)
        g_loss_sum = tf.summary.scalar("g_loss", self.g_loss)
        g_vars_sum = [tf.summary.histogram(var.name, var) for var in g_vars]
        d_vars_sum = [tf.summary.histogram(var.name, var) for var in d_vars]
        g_output_sum = tf.summary.histogram('G_output',G)
        d_output_fake_sum = tf.summary.histogram('D_fake', D_fake)
        d_output_real_sum = tf.summary.histogram('D_real', D_real)

        # final summary operations
        self.g_sum = tf.summary.merge([d_loss_fake_sum, g_loss_sum , g_output_sum] + g_vars_sum)
        self.d_sum = tf.summary.merge([d_loss_real_sum, d_loss_sum, d_output_fake_sum, d_output_real_sum] + d_vars_sum)


    def train(self):

        # initialize all variables
        tf.global_variables_initializer().run()

        # graph inputs for visualize training results
        self.sample_z = np.random.normal(0, 1, size=(self.batch_size, self.hidden_size))

        # saver to save model
        self.saver = tf.train.Saver()

        # summary writer
        self.writer = tf.summary.FileWriter(self.log_dir + '/' + self.model_name, self.sess.graph)


        # restore check-point if it exits
        could_load, checkpoint_counter = self.load(self.checkpoint_dir)
        if could_load:
            start_epoch = (int)(checkpoint_counter / self.iters_per_epoch)
            start_batch_id = checkpoint_counter - start_epoch * self.iters_per_epoch
            counter = checkpoint_counter
            print(" [*] Load SUCCESS")
        else:
            start_epoch = 0
            start_batch_id = 0
            counter = 1
            print(" [!] Load failed...")

        max_len_list = sorted([1, 2, 4, 8, 16, 32] * (self.epoch//6))

        # loop for epoch
        start_time = time.time()
        for epoch in range(start_epoch, self.epoch):

            start_epoch_time = time.time()
            cur_max_len = max_len_list[epoch]

            self.data_handler.epoch_start(start_batch_id = start_batch_id,
                                          max_len=cur_max_len,
                                          teacher_helping_mode='th_extended')

            print("===starting epoch [%0d] with [max_len=%0d]==="%(epoch,cur_max_len))

            # get batch data
            for idx in range(start_batch_id, self.iters_per_epoch):

                batch_sents_generator, batch_mask_generator = self.data_handler.get_batch(create_mask=True)
                batch_sents_discriminator = self.data_handler.get_batch(create_mask=False)
                batch_z = np.random.normal(0, 1, [self.batch_size, self.z_dim]).astype(np.float32)

                # update D network
                _, summary_str, d_loss = self.sess.run([self.d_optim, self.d_sum, self.d_loss],
                                               feed_dict={self.discrimnator_input: batch_sents_discriminator,
                                                          self.generator_input: batch_sents_generator,
                                                          self.generator_mask: batch_mask_generator,
                                                          self.z: batch_z,
                                                          self.max_len: cur_max_len,
                                                          self.dropout_ph: self.dropout_rate_for_train})
                self.writer.add_summary(summary_str, counter)

                # update G network
                if (counter-1) % self.disc_iters == 0:
                    batch_z = np.random.normal(0, 1, [self.batch_size, self.z_dim]).astype(np.float32)
                    _, summary_str, g_loss = self.sess.run([self.g_optim, self.g_sum, self.g_loss],
                                                           feed_dict={self.z: batch_z,
                                                                      self.generator_mask: batch_mask_generator,
                                                                      self.generator_input: batch_sents_generator,
                                                                      self.max_len: cur_max_len,
                                                                      self.dropout_ph: self.dropout_rate_for_train})
                    self.writer.add_summary(summary_str, counter)

                #for debug
                if self.debug_mode and idx % 10 == 0:
                    batch_z = np.random.normal(0, 1, [self.batch_size, self.z_dim]).astype(np.float32)
                    out_data, out_gen, out, output_prob = self.sess.run([
                                                            self.g_debug_dict['out_data'],
                                                            self.g_debug_dict['out_gen'],
                                                            self.g_debug_dict['out'],
                                                            self.g_debug_dict['output_prob']],
                                                           feed_dict={self.z: batch_z,
                                                                      self.generator_mask: batch_mask_generator,
                                                                      self.generator_input: batch_sents_generator,
                                                                      self.max_len: cur_max_len,
                                                                      self.dropout_ph: self.dropout_rate_for_train})
                    print('debug\n')

                counter += 1


                # display training status
                print("\rEpoch: [%2d] [%4d/%4d] time: %4.4f, d_loss: %.8f, g_loss: %.8f" \
                      % (epoch, idx, self.iters_per_epoch, time.time() - start_time, d_loss, g_loss),end='')

                # save training results for every 1000 steps
                if np.mod(counter, 1000) == 0:
                    self.visualize_results(counter, max_len=32,description='step')

            # After an epoch, start_batch_id is set to zero
            # non-zero value is only for the first epoch after loading pre-trained model
            start_batch_id = 0
            print('')

            # save model
            self.save(self.checkpoint_dir, counter)

            # show temporal results
            self.visualize_results(epoch,max_len=32) # for debug - max len remains constant and maximal

            self.data_handler.epoch_end()

            # print("\rEpoch SUMMARY: [%2d] [%4d/%4d] time: %4.4f, d_loss: %.8f, g_loss: %.8f" \
            #       % (epoch, idx, self.iters_per_epoch, time.time() - start_epoch_time, d_loss, g_loss))

        # save model for final step
        self.save(self.checkpoint_dir, counter)

    def visualize_results(self, epoch, max_len, description='epoch'):
        tot_num_samples = min(self.sample_num, self.batch_size)

        """ random condition, random noise """

        z_sample = np.random.normal(0, 1, size=(self.batch_size, self.z_dim))
        batch_empty_mask, batch_empty_data = data.create_empty_data(sent_num=self.batch_size,seq_len=self.seq_len,max_len=max_len)

        samples = self.sess.run(self.fake_sents, feed_dict={self.z: z_sample,
                                                            self.generator_mask: batch_empty_mask,
                                                            self.generator_input: batch_empty_data,
                                                            self.max_len: max_len,
                                                            self.dropout_ph: 1.})
        tags = np.argmax(samples, axis=2).astype(int)
        sentences = []

        for sample_idx in range(tot_num_samples):
            char_idx = 1
            sent = ''
            while tags[sample_idx, char_idx].astype(int) != data.END_TAG:
                sent += data.tag2char(int(tags[sample_idx, char_idx].astype(int)))
                char_idx += 1
            sentences.append(sent)

        text = '\n'.join(sentences)

        if not os.path.isdir('results'):
            os.mkdir('results')
        if not os.path.isdir(os.path.join(self.result_dir, self.model_name)):
            os.mkdir(os.path.join(self.result_dir, self.model_name))

        save_path = os.path.join(self.result_dir, self.model_name, 'results_' + description + '_' + str(epoch) + '.txt')
        with open(save_path, 'w') as file:
            file.write(text)

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