#-*- coding: utf-8 -*-
#header comes here
#based on code from ****

from __future__ import division
import os
import time
import tensorflow as tf
import numpy as np
import consts
import data

from ops import *
from utils import *

class acgan_based(object):
    model_name = "acgan_based"     # name for checkpoint

    def __init__(self, sess, epoch, batch_size, z_dim, dataset_name, checkpoint_dir, result_dir, log_dir):
        self.sess = sess
        self.dataset_name = dataset_name
        self.checkpoint_dir = checkpoint_dir
        self.result_dir = result_dir
        self.log_dir = log_dir
        self.epoch = epoch
        self.batch_size = batch_size

        if dataset_name == 'sanity_data':
            # parameters
            self.embed_size = 512
            self.class_embed_size = 32
            self.class_num = 3
            self.hidden_size = self.embed_size
            self.dropout_rate = 0.5
            self.z_dim = self.hidden_size
            self.seq_len = 32

            # WGAN_GP parameter
            self.lambd = 0.25       # The higher value, the more stable, but the slower convergence
            self.disc_iters = 1     # The number of critic iterations for one-step of generator

            # train
            self.learning_rate = 0.0002
            self.beta1 = 0.5

            # test
            self.sample_num = 64  # number of generated sents to be saved

            # load sanity_data
            # self.data_X, self.data_y = load_mnist(self.dataset_name)

            # get number of batches for a single epoch
            # self.num_batches = len(self.data_X) // self.batch_size
        else:
            raise NotImplementedError

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
            final_state_drop = tf.nn.dropout(o_t, self.dropout_rate, name='final_state_drop')
            out_for_classifier = final_state_drop
            out_logit = linear(final_state_drop, 1, scope='d_fc')
            out = tf.nn.sigmoid(out_logit)

            return out, out_logit, out_for_classifier

    def generator(self, z, data_input, mask, max_len, class_input, is_training=True, reuse=False):
        # Network Architecture is exactly same as in infoGAN (https://arxiv.org/abs/1606.03657)
        # Architecture : FC1024_BR-FC7x7x128_BR-(64)4dc2s_BR-(1)4dc2s_S
        with tf.variable_scope("generator", reuse=reuse):

            data_len = data_input.shape[1] #max_len+2

            #define embedding matrix
            g_embeddings = tf.get_variable(name='g_embeddings', shape=[data.TAG_NUM, self.embed_size], dtype=tf.float32,
                                         initializer=tf.contrib.layers.xavier_initializer()) #dims=[tag_num,embed_size]
            #define class embeddings
            g_class_embed_matrix = tf.get_variable(name='g_class_embed_matrix', shape=[self.class_num, self.class_embed_size],
                                                 dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer()) #dims=[class_num,class_embed_size]

            # extract class embeddings
            g_class_embed = tf.nn.embedding_lookup(g_class_embed_matrix, class_input, name='g_class_embed')#dims=[bs,class_embed_size]
            g_class_embed_tiled = tf.tile(tf.expand_dims(g_class_embed, axis=1), [1, self.seq_len+2, 1], name='g_class_embed_tiled') #dims=[bs,max_len+2,class_embed_size]

            #get embeddings of the real data
            data_embeddings = tf.nn.embedding_lookup(g_embeddings, data_input, name='data_embeddings') #dims=[bs,max_len+2,embed_size]
            data_embeddings = tf.concat([data_embeddings, g_class_embed_tiled], axis=2, name='data_embeddings_concat') #dims=[bs,max_len+2,embed_size+class_embed_size]

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

                # concatanate class embeddings to output_embeddings
                output_embeddings = tf.concat([output_embeddings, g_class_embed], axis=1, name='output_embeddings_concat')  # dims=[bs,embed_size+class_embed_size]
                x_data = tf.multiply(tf.tile(tf.expand_dims(tf.cast(1-mask[:, i], dtype=tf.float32),axis=1),
                                             [1,self.embed_size+self.class_embed_size]), data_embeddings[:, i, :]) #dims=[bs,embed_size+class_embed_size]
                x_gen = tf.multiply(tf.tile(tf.expand_dims(tf.cast(mask[:, i], dtype=tf.float32),axis=1),
                                            [1, self.embed_size+self.class_embed_size]), output_embeddings) #dims=[bs,embed_size+class_embed_size]
                x = x_data + x_gen #dims=[bs,embed_size+class_embed_size]

                g_probs = tf.cond(pred=tf.less(tf.constant(i), max_len + 2),
                                 true_fn=lambda: tf.nn.softmax(linear(tf.nn.dropout((gru(x, h, scope='g_gru'))[1], self.dropout_rate, name='o_drop_t'), data.TAG_NUM, scope='g_fc'),axis=1),
                                 false_fn=lambda: g_probs)

                # h, o_t = gru(x, h, scope='g_gru')
                # o_drop_t = tf.nn.dropout(o_t, self.dropout_rate, name='o_drop_t') # dims=[bs,hidden_size]
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

    def classifier(self, x, is_training=True, reuse=False):
        with tf.variable_scope('classifier', reuse=reuse):
            net = lrelu(bn(linear(x, self.hidden_size, scope='c_fc1'), is_training=is_training, scope='c_bn1'))
            out_logit = linear(net, self.class_num, scope='c_fc2')
            out = tf.nn.softmax(out_logit)

        return out, out_logit

    def build_model(self):
        """ Graph Input """
        # generator inputs
        self.generator_input = tf.placeholder(shape=[self.batch_size, self.seq_len+2], dtype=tf.int32, name='generator_input')
        self.generator_mask = tf.placeholder(shape=[self.batch_size, self.seq_len+2], dtype=tf.int32, name='generator_mask')
        self.z = tf.placeholder(tf.float32, shape=[self.batch_size, self.hidden_size], name='z')
        self.max_len = tf.placeholder(tf.int32, name='max_len')
        self.generator_class = tf.placeholder(dtype=tf.int32, name='generator_class', shape=self.batch_size)
        self.generator_class_one_hot = tf.one_hot(self.generator_class, dtype=tf.float32, depth=self.class_num)


        # discrimnator input
        self.discrimnator_input = tf.placeholder(shape=[self.batch_size, self.seq_len+2], dtype=tf.int32, name='discrimnator_input')
        self.discrimnator_input_one_hot = tf.one_hot(self.discrimnator_input, dtype=tf.float32, depth=data.TAG_NUM)
        self.discriminator_class = tf.placeholder(dtype=tf.int32, name='discriminator_class', shape=self.batch_size)
        self.discriminator_class_one_hot = tf.one_hot(self.discriminator_class, dtype=tf.float32, depth=self.class_num)


        """ Loss Function """
        # output of D for real images
        D_real, D_real_logits, input4classifier_real = self.discriminator(self.discrimnator_input_one_hot,self.max_len, is_training=True, reuse=False)

        # output of D for fake images
        G, self.g_debug_dict = self.generator(self.z, self.generator_input, self.generator_mask, self.max_len,
                                              class_input=self.generator_class, is_training=True, reuse=False)
        D_fake, D_fake_logits, input4classifier_fake = self.discriminator(G, self.max_len, is_training=True, reuse=True)

        # get loss for discriminator
        d_loss_real = - tf.reduce_mean(D_real_logits)
        d_loss_fake = tf.reduce_mean(D_fake_logits)

        self.d_loss = d_loss_real + d_loss_fake

        # get loss for generator
        self.g_loss = - d_loss_fake

        # get classifier loss
        code_real, code_logit_real = self.classifier(input4classifier_real, is_training=True, reuse=False) #dims=[bs, class_num]
        code_fake, code_logit_fake = self.classifier(input4classifier_fake, is_training=True, reuse=True) #dims=[bs, class_num]

        # For real samples
        q_real_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=code_logit_real,
                                                                             labels=self.discriminator_class_one_hot))

        # For fake samples
        q_fake_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=code_logit_fake,
                                                                             labels=self.generator_class_one_hot))

        # get classifier loss
        self.q_loss = q_fake_loss + q_real_loss

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
        q_vars = [var for var in t_vars if ('d_' in var.name) or ('c_' in var.name) or ('g_' in var.name)] #FIXME - why do we need g_vars?
        c_vars = [var for var in t_vars if 'c_' in var.name]

        # optimizers
        with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
            self.d_optim = tf.train.AdamOptimizer(self.learning_rate, beta1=self.beta1) \
                      .minimize(self.d_loss, var_list=d_vars)
            self.g_optim = tf.train.AdamOptimizer(self.learning_rate*5, beta1=self.beta1) \
                      .minimize(self.g_loss, var_list=g_vars)
            self.q_optim = tf.train.AdamOptimizer(self.learning_rate*5, beta1=self.beta1) \
                      .minimize(self.q_loss, var_list=q_vars)

        """" Testing """
        # for test
        self.fake_sents, _ = self.generator(self.z, self.generator_input, self.generator_mask, max_len=self.max_len,
                                            class_input=self.generator_class, is_training=False, reuse=True)

        """ Summary """
        d_loss_real_sum = tf.summary.scalar("d_loss_real", d_loss_real)
        d_loss_fake_sum = tf.summary.scalar("d_loss_fake", d_loss_fake)
        d_loss_sum = tf.summary.scalar("d_loss", self.d_loss)

        q_loss_real_sum = tf.summary.scalar("q_loss_real_sum", q_real_loss)
        q_loss_fake_sum = tf.summary.scalar("q_loss_fake_sum", q_fake_loss)
        q_loss_sum = tf.summary.scalar("q_loss_sum", self.q_loss)

        g_loss_sum = tf.summary.scalar("g_loss", self.g_loss)

        g_vars_sum = [tf.summary.histogram(var.name, var) for var in g_vars]
        d_vars_sum = [tf.summary.histogram(var.name, var) for var in d_vars]
        c_vars_sum = [tf.summary.histogram(var.name, var) for var in c_vars]

        g_output_sum = tf.summary.histogram('G_output', G)
        d_output_fake_sum = tf.summary.histogram('D_fake', D_fake)
        d_output_real_sum = tf.summary.histogram('D_real', D_real)
        q_output_fake_sum = tf.summary.histogram('Q_fake', code_fake)
        q_output_real_sum = tf.summary.histogram('Q_real', code_real)

        # final summary operations
        self.g_sum = tf.summary.merge([d_loss_fake_sum, g_loss_sum, g_output_sum] + g_vars_sum)
        self.d_sum = tf.summary.merge([d_loss_real_sum, d_loss_sum, d_output_fake_sum, d_output_real_sum] + d_vars_sum)
        self.q_sum = tf.summary.merge([q_loss_sum, q_loss_real_sum, q_loss_fake_sum, q_output_real_sum,
                                       q_output_fake_sum] + c_vars_sum)

    def train(self):

        # initialize all variables
        tf.global_variables_initializer().run()

        # graph inputs for visualize training results
        self.sample_z = np.random.normal(0, 1, size=(self.batch_size, self.hidden_size))

        # saver to save model
        self.saver = tf.train.Saver()

        # summary writer
        self.writer = tf.summary.FileWriter(self.log_dir + '/' + self.model_name, self.sess.graph)

        # load text
        self.text = data.load_sanity_data()
        self.num_batches = len(self.text) // (self.batch_size * 2)

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

        max_len_list = [1, 1 ,2 ,2, 4, 4, 8, 8, 16, 16, 32, 32]

        # loop for epoch
        start_time = time.time()
        assert(self.epoch == len(max_len_list))
        for epoch in range(start_epoch, self.epoch):
            # self.visualize_results(epoch, max_len=32)  # for debug - max len remains constant and maximal
            #arrange data
            cur_max_len = max_len_list[epoch]
            mask_list, feed_tags = data.create_shuffle_data(self.text, max_len=cur_max_len, seq_len=self.seq_len,
                                                            mode='th_extended')

            # get batch data
            for idx in range(start_batch_id, self.num_batches):
                batch_sents = feed_tags[idx*(self.batch_size*2):(idx+1)*(self.batch_size*2)]
                batch_sents_generator = batch_sents[:self.batch_size]
                batch_sents_discriminator = batch_sents[self.batch_size:]

                batch_z = np.random.normal(0, 1, [self.batch_size, self.z_dim]).astype(np.float32)
                batch_mask = mask_list[idx*(self.batch_size*2):(idx+1)*(self.batch_size*2)]
                batch_mask_generator = batch_mask[:self.batch_size]

                # update D network
                _, summary_str, d_loss, q_loss = self.sess.run([self.d_optim, self.d_sum, self.d_loss, self.q_loss],
                                               feed_dict={self.discrimnator_input: batch_sents_discriminator,
                                                          self.generator_input: batch_sents_generator,
                                                          self.generator_mask: batch_mask_generator,
                                                          self.generator_class: np.ones(self.batch_size, dtype=np.int32),
                                                          self.discriminator_class: np.ones(self.batch_size, dtype=np.int32),
                                                          self.z: batch_z,
                                                          self.max_len: cur_max_len}) #FIXME - update generator and discriminator class
                self.writer.add_summary(summary_str, counter)

                # update G network
                if (counter-1) % self.disc_iters == 0:
                    batch_z = np.random.normal(0, 1, [self.batch_size, self.z_dim]).astype(np.float32)
                    _, summary_str, c_summary_str, g_loss = self.sess.run([self.g_optim, self.g_sum, self.q_sum, self.g_loss],
                                                           feed_dict={self.z: batch_z,
                                                                      self.generator_mask: batch_mask_generator,
                                                                      self.generator_input: batch_sents_generator,
                                                                      self.generator_class: np.ones(self.batch_size, dtype=np.int32),
                                                                      self.max_len: cur_max_len}) #FIXME - update generator class
                    self.writer.add_summary(summary_str, counter)
                    self.writer.add_summary(c_summary_str, counter)

                # #for debug
                # if idx % 10 == 0:
                #     batch_z = np.random.normal(0, 1, [self.batch_size, self.z_dim]).astype(np.float32)
                #     out_data, out_gen, out, output_prob = self.sess.run([
                #                                             self.g_debug_dict['out_data'],
                #                                             self.g_debug_dict['out_gen'],
                #                                             self.g_debug_dict['out'],
                #                                             self.g_debug_dict['output_prob']],
                #                                            feed_dict={self.z: batch_z,
                #                                                       self.generator_mask: batch_mask_generator,
                #                                                       self.generator_input: batch_sents_generator,
                #                                                       self.max_len: cur_max_len})
                #     print('debug')

                counter += 1


                # display training status
                print("Epoch: [%2d] [%4d/%4d] time: %4.4f, d_loss: %.8f, g_loss: %.8f, q_loss: %.8f" \
                      % (epoch, idx, self.num_batches, time.time() - start_time, d_loss, g_loss, q_loss))

                # # save training results for every 300 steps
                # if np.mod(counter, 300) == 0:
                #     samples = self.sess.run(self.fake_sents,
                #                             feed_dict={self.z: self.sample_z})
                #     tot_num_samples = min(self.sample_num, self.batch_size)
                #     manifold_h = int(np.floor(np.sqrt(tot_num_samples)))
                #     manifold_w = int(np.floor(np.sqrt(tot_num_samples)))
                #     save_images(samples[:manifold_h * manifold_w, :, :, :], [manifold_h, manifold_w],
                #                 './' + check_folder(self.result_dir + '/' + self.model_dir) + '/' + self.model_name + '_train_{:02d}_{:04d}.png'.format(
                #                     epoch, idx))

            # After an epoch, start_batch_id is set to zero
            # non-zero value is only for the first epoch after loading pre-trained model
            start_batch_id = 0

            # save model
            self.save(self.checkpoint_dir, counter)

            # show temporal results
            self.visualize_results(epoch,max_len=32) # for debug - max len remains constant and maximal

        # save model for final step
        self.save(self.checkpoint_dir, counter)

    def visualize_results(self, epoch, max_len):
        tot_num_samples = min(self.sample_num, self.batch_size)

        """ random condition, random noise """

        z_sample = np.random.normal(0, 1, size=(self.batch_size, self.z_dim))
        batch_empty_mask, batch_empty_data = data.create_empty_data(sent_num=self.batch_size,seq_len=self.seq_len,max_len=max_len)

        samples = self.sess.run(self.fake_sents, feed_dict={self.z: z_sample,
                                                            self.generator_mask: batch_empty_mask,
                                                            self.generator_input: batch_empty_data,
                                                            self.generator_class: np.ones(self.batch_size, dtype=np.int32),
                                                            self.max_len: max_len}) #FIXME - update generator class
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
        if not os.path.isdir(os.path.join('results', 'baseline')):
            os.mkdir(os.path.join('results', 'baseline'))

        save_path = os.path.join('results', 'baseline', 'results_epoch_' + str(epoch) + '.txt')
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