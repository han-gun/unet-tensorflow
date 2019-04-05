import os

import tensorflow as tf
import numpy as np
import scipy.io as sio
from tqdm import tqdm
import visdom

from utils import Logger
from utils import Parser

LOGGER = Logger(name=__name__).get_logger()

VIS = visdom.Visdom

class Trainer:
    def __init__(self, param):
        self.__param = param

        self.scope = param.scope
        self.train_continue = param.train_continue
        self.checkpoint_dir = param.checkpoint_dir
        self.log_dir = param.log_dir

        self.output_dir = param.output_dir

        self.optimizer = param.optimizer
        self.loss = param.loss

        self.global_step = 0
        self.epoch_num = param.epoch_num
        self.batch_size = param.batch_size

        self.input_y_size = param.input_y_size
        self.input_x_size = param.input_x_size
        self.input_ch_size = param.input_ch_size

        if param.patch_y_size == -1:
            self.patch_y_size = param.input_y_size
        else:
            self.patch_y_size = param.patch_y_size

        if param.patch_x_size == -1:
            self.patch_x_size = param.input_x_size
        else:
            self.patch_x_size = param.patch_x_size

        if param.patch_ch_size == -1:
            self.patch_ch_size = param.input_ch_size
        else:
            self.patch_ch_size = param.patch_ch_size

        if param.data_type == 'float32':
            self.data_type = tf.float32
            self.label_type = tf.float32
        elif param.data_type == 'float16':
            self.data_type = tf.float16
            self.label_type = tf.float16

        if self.loss == 'softmax':
            self.label_type = tf.int32

        self.decay_type = param.decay_type

        self.learning_rate = param.learning_rate
        self.decay_factor = param.decay_factor
        self.decay_step = param.decay_step

        if self.decay_type == 'log':
            self.learning_rate_set = self.learning_rate*np.logspace(0, self.decay_factor, num=self.epoch_num)
        elif self.decay_type == 'stair':
            self.learning_rate_set = np.power(1e-1, [i // self.decay_step for i in range(self.epoch_num)])

    def _summary_image(self, image_dict={}):
        _summary_image_merge = [tf.summary.image(key, tf.cast(tf.clip_by_value(image_dict[key], 0, 255), dtype=self.data_type), max_outputs=self.batch_size) for key in image_dict]
        summary_image_op = tf.summary.merge(_summary_image_merge)

        return summary_image_op

    def _summary_scalar(self, scalar_dict={}):
        _summary_scalar_merge = [tf.summary.scalar(key, scalar_dict[key]) for key in scalar_dict]
        summary_scalar_op = tf.summary.merge(_summary_scalar_merge)

        return summary_scalar_op

    def save(self, saver, sess, step=0):
        checkpoint_dir = os.path.join(self.checkpoint_dir, self.scope)

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        saver.save(sess, "{}/model.ckpt".format(checkpoint_dir), global_step=step)

    def load(self, saver, sess):
        checkpoint_dir = os.path.join(self.checkpoint_dir, self.scope)
        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
            self.global_step = int(ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1])
            return True
        else:
            return False

    def test(self, MODELS, DATABASE):
        '''
        if [training] is True, train_mode
        else [training] is False, validation_ and test_mode
        '''

        ''' 
        construct the MODEL, LOSS, and OPTIMIZER for training
        '''
        training_op = tf.placeholder(tf.bool, name='training')
        input_op = tf.placeholder(self.data_type, shape=[None, self.patch_y_size, self.patch_x_size, self.patch_ch_size], name='input')
        pred_op = MODELS.get_network(input=input_op, training=training_op, network_type='unet', reuse=False, name='generator')
        LOGGER.info('CONSTRUCT the model for training')

        # output_dir = os.path.join(self.output_dir, self.scope)
        output_dir = self.output_dir
        if not os.path.exists(output_dir):
            LOGGER.info('CREATE the directory for saving the results')
            os.makedirs(output_dir)

        config = tf.ConfigProto(allow_soft_placement=True)
        config.gpu_options.allow_growth = True
        # config.gpu_options.per_process_gpu_memory_fraction = 0.2
        with tf.Session(config=config) as sess:
            init = tf.global_variables_initializer()
            sess.run(init)
            LOGGER.info('INITIALIZE the global variables')

            saver = tf.train.Saver()
            if self.load(saver, sess):
                LOGGER.info("[*] LOAD SUCCESS")
            else:
                LOGGER.info("[!] LOAD failed...")

            '''
            Tset mode
            '''
            test_idx = DATABASE.get_index(shuffle=False, type='test')
            test_num = len(test_idx)

            # for itest in range(test_num):
            ttest = tqdm(range(test_num))
            for itest in ttest:
                input = DATABASE.get_data(itest, mode='test', type='input')
                batch, batch_num = DATABASE.get_image2patch(input)

                for st in range(0, batch_num, self.batch_size):
                    ed = st + self.batch_size if st + self.batch_size < batch_num else batch_num

                    test_input = batch[st:ed, :, :, :]
                    pred_img = sess.run(pred_op, feed_dict={input_op: test_input, training_op: False})

                    batch[st:ed, :, :] = pred_img

                pred, _ = DATABASE.get_patch2image(batch, type='count')
                sio.savemat(os.path.join(output_dir, 'pred_%04d.mat' % itest), {'pred': pred})

    def train(self, MODELS, DATABASE):
        ''' 
        if [training] is True, train_mode
        else [training] is False, validation_ and test_mode
        '''
        training_op = tf.placeholder(tf.bool, name='training')
        learning_rate_op = tf.placeholder(self.data_type, name='learning_rate')
        fid_loss_mean_op = tf.placeholder(self.data_type, name='mean_fid_loss')

        LOGGER.info('CONSTRUCT the hyper parameters')

        ''' 
        construct the MODEL, LOSS, and OPTIMIZER for training
        '''
        input_op = tf.placeholder(self.data_type, shape=[None, self.patch_y_size, self.patch_x_size, self.patch_ch_size], name='input')
        label_op = tf.placeholder(self.label_type, shape=[None, self.patch_y_size, self.patch_x_size, self.patch_ch_size], name='label')
        pred_op = MODELS.get_network(input=input_op, training=training_op, network_type='unet', reuse=False, name='generator')

        LOGGER.info('CONSTRUCT the model for training')

        LOSSES = Losses(self.get_param())
        OPTIMIZERS = Optimizers(self.get_param())

        fid_loss_op = LOSSES.get_loss(label=label_op, pred=pred_op, loss_type=self.loss)
        reg_loss_op = LOSSES.get_regularization()
        gen_loss_op = tf.add_n([fid_loss_op, reg_loss_op])

        gen_vars = MODELS.get_vars(name='generator')
        optimizer_op = OPTIMIZERS.get_optimizer(loss=gen_loss_op, var_list=gen_vars, optimizer_type=self.optimizer, learning_rate=learning_rate_op)

        LOGGER.info('CONSTRUCT the loss & optimizers for training')

        ''' 
        create tensorboard summary
        '''
        summary_image_op = self._summary_image(image_dict={'label': label_op, 'input': input_op, 'prediction': pred_op})
        summary_loss_op = self._summary_scalar(scalar_dict={'fidelity_loss': fid_loss_mean_op})

        train_log_dir = os.path.join(self.log_dir, self.scope, 'train')
        val_log_dir = os.path.join(self.log_dir, self.scope, 'test')

        LOGGER.info('CONSTRUCT the tensorboard')

        config = tf.ConfigProto(allow_soft_placement=True)
        config.gpu_options.allow_growth = True
        # config.gpu_options.per_process_gpu_memory_fraction = 0.2
        with tf.Session(config=config) as sess:
            train_summary_writer = tf.summary.FileWriter(train_log_dir, sess.graph)
            val_summary_writer = tf.summary.FileWriter(val_log_dir)

            init = tf.global_variables_initializer()
            sess.run(init)

            LOGGER.info('INITIALIZE the global variables')

            saver = tf.train.Saver()
            if self.train_continue:
                if self.load(saver, sess):
                    LOGGER.info("[*] LOAD SUCCESS")
                else:
                    LOGGER.info("[!] LOAD failed...")

            try:
                LOGGER.info('START the training network')
                # for epoch in range(self.epoch_num):
                tepoch = tqdm(range(self.global_step, self.epoch_num))
                # tepoch = trange(self.epoch_num)
                for epoch in tepoch:
                    lr = self.learning_rate_set[epoch]

                    '''
                    Train mode
                    '''
                    train_idx = DATABASE.get_index(type='train')
                    train_num = len(train_idx)

                    train_fid_loss_mean = 0
                    train_cnt = 0

                    # tbatch = tqdm(range(0, train_num, self.batch_size), position=1)
                    # tbatch = trange(0, train_num, self.batch_size)
                    # for st in tbatch:
                    for st in range(0, train_num, self.batch_size):
                        ed = st + self.batch_size if st + self.batch_size < train_num else train_num

                        idx = train_idx[st:ed]
                        train_label, train_input = DATABASE.get_database(idx, mode='train')
                        # train_input = DATABASE.add_noise(data=train_label, sgm=30)

                        _, train_fid_loss = sess.run([optimizer_op, fid_loss_op],
                                                     feed_dict={label_op: train_label,
                                                                input_op: train_input,
                                                                training_op: True,
                                                                learning_rate_op: lr})

                        train_fid_loss_mean += train_fid_loss
                        train_cnt += 1

                    '''
                    Add tensorboard for training mode 
                    '''
                    idx = train_idx[0:self.batch_size]
                    train_label, train_input = DATABASE.get_database(idx, mode='train')

                    train_summary_image = sess.run(summary_image_op, feed_dict={label_op: train_label, input_op: train_input, training_op: True})
                    train_summary_loss_mean = sess.run(summary_loss_op, feed_dict={fid_loss_mean_op: train_fid_loss_mean / train_cnt})

                    train_summary_writer.add_summary(train_summary_image, epoch)
                    train_summary_writer.add_summary(train_summary_loss_mean, epoch)

                    '''
                    Validation mode
                    '''
                    val_idx = DATABASE.get_index(type='val')
                    val_num = len(val_idx)

                    val_fid_loss_mean = 0
                    val_cnt = 0

                    for st in range(0, val_num, self.batch_size):
                    # tbatch = trange(0, val_num, self.batch_size)
                    # for st in tbatch:
                    #     tbatch.set_description("{} ".format(epoch) + " / {}".format(self.epoch_num) + ', valid')

                        ed = st + self.batch_size if st + self.batch_size < val_num else val_num

                        idx = val_idx[st:ed]
                        val_label, val_input = DATABASE.get_database(idx, mode='val')
                        # val_input = DATABASE.add_noise(data=val_label, sgm=30)

                        val_fid_loss = sess.run(fid_loss_op, feed_dict={label_op: val_label, input_op: val_input, training_op: False})
                        val_fid_loss_mean += val_fid_loss
                        val_cnt += 1

                    '''
                    Add tensorboard for val mode 
                    '''
                    idx = val_idx[0:self.batch_size]
                    val_label, val_input = DATABASE.get_database(idx, mode='val')

                    val_summary_image = sess.run(summary_image_op, feed_dict={label_op: val_label, input_op: val_input, training_op: False})
                    val_summary_loss_mean = sess.run(summary_loss_op, feed_dict={fid_loss_mean_op: val_fid_loss_mean / val_cnt})

                    val_summary_writer.add_summary(val_summary_image, epoch)
                    val_summary_writer.add_summary(val_summary_loss_mean, epoch)

                    '''
                    Print losses
                    '''
                    tepoch.pos = 1
                    tepoch.set_postfix(train='{:.3f}'.format(train_fid_loss_mean / train_cnt), val='{:.3f}'.format(val_fid_loss_mean / val_cnt))
                    tepoch.pos = 0

                    if (epoch % 10) == 0:
                        self.save(saver, sess, epoch)

            finally:
                self.save(saver, sess, epoch+1)

    def get_param(self):
        return self.__param


class Optimizers:
    def __init__(self, param):
        self.optimizer = param.optimizer

    def get_optimizer(self, loss, var_list, optimizer_type='adam', learning_rate=1e-1):
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.variable_scope(optimizer_type):
            with tf.control_dependencies(update_ops):
                if optimizer_type == 'sgd':
                    optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, var_list=var_list)
                elif optimizer_type == 'adam':
                    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss, var_list=var_list)
                elif optimizer_type == 'rmsprop':
                    optimizer = tf.train.RMSPropOptimizer(learning_rate).minimize(loss, var_list=var_list)

        return optimizer


class Losses:
    def __init__(self, param):
        self.loss = param.loss
        self.weight_decay = param.weight_decay

    def get_regularization(self):
        reg_loss = tf.reduce_sum(tf.losses.get_regularization_losses())
        return reg_loss

    def get_l1_loss(self, x):
        loss = tf.reduce_mean(tf.abs(x))/1
        return loss

    def get_l2_loss(self, x):
        loss = tf.reduce_mean(tf.square(x))/2
        return loss

    def get_norm_loss(self, x, ord=2):
        loss = tf.norm(tensor=x, ord=ord)
        return loss

    def get_tv_loss(self, x):
        loss = tf.reduce_mean(tf.image.total_variation(x))
        return loss

    def get_loss(self, label, pred, loss_type='l2'):
        if loss_type == 'l2':
            loss = tf.losses.mean_squared_error(label, pred)
        elif loss_type == 'l1':
            loss = tf.losses.absolute_difference(label, pred)
        elif loss_type == 'softmax':
            loss = tf.losses.sparse_softmax_cross_entropy(label, pred)

        return loss