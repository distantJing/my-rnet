import tensorflow as tf
import os
import shutil

from data_load import *
from model import *
from graph_handler import *


def main(config):
    config = set_dirs(config)
    with tf.device(config.device):
        if config.mode == 'train':
            train(config)
        elif config.mode == 'test':
            test(config, 'dev')
        elif config.mode == 'debug':
            debug(config)
        else:
            raise ValueError("Invalid value for 'mode': {}".format(config.mode))


def set_dirs(config):
    '''
    create directories 创建输出目录
    '''
    # 训练以外的模式 config.load必须为True
    assert config.load or config.mode=='train', "config.load must be True if not training"
    if not config.load and os.path.exists(config.out_dir):
        # 在train过程中，如果输出目录存在，则将其删除
        shutil.rmtree(config.out_dir)

    config.save_dir = os.path.join(config.out_dir, "save")
    config.log_dir = os.path.join(config.out_dir, "log")        # summary
    config.eval_dir = os.path.join(config.out_dir, "eval")
    config.answer_dir = os.path.join(config.out_dir, "answer")  # answer_text
    if not os.path.exists(config.out_dir):
        os.makedirs(config.out_dir)
    if not os.path.exists(config.save_dir):
        os.mkdir(config.save_dir)
    if not os.path.exists(config.log_dir):
        os.mkdir(config.log_dir)
    if not os.path.exists(config.eval_dir):
        os.mkdir(config.eval_dir)
    if not os.path.exists(config.answer_dir):
        os.mkdir(config.answer_dir)
    return config


def train(config):
    train_data = read_data(config, 'train')
    dev_data = read_data(config, 'dev')
    # 得到多个batch的集合
    train_batches = train_data.get_batches(config.train_batch_size)
    dev_batches = dev_data.get_batches(config.train_batch_size)

    model = Model(config, 'train', scope="model")
    graph_handler = GraphHandler(config, model)

    sess = tf.Session()  # todo: check here

    # 加载 或 设置保存 模型
    graph_handler.initialize(sess)

    # begin training
    # 每次训练一个batch, num_step * batch_size = num_examples * epoch
    # todo: 注意此处参数的计算
    num_step = config.num_steps
    global_step = 0
    for batch in tqdm(train_batches, total=num_step):
        # 用于判断是否保存中间模型、计算结果KL
        global_step = sess.run(model.global_step) + 1
        get_summary = global_step % config.log_period == 0

        # 训练，并在适当时刻保存summary
        if get_summary:
            loss, summary, train_op = sess.run([model.mean_loss, model.summary, model.train_op], feed_dict=model.batch_to_feed_dict(batch))
            graph_handler.add_summary(summary, global_step)
        else:
            # print('model batch:', model.batch_to_feed_dict(batch).get_shape().as_list())
            loss, train_op = sess.run([model.mean_loss, model.train_op], feed_dict=model.batch_to_feed_dict(batch))

        # 适当时刻保存整个模型
        if global_step % config.save_peroid == 0:
            graph_handler.save(sess, global_step=global_step)

        # 进行模型评价
        if global_step % config.eval_period == 0:
            # num_dev_step * batch_size = num_dev_examples
            num_dev_step = math.ceil(dev_data.num_examples / config.batch_size)
            whole_pred_dev_answer_text = get_whole_pred_answer_text(model, sess, dev_batches, num_dev_step)
            # 保存answer
            graph_handler.dump_answer(whole_pred_dev_answer_text, 'dev', global_step)
    if global_step % config.save_period != 0:
        graph_handler.save(sess, global_step=global_step)


def test(config, data_type):
    data = read_data(config, data_type)
    batches = data.get_batches(config.train_batch_size)
    model = Model(config, 'test', scope="model")
    graph_handler = GraphHandler(config, model)
    sess = tf.Session()    # todo: check here
    graph_handler.initialize(sess) # 加载模型

    # num_step * batch_size = num_examples
    num_step = math.ceil(data.num_examples / config.train_batch_size)
    whole_pred_answer_text = get_whole_pred_answer_text(model, sess, batches, num_step)
    # 保存答案文本
    global_step = sess.run(model.global_step)
    graph_handler.dump_answer(whole_pred_answer_text, data_type, global_step)


def get_whole_pred_answer_text(model, sess, batches, num_step):
    pred_answer_text = {}
    for batch in tqdm(batches, total=num_step):
        batch_ans = sess.run(model.pred_answer_text, feed_dict=model.batch_to_feed_dict(batch,mode='test'))
        pred_answer_text = {**pred_answer_text, **batch_ans}
    return pred_answer_text
