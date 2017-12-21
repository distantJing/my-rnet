import tensorflow as tf
import os
import json

class GraphHandler(object):
    def __init__(self, config, model):
        self.config = config
        self.model = model
        self.saver = tf.train.Saver(max_to_keep=config.max_to_keep)
        self.writer = None
        self.save_path = os.path.join(config.save_dir, config.model_name)

    def initialize(self, sess):
        init_op = tf.global_variables_initializer()
        sess.run(init_op)

        # 非train, 加载已有的model模型
        if self.config.load:
            self.load(sess)
        # train, 保存summary，可用于tensor board
        if self.config.mode == 'train':
            self.writer = tf.summary.FileWriter(self.config.log_dir, graph=tf.get_default_graph())

    def save(self, sess, global_step=None):
        # 保存模型
        saver = tf.train.Saver(max_to_keep=self.config.max_to_keep)
        saver.save(sess, self.save_path, global_step=global_step)

    def load(self, sess):
        # todo: add load
        config = self.config
        vars = {var.name.split(":")[0]: var for var in tf.all_variables()}

    def add_summary(self, summary, global_step):
        self.writer.add_summary(summary, global_step)
    def add_summaries(self, summaries, global_step):
        for summary in summaries:
            self.add_summary(summary, global_step)

    def dump_answer(self, answer, data_type, global_step):
        answer_path = os.path.join(self.config.answer_dir, "{}-{}.json".format(data_type, global_step))
        json.dump(answer, open(answer_path, 'w'))
