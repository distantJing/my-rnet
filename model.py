import tensorflow as tf

from layer import *
from RNNCell import *

class Model():
    def __init__(self, config, mode, scope):
        self.config = config
        self.mode = mode
        self.scope = scope
        # 用于保存模型等
        self.global_step = tf.get_variable("global_step", shape=[], dtype='int32',
                                           initializer=tf.constant_initializer(0), trainable=False)

        # hyper parameters
        self.num_epochs = config.num_epochs
        # self.num_steps = config.num_steps
        self.learning_rate = config.init_lr
        self.input_keep_prob = config.input_keep_prob
        self.hidden_size = config.hidden_size
        self.max_question_word = config.max_question_word     # Q, 问题长度
        self.max_passage_word = config.max_passage_word       # P, 文章长度
        self.word_emb_size = config.word_emb_size             # D_w, word embedding，与glove相关
        self.char_emb_size = config.char_emb_size             # D_c, char embedding, 即character_embedding()
        self.cell_fn = config.cell_fn                         # 无括号的
        self.lr = config.init_lr                              # learning rate

        # define input
        self.batch_size = config.train_batch_size
        # self.batch_size = tf.placeholder('int32',1)
        # passage and question words, and its sequence len, 第一维为self.batch_size
        self.passage_words = tf.placeholder('float',[None, self.max_passage_word, self.word_emb_size])
        self.question_words = tf.placeholder('float', [None, self.max_question_word, self.word_emb_size])
        self.passage_words_len = tf.placeholder('int32', [None])
        self.question_words_len = tf.placeholder('int32', [None])
        # passage and question char, and its sequence len
        self.passage_words_char = tf.placeholder('float', [None, self.max_passage_word,])
        self.question_words_char = tf.placeholder('float', [None, self.max_question_word, ])
        # question id and answer label
        self.question_id = tf.placeholder(tf.string, [None])
        self.answer_index = tf.placeholder('int32', [None, 2])

        # define the first layer output 分别对P, Q编码
        # self.u_P
        # self.u_Q

        # define output
        self.pre_answer = None

        # define loss output
        self.loss = None

        # define network structure
        # 第一层，encoding 层
        self.encode()
        # gated attention-based recurrent networks
        self.params = get_attn_params(self.hidden_size)
        self.gated_attention_layer()   # gated, self-matching
        # output layer
        self.pointer_network()
        # 计算question_id, answer 集合对
        self.answer_select()
        # 设置loss计算方式
        self.compute_loss()
        # 设置train_op
        self.train()

    def character_embedding(self, w):
        #todo: 未完成
        '''
        对于每一个单词word，生成其character-level embedding
        方法：    将单词的每个字母作为输入，使用biRNN网络，将最后一个hidden state作为char-level embedding
        input:    w_P       [batch_size, P, D_w] or [batch_size, Q, D_w]
        结果:     c_P       [batch_size, P, self.char_embedding] or [batch_size, Q, self.char_embedding]
        '''
        # 设置rnn单元，默认GRU
        cell_fn = self.cell_fn
        states = bidirectional_RNN(w, cell_fn, self.hidden_size, self.input_keep_prob,
                                   scope="character-level embedding", output_type=1)
        return states

    def encode(self):
        '''
        use biRNN to build question and passage representation separately
        网络结构第一层
        input：    [batch_size, Q, D] passage_words, 一篇文章的向量表示， 使用glove生产的word level表示
                   [batch_size, P, D] question_words, 一个问题的向量表示

        结果:      passage和question的新表示
                   u_P, u_Q   [batch_size, P, 2*self.hidden_size]
        '''
        # generate word-level embedding
        e_P = self.passage_words        # [batch_size, P, D_w]
        e_Q = self.question_words

        if self.config.use_char_emb:
            # generate character-level embedding
            c_P = character_embedding(w_P)  # [batch_size, P, D_c]
            c_Q = character_embedding(w_Q)

            P_w_c = tf.concat([e_P, c_P], 2)   # [batch_size, P, D_W+D_c]
            Q_w_c = tf.concat([e_Q, c_Q], 2)
        else:
            P_w_c = e_P
            Q_w_c = e_Q

        # 设置rnn 单元, 默认 GRU
        cell_fn = self.cell_fn
        # 网络第一层的passage encoding
        self.u_P = bidirectional_RNN(P_w_c, cell_fn, self.hidden_size, self.input_keep_prob, self.passage_words_len,
                                     scope='passage_encoding', output_type=0)
        self.u_Q = bidirectional_RNN(Q_w_c, cell_fn, self.hidden_size, self.input_keep_prob, self.question_words_len,
                                     scope='question_encoding', output_type=0)

    def gated_attention_layer(self):
        '''
        使用gated attention-based recurrent network 将问题纳入文章表示
        input:    u_P   [batch_size, P, 2*self.hidden_size]
                  u_Q   [batch_size, Q, 2*self.hidden_size]
        :return:  v_P   [batch_size, P, 2*self.hidden_size]

        使用self-matching 重新表示passage
        input:    v_P [batch_size, P, 2*self.hidden_size]
        :return:
        '''
        # define attention weights
        with tf.variable_scope("gated_attention-based_recurrent_networks"):
            memory = self.u_Q    # question encoding
            inputs = self.u_P    # passage encoding
            scopes = ["question_passage_matching", "self_matching"]
            params = [(([self.params["W_u_Q"],
                         self.params["W_u_P"],
                         self.params["W_v_P"]], self.params["v"]),
                       self.params["W_g"]),
                      (([self.params["W_v_P_2"],
                         self.params["W_v_Phat"]], self.params["v"]),
                       self.params["W_g"])]
            for i in range(2):
                args = {"num_units":self.hidden_size,
                        "memory":memory,
                        "params":params[i],
                        "cell_fn": self.cell_fn,
                        "self_matching":False if i==0 else True,
                        "memory_len":self.question_words_len if i==0 else self.passage_words_len,
                        "is_training":self.mode=='train'
                        }
                cell = [apply_dropout(gated_attention_Wrapper(**args), self.config.input_keep_prob,
                                      mode=self.mode) for _ in range(2)]
                inputs = bidirectional_RNN(inputs, cell, self.hidden_size, self.input_keep_prob, self.passage_words_len,
                                           scope=scopes[i])
                memory = inputs # 第三层 self_matching层，match against itsel
            self.h_t_P = inputs  # self_matching output

    def pointer_network(self):
        '''
        根据self-matching的结果，进行output_laywer层运算， 得到[batch_size, P, 2*hidden_size]
        从final_output, u_Q(question_encoding)计算 答案起始位置 概率分布
        :return:  [batch_size, 2]
        '''
        # 最终输出
        self.final_outputs = bidirectional_RNN(self.h_t_P, self.cell_fn, self.hidden_size, self.input_keep_prob,
                                               self.passage_words_len,
                                               scope="bidirectional_output", mode=self.mode)
        params = (([self.params["W_u_Q"],
                    self.params["W_v_Q"]], self.params["v"]),
                  ([self.params["W_h_P"],
                    self.params["W_h_a"]], self.params["v"]))
        cell = apply_dropout(self.cell_fn(self.hidden_size*2), self.config.input_keep_prob, mode=self.mode)
        # [batch_size, 2, P] 每个word为起始位置的概率分布
        self.points_logits = pointer(self.final_outputs, self.passage_words_len,   # passage representation
                                     self.u_Q, self.question_words_len,            # question representation
                                     cell, params, self.batch_size, self.hidden_size, scope="pointer_network")

    def answer_select(self):
        # [batch_size, 2] 计算每一个答案的起始位置
        self.output_index = tf.argmax(self.points_logits, axis=2)
        # 格式：{id:answer}  包含question_id, answer的集合
        pred_answer_text = {}
        print('self.output_index:', self.output_index[0][0], self.output_index[0][1])

        # for output_index, passage_word, question_id in zip(self.output_index, self.passage_words, self.question_id):
        #     start, end = output_index[0], output_index[1]
        #     start, end = 0, 1  #todo: what are start and end?
        #     answer_text = passage_word[start:end]
        #     pred_answer_text[question_id] = answer_text

        for i in range(0, self.batch_size):
            start, end = self.output_index[i][0], self.output_index[i][1]
            # todo: what are start and end?
            start, end = 0, 1
            answer_text = self.passage_words[i][start:end]
            question_id = self.question_id[i]
            pred_answer_text[question_id] = answer_text

        self.pred_answer_text =pred_answer_text

    def compute_loss(self):
        with tf.variable_scope("compute_loss"):
            # 答案的概率分布
            depth = self.max_passage_word
            self.answer_index_prob = tf.one_hot(self.answer_index, depth)
            self.mean_loss = compute_loss_value(self.points_logits, self.answer_index_prob)

    def train(self):
        optimizer = tf.train.AdamOptimizer(self.lr)
        self.train_op = optimizer.minimize(self.mean_loss)

    def summary(self):
        self.F1 = tf.Variable(tf.constant(0.0, tf.float32, shape=()), trainable=False, name="F1")
        self.EM = tf.Variable(tf.constant(0.0, tf.float32, shape=()), trainable=False, name='EM')
        self.dev_loss = tf.Variable(tf.constant(5.0, shape=(), dtype=tf.float32), trainable=False, name="dev_loss")
        self.F1_placeholder = tf.placeholder(tf.float32, shape=(), name="F1_placeholder")
        self.EM_placeholder = tf.placeholder(tf.float32, shape=(), name="EM_placeholder")
        self.dev_loss_placehodler = tf.placeholder(tf.float32, shape=(), name="dev_loss_placeholder")
        tf.summary.scalar("loss_training", self.mean_loss)
        tf.summary.scalar("loss_dev", self.dev_loss)
        tf.summary.scalar("F1_score", self.F1)
        tf.summary.scalar("Exact_Match", self.EM)
        tf.summary.scalar("learning_rate", self.lr)
        self.summary = tf.summary.merge_all()

    def batch_to_feed_dict(self, batch, mode='train'):
        '''
        将batch转换为feed——dict
            batch_data["batch_size"] = this_batch_size
            batch_data["passage_words"] = self.passage[start:end]
            batch_data["question_words"] = self.question[start:end]
            batch_data["passage_words_len"] = self.passage_len[start:end]
            batch_data["question_words_len"] = self.question_len[start:end]
            batch_data["question_id"] = self.question_id[start:end]
            batch_data["answer_index"] = self.answer_index[start:end]
        :return: feed_dict
        '''
        feed_dict = {}
        feed_dict[self.batch_size] = batch["batch_size"]
        feed_dict[self.passage_words] = batch["passage_words"]
        feed_dict[self.question_words] = batch["question_words"]
        feed_dict[self.passage_words_len] = batch["passage_words_len"]
        feed_dict[self.question_words_len] = batch["question_words_len"]
        feed_dict[self.question_id] = batch["question_id"]
        feed_dict[self.answer_index] = batch["answer_index"]
        feed_dict[self.mode] = mode

        batch_size = feed_dict[self.batch_size]
        print('model get feed dict: batch_size: ', batch_size)
        return feed_dict