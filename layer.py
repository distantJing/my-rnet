import tensorflow as tf

# from config import get_small_config

def get_batch_size():
    return 7


def bidirectional_RNN(inputs, cell_fn, units, input_keep_prob, inputs_len, cell=None, output_type=0,
                      layers=1, scope="Bidirectional_RNN", is_training=True):
    '''
    bidirectional recurrent neural network with LSTM or GRU cells
    Args:
        inputs:      rnn input of shape (batch_size, timestep, dim)
        cell_fn:     which unit we use, like LSTM or GRU
        units:       rnn_cell的初始化units
        inputs_len:  an int vector, size [batch_size], containing the actual length for each sequence
        cell:        rnn cell of type RNN_cell, 预先定义的 (cell_fw, cell_bw)
        output_type: if 0, output returns rnn output for every timestep, 输出
                            [batch_size, timestep, 2*units]
                     if 1, output returns concatenated state of backward and forward rnn. 隐藏状态
                            [batch_size, timestep, 2*units]
        layers:      多层双向rnn，具体层数
    '''
    # config = get_small_config()
    cell_fn = tf.nn.rnn_cell.GRUCell
    # units = 128
    with tf.variable_scope(scope):
        inputs_shape = inputs.get_shape().as_list()
        # print('inputs_shape: ', inputs_shape)
        batch_size = inputs_shape[0]
        timestep = inputs_shape[1]
        # 定义rnn cell of forward direction, backward direction (fw, bw)
        if cell is not None:
            # 传入的cell
            (cell_fw, cell_bw) = cell
        else:
            # 传入的input维度过大，需要将其 reshape 为三维
            if len(inputs_shape) > 3:
                input = tf.reshape(inputs, (batch_size, inputs_shape[1], -1))
                # todo: 重写 input_len

            if layers > 1:
                # todo: 添加多层双向网络的cell
                pass
            else:
                cell_fw, cell_bw = [apply_dropout(cell_fn(units), input_keep_prob, is_training=is_training)
                                    for _ in range(2)]

        # outputs, states = tf.nn.dynamic_rnn(cell_fw, input, dtype=tf.float32)
        # print('input_len: ', inputs_len)
        outputs, states = tf.nn.bidirectional_dynamic_rnn(cell_fw, cell_bw, inputs, sequence_length=inputs_len,
                                                          dtype='float', time_major=False)

        # print('output: ', outputs)
        # print('states: ', states[0][1])
        # results = tf.matmul(states[1][0], weights['out']) + biases['out']
        # return results

        # 输出对应的结果
        if output_type == 0:
            # print('return:', tf.concat(2, outputs))
            return tf.concat(2, outputs)
        elif output_type == 1:
            # todo: check this
            return tf.reshape(tf.concat(states,1), (batch_size, inputs_shape[1], 2*units))

def apply_dropout(inputs, keep_prob, is_training=True):
    # todo: apply dropout here
    if is_training:
        return inputs
        return tf.nn.dropout(inputs, keep_prob=keep_prob)
    else:
        return inputs

def character_embedding(passage_words):
    pass

def get_attn_params(atten_size, initializer=tf.truncated_normal_initializer):
    '''
    初始化attention训练过程中的参数
    :param atten_size:   the size of attention
    :param initializer:
    :return:    a collection of parameters
    '''
    with tf.variable_scope("attention_weights"):
        params = {"W_u_Q":tf.get_variable("W_u_Q", dtype=tf.float32, shape=(2*atten_size,atten_size),
                                          initializer=initializer()),
                  "W_u_P":tf.get_variable("W_u_P", dtype=tf.float32, shape=(2*atten_size,atten_size),
                                          initializer=initializer()),
                  "W_v_P":tf.get_variable("W_v_P", dtype=tf.float32, shape=(atten_size,atten_size),
                                          initializer=initializer()),
                  "W_v_P_2":tf.get_variable("W_v_P_2", dtype=tf.float32, shape=(2*atten_size,atten_size),
                                            initializer=initializer()),
                  "W_g":tf.get_variable("W_g", dtype=tf.float32, shape=(4*atten_size,4*atten_size),
                                        initializer=initializer()),
                  "W_h_P":tf.get_variable("W_h_P", dtype=tf.float32, shape=(2*atten_size,atten_size),
                                          initializer=initializer()),
                  "W_v_Phat":tf.get_variable("W_v_Phat", dtype=tf.float32, shape=(2*atten_size,atten_size),
                                             initializer=initializer()),
                  "W_h_a":tf.get_variable("W_h_a", dtype=tf.float32, shape=(2*atten_size, atten_size),
                                          initializer=initializer()),
                  "W_v_Q":tf.get_variable("W_v_Q", dtype=tf.float32, shape=(atten_size, atten_size),
                                          initializer=initializer()),
                  "v":tf.get_variable("v", dtype=tf.float32, shape=(atten_size), initializer=initializer())}
    return params

def gated_attention(memory, inputs, states, units, params, self_matching=False,
                    memory_len=None, scope="gated_attention"):
    '''
    实现memory, inputs, state具体的attention pooling操作
    重新计算inputs的表达式
    gated attention-based:  计算passage中的每一个input word与question的相关度，得到新的passage representation
                            inputs = [u_t_P, c_t]*
                            c_t = att(u_Q, u_t_P, v_(t-1)_P) 即：att(memory, inputs, state)
    self-matching:          计算passage中的每一个word与passage的相关度，得到新的passage representation
                            inputs = [v_t_P, c_t]*
                            c_t = att(v_P, v_t_p) 即：att(memory, inputs)
    :param memory:           添加与memory的相关度
    :param inputs:           RNN_cell中的单个input    (batch_size, 2*hidden_size)
    :param states:           RNN_cell中的上一个state  (batch_size, hidden_size)
    :param units:            RNN_cell的 num_units
    :param params:           attention pooling 过程中需要使用的weights，格式：(([list_W], v), W_g)
    :param self_matching:    判断是网络第一层 or self-matching
    :param memory_len:       question_words_len 或者 passage_words_len
    :param scope:
    :return:                 attention pooling 的最终结果，该结果传入RNN_cell中，作为新的input
    '''
    # config = get_small_config()
    with tf.variable_scope(scope):
        weights, W_g = params
        inputs_ = [memory,  inputs]  # layer2: question_encoding, current_passage_word
        # states = tf.reshape(states, (config.train_batch_size, config.hidden_size))
        if not self_matching:
            inputs_.append(states)    # 与att具体公式相关
        # 计算inputs_与params中多个weights的和
        scores = attention(inputs_, units, weights, memory_len=memory_len) # [batch_size, Q]
        # 将score中的每一个数都扩展为一个维度 => [batch_size, Q, 1] 表示memory每一个word的权重
        scores = tf.expand_dims(scores, -1)
        # 将权重与word相乘
        # scores*memory: (batch_size, Q, 1)*(batch_size, Q, 2*hidden_size) = (batch_size, Q, 2*hidden_size)
        attention_pool = tf.reduce_sum(scores* memory, 1)  # (batch_size, 2*hidden_size)
        inputs = tf.concat(1, (inputs, attention_pool))    # (batch_size, 4*hidden_size)
        g_t = tf.sigmoid(tf.matmul(inputs, W_g))
        return g_t * inputs


def attention(inputs, units, weights, scope="attention", memory_len=None, reuse=None):
    '''
    多个W*X的和，并进行softmax操作
    :param inputs:     多个输入，分别于多个weight相乘
    :param units:
    :param weights:    多个weights，以及v.前者与x相乘，v为
    :param scope:
    :param memory_len:
    :param reuse:
    :return:           [batch_size, P] or [batch_size, Q]
    '''
    # config = get_small_config()
    with tf.variable_scope(scope, reuse=reuse):
        outputs_ = []
        weights, v = weights
        for i, (inp, w) in enumerate(zip(inputs, weights)):
            shapes = inp.get_shape().as_list()
            print('attention shape: ', shapes)
            inp = tf.reshape(inp, (-1, shapes[-1]))
            if w is None:
                w = tf.get_variable("w_%d"%i, dtype=tf.float32, shape=[shapes[-1],units],
                                    initializer=tf.contrib.layers.xavier_initializer()) #todo initializer()
            outputs = tf.matmul(inp, w)

            # batch_size = get_batch_size() if shapes[0] is None else shapes[0]
            if shapes[0] is None:
                batch_size = get_batch_size()
            else:
                batch_size = shapes[0]

            # print('attention output: ', outputs, shapes[0], shapes[-1])
            if len(shapes) > 2:
                outputs = tf.reshape(outputs, (batch_size, -1, units))
            # elif len(shapes) == 2 and shapes[0] is config.train_batch_size:
            elif len(shapes) == 2:
                outputs = tf.reshape(outputs, (batch_size, 1, units))
            else:
                outputs = tf.reshape(outputs, (1, batch_size, units))
            outputs_.append(outputs)
        outputs = sum(outputs_)     # todo check here  [batch_size, Q, 2*hidden_size]
        # [batch_size, timestep, hidden_size]
        # todo: 添加bias信息
        scores = tf.reduce_sum(tf.tanh(outputs)*v, [-1])  # [batch_size, Q]
        # todo: mask_attn_score
        # all attention output is softmaxed now
        return tf.nn.softmax(scores)

def pointer(passage, passage_len, question, question_len, cell, params, batch_size, units, scope="pointer_network"):
    '''
    根据最终的passage representation 和question encoding 预测答案的起始位置
    :param passage:         passage representation, 网络最终的输出    [batch_size, timestep, 2*hidden_size]
    :param passage_len:     大小：[train_batch_size], variable lengths for passage length
    :param question:        question encoding, 即第一层网络的question表示，u_Q  [batch_size, timestep, 2*hidden_size]
    :param question_len:
    :param cell:            rnn cell, related to config.cell_fn
    :param params:          当前网络层的两部分weights for attention pooling computation
    :param scope:
    :return:                softmax logits for the answer pointer of the beginning and the end of the answer span
                            每一个位置为最终起始位置的概率
                            [batch_size, 2, P]
    '''
    weights_q, weights_p = params
    passage_shape = passage.get_shape().as_list()
    # 计算 question的新表达，r_q = att(u_q, V_r_Q) V_r_Q is parameter;
    # todo: 添加更多 计算initial state的方法
    initial_state = question_pooling(question)
    # 计算start的概率分布 [batch_size, P]
    inputs = [passage, initial_state]
    p1_logits = attention(inputs, units, weights_p, memory_len=passage_len)  # [batch_size, P]
    # 计算 h_t_a
    scores = tf.expand_dims(p1_logits, -1)
    c_t = tf.reduce_sum(scores*passage, 1)
    _, initial_state_1 = cell(c_t, initial_state)
    # 计算end的概率分布 [batch_size, P]
    inputs = [passage, initial_state_1]
    p2_logits = attention(inputs, units, weights_p, memory_len=passage_len)
    return tf.stack((p1_logits, p2_logits), 1)

def question_pooling(question):
    # question: [batch_size, Q, 2*hidden_size]
    # return [batch_size, 2*hidden_size]
    return tf.reduce_sum(question, 1)


def compute_loss_value(pred_answer_prob, ground_truth_prob):
    '''
    根据答案起始位置的概率分布，计算loss
    :param pred_answer_prob:   [batch_size, 2, P] 预测值，表示passage中每一个word分别为起始位置的概率
    :param ground_truth_prob:  [batch_size, 2, P]
    :return:
    '''
    print('1: ', tf.shape(ground_truth_prob))
    print('2: ', tf.shape(pred_answer_prob))
    ans = ground_truth_prob * tf.log(pred_answer_prob + 1e-8)
    ans = -tf.reduce_sum(ans, 2)  # [batch_size, 2] 每一个question的start end的loss
    ans = tf.reduce_mean(ans, 1)  # [batch_size] average loss for start and end
    ans = tf.reduce_mean(ans)     # a value      average loss across batch size
    return ans

