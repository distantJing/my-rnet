import tensorflow as tf

from tensorflow.python.ops.rnn_cell import RNNCell
from tensorflow.python.ops import math_ops
from layer import *


class gated_attention_Wrapper(RNNCell):
    def __init__(self, num_units, memory, params, cell_fn,
                 self_matching=False,
                 memory_len=None,
                 is_training=True,
                 reuse=None):  # TODO: REUSE???
        '''
        :param num_units:
        :param cell_fn:  default: tf.nn.rnn_cell.GRUCell 或者新定义的cell
        memory: 相当于 question encoding u_Q
        '''
        super(gated_attention_Wrapper, self).__init__()    #todo what's this mean？？？
        cell = cell_fn
        print(cell)
        self._cell = cell(num_units)     #cell为新定义的或默认的# todo
        self._num_units = num_units
        self._activation = math_ops.tanh
        # todo: self._kernel_initializer =
        # todo: self._bias_initialzer =
        self._attention = memory   # question encoding
        self._params = params
        self._self_matching = self_matching
        self._memory_len = memory_len
        self._is_training = is_training

    @property
    def state_size(self):
        return self._num_units

    @property
    def output_size(self):
        return self._num_units

    def call(self, inputs, state):
        """Gated recurrent unit (GRU) with nunits cells."""
        with tf.variable_scope("attention_pool"):
            # 实现memory, inputs, state具体的attention pooling操作
            # 重新计算inputs的表达式
            # gated attention-based:  计算passage中的每一个word与question的相关度，得到新的passage representation
            #                         inputs = [u_t_P, c_t]*
            #                         c_t = att(u_Q, u_t_P, v_(t-1)_P) 即：att(memory, inputs, state)
            # self-matching:          计算passage中的每一个word与passage的相关度，得到新的passage representation
            #                         inputs = [v_t_P, c_t]*
            #                         c_t = att(v_P, v_t_p) 即：att(memory, input)
            inputs = gated_attention(self._attention,
                                     inputs,
                                     state,
                                     self._num_units,
                                     params=self._params,
                                     self_matching = self._self_matching,
                                     memory_len=self._memory_len)
        output, new_state = self._cell(inputs, state)
        return output, new_state

