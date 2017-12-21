import os
import json
import math

from tqdm import tqdm

class DataSet(object):
    def __init__(self, passage, passage_len, question, question_len,
                 answer_index, question_id):
        '''
        以上所有矩阵，每一个元素均为一个sample
        passage:      [passage1, passage2, passage3...] 有重复的passage
        passage_len:  [passage1_len, passage2_len, ...]
        question:     [question1, question2, ...] 依次与passage1, passage2对应
        question_len: [question1_len, question2_len, ...]
        answer_index: [index1, index2, ...] index = [star, end]
        question_id:  [question1_id, question2_id, ...]
        '''
        self.passage = passage
        self.question = question
        self.passage_len = passage_len
        self.question_len = question_len
        self.answer_index = answer_index
        self.question_id = question_id
        self.num_examples = len(question)

    def get_batches(self, batch_size):
        # 根据batch_size的值，返回多个batch的json集合，可直接作为feed_dict
        last_batch_size = 0
        batches = []
        num_batches = int(math.ceil(self.num_examples / batch_size))
        for i in range(num_batches):
            # 计算当前batch的起始范围,左闭右开区间 [0,5) batch_size = 5
            start, end = i, min(i+num_batches, self.num_examples)
            this_batch_size = end - start
            # 构造当前batch的具体data
            batch_data = {}
            batch_data["batch_size"] = this_batch_size
            batch_data["passage_words"] = self.passage[start:end]
            batch_data["question_words"] = self.question[start:end]
            batch_data["passage_words_len"] = self.passage_len[start:end]
            batch_data["question_words_len"] = self.question_len[start:end]
            batch_data["question_id"] = self.question_id[start:end]
            batch_data["answer_index"] = self.answer_index[start:end]
            batches.append(batch_data)
        return batches

def read_data(config, data_type):
    '''
        生成一一对应的passage quesiton answer对，及word_len等信息
        passage_word, question_word
        passage_word_len, question_word_len
        answer_index, question_id
        #todo: 多个answer如何计算
    '''
    data_path = os.path.join(config.data_dir, "data_{}.json".format(data_type))
    with open(data_path, 'r', encoding='utf-8') as fh:
        data = json.load(fh)

    # source
    passage_word = data["passage_word_vec"]
    question_word = data["question_word_vec"]
    answer_index = data["answer_index"]
    question_id = data["question_id"]
    # target
    new_passage_word = []
    new_question_word = []
    new_passage_len = []
    new_question_len = []
    new_answer_index = []
    new_question_id = []


    para_num = len(question_word)
    for pi, qu_to_para in enumerate(tqdm(question_word[0:para_num])):
        passage = passage_word[pi]
        passage_len = len(passage)
        for qi, question in enumerate(qu_to_para):
            question_len = len(question)
            id = question_id[pi][qi]
            # todo: 添加多个答案
            index = answer_index[pi][qi][0]
            # 将处理好的数据对添加到目标矩阵中
            new_passage_word.append(passage)
            new_passage_len.append(passage_len)
            new_question_word.append(question)
            new_question_len.append(question_len)
            new_answer_index.append(index)
            new_question_id.append(id)

    data_set = DataSet(new_passage_word, new_passage_len, new_question_word,
                       new_question_len, new_answer_index, new_question_id)
    return data_set


