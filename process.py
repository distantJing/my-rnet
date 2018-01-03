import json
import os
import argparse
import re
import sys
import numpy as np

from tqdm import tqdm

# created by zhijing at 2017/12/11
# generate passages questions at word level
# passage_word = [passage1, passage2, ...]
# question_word =  [
#     [question1_to_p1, question2_to_p1, ...], [question1_to_p2, question2_to_p2, ...], ...
# ]
# answer_index = [
#     [
#         [ [q1_p1_answer1, ...], [q2_p1_ans], ...]
#     ],
#     []
# ]
# answer_index = [0,1] passage_word = ['hi', ',', 'how', ...]
# answer = 'hi', 左闭右开区间


def generate_data():
    '''
    生成 passage_word, question_word, answer_index
    passage_word:  [passage1, passage2]
    question_word: [[q1_to_p1,q2_to_p1],[q1_to_p2,q2_to_p2]]
    question_id:   above
    answer_index:  [
                    [[a1_q1_p1,a2_q1_p1],[a1_q2_p1,a2_q2_p1]],
                    [[a1_q1_p2,a2_q1_p2],[a1_q2_p2,a2_q2_p2]]
                   ]
    '''
    args = get_args()
    process(args)


# 生成数据后，检查数据合法性
# answer_index 包含的片段 是否与 原文答案一致
def check_data(data_type):
    args = get_args()
    data_path = os.path.join(args.target_dir, "data_{}.json".format(data_type))
    data = json.load(open(data_path, 'r', encoding=args.encoding))
    answer_index = data['answer_index']
    answer_text = data['answer_text']
    context_word = data['passage_word']
    invalid_answer = 0
    para_num = len(answer_text)
    for pi, ans2para in enumerate(tqdm(answer_index[0:para_num])):
        for qi, ans2qa in enumerate(ans2para):
            for ani, answer in enumerate(ans2qa):
                start, stop = answer[0], answer[1]
                x_list = context_word[pi][start:stop]
                y_str = answer_text[pi][qi][ani]
                if ''.join(x_list) != ''.join(y_str.split()):
                    invalid_answer += 1
    print(data_type, " invalid_answer: ", invalid_answer)


# 添加参数信息
def get_args():
    parser = argparse.ArgumentParser()
    home = os.getcwd()
    source_dir = os.path.join(home, "data", "squad")
    target_dir = os.path.join(home, "data", "squad", "small_data")
    glove_dir = os.path.join(home, "data", "glove")
    parser.add_argument('-s', "--source_dir", default=source_dir)
    parser.add_argument('-t', "--target_dir", default=target_dir)
    parser.add_argument("--glove_dir", default=glove_dir)
    parser.add_argument("--glove_vector_size", default=200, type=int)
    parser.add_argument("--glove_corpus", default="6B")
    parser.add_argument("--tokenizer", default="PTB")     ###################3
    parser.add_argument("--data_version", default="v1.1")
    parser.add_argument("--encoding", default='utf-8')
    # todo: put more args here
    return parser.parse_args()


def process(args):
    if not os.path.exists(args.target_dir):
        os.mkdir(args.target_dir)
    process_each(args, 'train', 'train')
    process_each(args, 'dev', 'dev')


# 对当前json文件进行处理
def process_each(args, data_type, out_name, start_ratio=0.0, stop_ratio=1.0):
    source_path = os.path.join(args.source_dir, "{}-{}.json".format(data_type, args.data_version))
    print("process {}-{}.json, splitting words, generating answer index ...".format(data_type, args.data_version))
    source_data = json.load(open(source_path, 'r', encoding=args.encoding))

    passage_word = []
    passage_char = []
    questions_word= []
    questions_char = []
    questions_id = []
    answer_index = []
    answersss = []

    start_ai = int(len(source_data['data']) * start_ratio)
    stop_ai = int(len(source_data['data']) * stop_ratio)
    stop_ai = 1         # 调整stop_ai的大小，使用小数据进行测试
    invalid_answer = 0  # 统计非法答案，情况包括但不局限于：答案是文中某个单词的前缀

    for ai, article in enumerate(tqdm(source_data['data'][start_ai:stop_ai])):
        for pi, para in enumerate(article['paragraphs']):
            context = para['context']
            context_word, context_char = get_words_and_char(context)

            questions_to_context_word = []
            questions_to_context_char = []
            question_to_context_id = []
            answers_to_context = []
            answerss = []
            for qa in para['qas']:
                question = qa['question']
                question_word, question_char = get_words_and_char(question)

                answers_to_question = []
                answers = []
                for answer in qa['answers']:
                    answer_text = answer['text']
                    answer_start = answer['answer_start']
                    answer_index_start, answer_index_stop = find_answer_index(context, context_word, answer_text, answer_start)
                    if answer_index_start == -1:
                        invalid_answer += 1
                        continue
                    answers_to_question.append([answer_index_start, answer_index_stop])
                    answers.append(answer_text)

                if len(answers_to_question) != 0:
                    questions_to_context_word.append(question_word)
                    questions_to_context_char.append(question_char)
                    question_to_context_id.append(qa['id'])
                    answers_to_context.append(answers_to_question)
                    answerss.append(answers)
            if len(questions_to_context_word) !=0:
                passage_word.append(context_word)
                passage_char.append(context_char)
                questions_word.append(questions_to_context_word)
                questions_char.append(questions_to_context_char)
                questions_id.append(question_to_context_id)
                answer_index.append(answers_to_context)
                answersss.append(answerss)

    data = {"passage_word":passage_word, "passage_char":passage_char, "question_word":questions_word,
            "question_char":questions_char, "answer_index":answer_index, "answer_text":answersss,
            "question_id":questions_id}
    print("invalid answer: ",invalid_answer)
    save(args, data, out_name, 'generate')


def save(args, data, data_type, save_type):
    print("saving {} {} data".format(save_type, data_type))
    if save_type == 'generate':
        data_path = os.path.join(args.target_dir, "data_{}.json".format(data_type))
    elif save_type == 'encode':
        data_path = os.path.join(args.target_dir, "data_{}d_{}.json".format(args.glove_vector_size,data_type))
    # print('data_path: ', data_path)
    json.dump(data, open(data_path, 'w'))


# 将SQuAD中的start_answer转化为word层面
def find_answer_index(context, context_word, answer_text, answer_start):
    temp, _ = get_words_and_char(context[0:answer_start])
    answer_start_index = len(temp)

    answer_word, _ = get_words_and_char(answer_text)
    window_len = len(answer_word)
    # print('answer_word: ', answer_word)
    # print('window_len: ', window_len)
    # print('answer_start_index: ',answer_start_index, context_word[answer_start_index])
    if context_word[answer_start_index:min(answer_start_index + window_len, len(context_word))] == answer_word:
        # print(context_word[answer_start_index:answer_start_index + window_len])
        return (answer_start_index, answer_start_index + window_len)
    else:
        return (-1, -1)


# input：一段英文文本
# output: 将输入文本按照单词分割，将单词按char分割
# word tokenize: 处理文本中的引号，部分单词特殊处理，如1-2， 1/4
def get_words_and_char(text):
    words = word_tokenize(text)
    char = [[list(char) for char in word] for word in words]
    return words, char


# input：tokens 单词序列
# output：特殊处理后的单词序列
# 特殊处理： 见 l
def process_tokens(temp_tokens):
    tokens = []
    for token in temp_tokens:
        flag = False
        l = ("-", "\u2212", "\u2014", "\u2013", "/", "~", '"', "'", "\u201C", "\u2019", "\u201D", "\u2018", "\u00B0")
        # \u2013 is en-dash. Used for number to nubmer
        # l = ("-", "\u2212", "\u2014", "\u2013")
        # l = ("\u2013",)
        tokens.extend(re.split("([{}])".format("".join(l)), token))
    return tokens


def word_tokenize(text):
    import nltk
    tokens = [token.replace("''", '"').replace("``", '"') for token in nltk.word_tokenize(text)]
    tokens = process_tokens(tokens)
    return tokens


def test_find_answer_index():
    context = "Super Bowl 50 was an American football game to determine the champion of the National Football League (NFL) for the 2015 season. The American Football Conference (AFC) champion Denver Broncos defeated the National Football Conference (NFC) champion Carolina Panthers 24–10 to earn their third Super Bowl title. The game was played on February 7, 2016, at Levi's Stadium in the San Francisco Bay Area at Santa Clara, California. As this was the 50th Super Bowl, the league emphasized the \"golden anniversary\" with various gold-themed initiatives, as well as temporarily suspending the tradition of naming each Super Bowl game with Roman numerals (under which the game would have been known as \"Super Bowl L\"), so that the logo could prominently feature the Arabic numerals 50."
    answer_text = "gold"
    answer_start = 521
    context_word, context_char = get_words_and_char(context)
    answer_index_start, answer_index_stop = find_answer_index(context, context_word, answer_text, answer_start)
    print('s,e: ', answer_index_start, answer_index_stop)
    print('context_word: ', context_word)


# 将passage_word, question_word 添加glove信息，word2vec
def encode():
    args = get_args()
    encode_each(args, 'dev', 'dev')
    encode_each(args, 'train', 'train')


def encode_each(args, data_type, out_name, start_ratio=0.0, stop_ratio=1.0):
    print("encode data_{}.json".format(data_type))
    source_path = os.path.join(args.target_dir, "data_{}.json".format(data_type))
    source_data = json.load(open(source_path, 'r', encoding=args.encoding))
    target_data = source_data
    passage_word = source_data["passage_word"]
    question_word = source_data["question_word"]

    # 加载glove字典文件
    word2vec_dict = get_word2vec_dict(args)

    target_data["passage_word_vec"] = get_word2vec_passage(args, word2vec_dict, passage_word)
    target_data["question_word_vec"] = get_word2vec_question(args, word2vec_dict, question_word)
    save(args, target_data, data_type, 'encode')


def get_word2vec_passage(args, word2vec_dict, passage_word):
    passage_word_vec = passage_word
    para_num = len(passage_word)
    num_out_of_vocab = 0
    for pi, para in enumerate(tqdm(passage_word[0:para_num])):
        para_vector = []
        for word in para:
            if len(word2vec_dict.get(word, "")) != 0:
                para_vector.append(word2vec_dict[word])
            else: #  out of vacab
                para_vector.append(list(np.zeros(args.glove_vector_size)))
                num_out_of_vocab += 1
        passage_word_vec[pi] = para_vector
    print('passage num_out_of_vocab: ', num_out_of_vocab)
    return passage_word_vec


def get_word2vec_question(args, word2vec_dict, question_word):
    question_word_vec = question_word
    para_num = len(question_word)
    num_out_of_vocab = 0
    for pi, qu_to_paras in enumerate(tqdm(question_word[0:para_num])):
        questions_to_para = []
        for qi, question in enumerate(qu_to_paras):
            question_vec = []
            for word in question:
                if len(word2vec_dict.get(word, "")) != 0:
                    question_vec.append(word2vec_dict[word])
                else: # out of vocab
                    question_vec.append(list(np.zeros(args.glove_vector_size)))
                    num_out_of_vocab += 1
            question_word_vec[pi][qi] = question_vec
    print('question num_out_of_vocab: ', num_out_of_vocab)
    return question_word_vec


def test_get_word2vec():
    args = get_args()
    args.glove_vector_size = 2
    word2vec_dict = {'hi':[3,3],'my':[1,1],'name':[2,2]}
    passage_word = [['hi','my','zhijing'],['name','hi']]
    passage_vec = get_word2vec_passage(args, word2vec_dict, passage_word)
    question_word = [ [['p1q1','hi'],['p1q2','my']], [['p2q1','name']] ]
    question_vec = get_word2vec_question(args, word2vec_dict, question_word)
    print('passage: ', passage_vec)
    print('question: ', question_vec)


def get_word2vec_dict(args):
    glove_path = os.path.join(args.glove_dir, "glove.{}.{}d.txt".format(args.glove_corpus, args.glove_vector_size))
    sizes = {'6B': int(4e5), '42B': int(1.9e6), '840B': int(2.2e6), '2B': int(1.2e6)}
    total = sizes[args.glove_corpus]
    word2vec_dict = {}
    with open(glove_path, 'r', encoding=args.encoding) as fh:
        for line in tqdm(fh, total=total):
            array = line.lstrip().rstrip().split()
            word = array[0]
            vector = list(map(float, array[1:]))
            word2vec_dict[word] = vector
    print("{} of word vocab have corresponding in {}".format(len(word2vec_dict),glove_path))
    return word2vec_dict


if __name__ == "__main__":
    # test()
    generate_data()
    encode()

    # if sys.argv[0] == 'squad':
    #     generate_data()  # 生成passage_word, question_word, answer_index
    # elif sys.argv[0] == 'glove':
    #     encode()         # 将generate_data()中的word 替换为向量表示
    # else :
    #     encode()
        # test_get_word2vec()

    # check_data('train')
    # check_data('dev')