import json
import os
import argparse
import re

from tqdm import tqdm

def generate_data():
    args = get_args()
    process(args)

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

def get_args():
    parser = argparse.ArgumentParser()
    home = os.getcwd()
    source_dir = os.path.join(home, "data")
    target_dir = os.path.join(home, "data")
    glove_dir = os.path.join(home, "data", "glove")
    parser.add_argument('-s', "--source_dir", default=source_dir)
    parser.add_argument('-t', "--target_dir", default=target_dir)
    parser.add_argument("--glove_dir", default=glove_dir)
    parser.add_argument("--glove_vector_size", default=300, type=int)
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

def process_each(args, data_type, out_name, start_ratio=0.0, stop_ratio=1.0):
    source_path = os.path.join(args.source_dir, "{}-{}.json".format(data_type, args.data_version))
    print('source_path: ', source_path)
    source_data = json.load(open(source_path, 'r', encoding=args.encoding))

    passage_word = []
    passage_char = []
    questions_word= []
    questions_char = []
    answer_index = []
    answersss = []

    start_ai = int(len(source_data['data']) * start_ratio)
    stop_ai = int(len(source_data['data']) * stop_ratio)
    # stop_ai = 1
    invalid_answer = 0

    for ai, article in enumerate(tqdm(source_data['data'][start_ai:stop_ai])):
        for pi, para in enumerate(article['paragraphs']):
            context = para['context']
            context_word, context_char = get_words_and_char(context)

            questions_to_context_word = []
            questions_to_context_char = []
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
                    answers_to_context.append(answers_to_question)
                    answerss.append(answers)
            if len(questions_to_context_word) !=0:
                passage_word.append(context_word)
                passage_char.append(context_char)
                questions_word.append(questions_to_context_word)
                questions_char.append(questions_to_context_char)
                answer_index.append(answers_to_context)
                answersss.append(answerss)

    data = {"passage_word":passage_word, "passage_char":passage_char, "question_word":questions_word,
            "question_char":questions_char, "answer_index":answer_index, "answer_text":answersss}
    print("invalid answer: ",invalid_answer)
    save(args, data, out_name)

def save(args, data, data_type):
    data_path = os.path.join(args.target_dir, "data_{}.json".format(data_type))
    # print(data)
    # print('data_path: ', data_path)
    json.dump(data, open(data_path, 'w'))

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
    #
    # if window_len == 1:
    #     if context_word[answer_start_index] == answer_word[0]:
    #         return (answer_start_index, answer_start_index)
    #     else:
    #         return (-1, -1)
    # else:
    #     if context_word[answer_start_index:min(answer_start_index + window_len, len(context_word))] == answer_word:
    #         # print(context_word[answer_start_index:answer_start_index + window_len])
    #         return (answer_start_index, answer_start_index+window_len)
    #     else:
    #         return (-1, -1)

def get_words_and_char(text):
    words = word_tokenize(text)
    char = [[list(char) for char in word] for word in words]
    return words, char

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

def test():
    context = "Super Bowl 50 was an American football game to determine the champion of the National Football League (NFL) for the 2015 season. The American Football Conference (AFC) champion Denver Broncos defeated the National Football Conference (NFC) champion Carolina Panthers 24–10 to earn their third Super Bowl title. The game was played on February 7, 2016, at Levi's Stadium in the San Francisco Bay Area at Santa Clara, California. As this was the 50th Super Bowl, the league emphasized the \"golden anniversary\" with various gold-themed initiatives, as well as temporarily suspending the tradition of naming each Super Bowl game with Roman numerals (under which the game would have been known as \"Super Bowl L\"), so that the logo could prominently feature the Arabic numerals 50."
    answer_text = "gold"
    answer_start = 521
    context_word, context_char = get_words_and_char(context)
    answer_index_start, answer_index_stop = find_answer_index(context, context_word, answer_text, answer_start)
    print('s,e: ', answer_index_start, answer_index_stop)
    print('context_word: ', context_word)


if __name__ == "__main__":
    # test()
    # generate_data()
    check_data('train')
    check_data('dev')