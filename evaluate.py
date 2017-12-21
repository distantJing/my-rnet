import sys
import json
import re
import string
from collections import Counter

def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())  # english split
        # return ' '.join(jieba.lcut(text, cut_all=False)) # chinese split, 单词
       # return ' '.join(x for x in text) # chinese split 单字

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def f1_score(prediction, ground_truth):
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


def exact_match_score(prediction, ground_truth):
    return (normalize_answer(prediction) == normalize_answer(ground_truth))


def metric_max_over_ground_truths(metric_fn, prediction, ground_truths):
    scores_for_ground_truths = []
    for ground_truth in ground_truths:
        score = metric_fn(prediction, ground_truth)
        scores_for_ground_truths.append(score)
    return max(scores_for_ground_truths)



def evaluate(dataset, predictions):
    '''
    评估标准答案，预测答案
    :param dataset:       原始数据集，train-v1.1.json, dev-v1.1.json
    :param predictions:   由model.answer_select()得来的 self.pred_answer_text
                          格式： {id:answer} json文件
    :return:  EM, F1
    '''
    f1 = 0
    exact_match = 0
    num_question = 0
    for article in dataset:
        for paragraph in article["paragraphs"]:
            for qa in paragraph['qas']:
                num_question += 1
                # 未预测的答案，或者被丢弃的数据
                if qa['id'] not in predictions:
                    message = 'Unanswerd question ' + qa['id'] + ' will receive score 0.'
                    print(message, file=sys.stderr)
                    continue
                ground_truths = list(map(lambda x: x['text'], qa['answers']))
                prediction = predictions[qa['id']]
                exact_match += metric_max_over_ground_truths(
                    exact_match_score, prediction, ground_truths)
                f1 += metric_max_over_ground_truths(
                    f1_score, prediction, ground_truths)
    exact_match = 100.0 * exact_match / num_question
    f1 = 100.0 * f1 / num_question

    return {'exact_match': exact_match, 'f1': f1}