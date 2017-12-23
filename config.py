import tensorflow as tf
import os

from main import main as m

flags = tf.app.flags

# names and directories  模型名称，文件路径
flags.DEFINE_string("model_name", "basic", "Model name [basic]")
flags.DEFINE_string("data_dir", "data/squad", "Data dir [data/squad]")
flags.DEFINE_string("run_id", "0", "to save model")
flags.DEFINE_string("out_base_dir", "out", "out base dir [out]")
flags.DEFINE_string("answer_path", "", "answer path []")
flags.DEFINE_string("load_path", "", "load path []")  # 已有模型地址
flags.DEFINE_string("glove_dir", "", "")

# device placement 使用CPU or GPU
flags.DEFINE_string("device", "/cpu:0", "default device for summing gradients. [/cpu:0]")

# essential training and test options 运行方式，是否加载已有模型
flags.DEFINE_string("mode", "train", "train | test | debug")
flags.DEFINE_boolean("load", False, "load saved data? [True]") # 默认加载已有model

# load data glove相关
flags.DEFINE_integer("glove_vector_size", 200, "glove vector size")

# training / test parameters  训练参数
flags.DEFINE_integer("train_batch_size",5, "batch size []")
flags.DEFINE_integer("num_epochs", 12, "total number of epochs for training [12]")
flags.DEFINE_string("num_steps", 20000, "number of steps []")
flags.DEFINE_integer("load_step", 0, "load step [0]")
flags.DEFINE_integer("init_lr", 1, "initial learning rate []")
flags.DEFINE_integer("input_keep_prob", 0.8, "keep prob for the dropout of LSTM weights []")
flags.DEFINE_integer("hidden_size", 75, "hidden size")
flags.DEFINE_integer("max_question_word", 50, "a question has 100 words at most")
flags.DEFINE_integer("max_passage_word", 500, "a passage has 300 words at most ")
flags.DEFINE_integer("word_emb_size", 200, "word2vec length, related to glove")
flags.DEFINE_integer("char_emb_size", 10, "char-level word embedding size")
flags.DEFINE_string("cell_fn", tf.nn.rnn_cell.GRUCell, "recurrent unit")

# optimizations


# longing and saving options
flags.DEFINE_integer("log_period", 100, "log period")
flags.DEFINE_integer("eval_period", 1000, "eval period")
flags.DEFINE_integer("save_period", 1000, "save period")
flags.DEFINE_integer("max_to_keep", 20, "maximum number of recent checkpoints to keep, save model")

# threshold for speed and less memory usage
# advanced training options

# ablation options
flags.DEFINE_bool("use_char_emb", False, "use char emb? [True]")
# flags.DEFINE_bool("")

def main(_):
    config = flags.FLAGS
    config.out_dir = os.path.join(config.out_base_dir, config.model_name, str(config.run_id).zfill(2))
    m(config)

if __name__ == "__main__":
    tf.app.run()