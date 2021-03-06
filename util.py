import os
import mmap
import torch
import time
import json
import random
import jieba
from tqdm import tqdm
import numpy as np
import torch.nn as nn
from genhtml import GenHtml
from datetime import timedelta
from torch.autograd import Variable
from nltk.tokenize import WordPunctTokenizer
import math
from multiprocessing import Pool
from torch.nn import Module, init
from torch.nn.parameter import Parameter
from torch.nn import functional as F

def set_seed(seed=7):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

def get_vocab(raw_dataset):
    vocab = []
    for example in raw_dataset:
        vocab.extend(example[0].split())
    vocab = set(vocab)
    print("vocab size: ", len(vocab))
    return vocab

def read_cataloge(data_path):
    labels_set = set()
    label_columns = {}
    with open(data_path, mode="r", encoding="utf-8") as f:
        for line_id, line in enumerate(f):
            try:
                line = line.replace('\ufeff','').strip().split("\t")
                if line_id == 0:  #####todo: title
                    for i, column_name in enumerate(line):
                        label_columns[column_name] = i
                    continue
                label = int(line[label_columns["label"]])
                labels_set.add(label)
            except:
                pass
    return label_columns

def get_query_matrix(args):
    if args.attention_layer == 'mpa' or args.attention_layer == 'mpoa' or args.attention_layer == 'm_pol_untrain_a':
        with open(args.query_matrix_path, encoding='utf-8') as f:
            emo_vector = json.load(f)
        querys = torch.FloatTensor(args.num_classes, args.attention_query_size).cuda()
        for i in range(args.num_classes):
            querys[i] = torch.Tensor(emo_vector[str(i)]).cuda()
    elif args.attention_layer == 'm_pre_orl_a' or args.attention_layer == 'm_pre_orl_pun_a':
        querys = torch.empty(args.num_classes, args.attention_query_size)
        nn.init.orthogonal_(querys)
    elif args.attention_layer == 'm_a':
        if args.m_a_type == 'rand':
            querys = torch.rand(args.num_classes, args.attention_query_size)
        elif args.m_a_type == 'FloatTensor':
            querys = torch.FloatTensor(args.num_classes, args.attention_query_size)
    elif args.attention_layer == 'att' or args.attention_layer == 'none' :
        querys = None
    else:
        print('error, attention layer type is error')
    query_matrix = nn.Embedding(args.num_classes, args.attention_query_size)
    query_matrix.weight = nn.Parameter(querys)
    return query_matrix

def mkdir(path):
    folder = os.path.exists(path)
    if not folder:
       os.makedirs(path)


def padding_triples(triple, number):
    return triple + [['_PAD_', '_PAD_', '_PAD_']] * (number-len(triple))

def padding_id2(id2_list, number):
    return id2_list + [-1] * (number-len(id2_list))


class Dataset(object):
    def __init__(self, path):
        self.path = path

    def build_and_save(self, args, workers_num, columns, vocab, vocab_entity, vocab_relation):
        lines_num = get_num_lines(self.path)
        print(lines_num)
        print("Starting %d workers for building datasets ... " % workers_num)
        assert (workers_num >= 1)
        if workers_num == 1:
            self.worker(args, columns, vocab, vocab_entity, vocab_relation, 1, 0, lines_num)
        else:
            pool = Pool(workers_num)
            datasets = []
            for i in range(workers_num):
                start = i * lines_num // workers_num
                if start == 0:
                    start = 1
                end = (i + 1) * lines_num // workers_num
                h = pool.apply_async(func = self.worker, args=[args, columns, vocab, vocab_entity, vocab_relation, i, start, end])
                for i in h.get():
                    datasets.append(i)
            pool.close()
            pool.join()
            print(len(datasets))
        return datasets

    def worker(self, args, columns, vocab, vocab_entity, vocab_relation, proc_id, start, end):
        print("Starting building　%d datasets ... " % proc_id)
        dataset = []
        pos = 0
        with open(self.path, mode="r", encoding="utf-8") as f:
            for line_id, line in enumerate(f):
                if line_id >= start:
                    if line_id < end:
                        line = line.strip().split('\t')
                        if len(line) == 6:
                            pos = pos + 1
                            tokens_triples = []
                            id2_list = [[]] * args.seq_length
                            all_triples = [[]] * args.seq_length
                            label = int(line[columns["label"]])
                            text = line[columns["text"]]
                            triples = line[columns["triples"]]
                            context = line[columns["context"]]
                            id2 = line[columns["id2"]]
                            triples=eval(triples)
                            triples = [i for i in triples if i != '']
                            id2 = eval(id2)
                            if args.language_type == 'zh':
                                text1 = jieba.lcut(text)
                                text1_context = jieba.lcut(context)
                            else:
                                text1 = WordPunctTokenizer().tokenize(text.lower())
                                text1_context = WordPunctTokenizer().tokenize(context.lower())

                            for i in range(len(id2)):
                                for j in range(len(text1)):
                                    if id2[i] == 1 and triples[i][0] == text1[j]:
                                        id2_list[j] = id2_list[j] + [id2[i]]
                                        all_triples[j] = all_triples[j] + [triples[i]]
                                        break
                                    if id2[i] == 2 and triples[i][1] == text1[j]:
                                        id2_list[j] = id2_list[j] + [id2[i]]
                                        all_triples[j] = all_triples[j] + [triples[i]]
                                        break

                            for i in range(len(all_triples)):
                                all_triples[i] = padding_triples(all_triples[i], args.triples_number)
                            for i in range(len(id2_list)):
                                id2_list[i] = padding_id2(id2_list[i], args.triples_number)

                            if args.language_type == 'zh':
                                tokens = [vocab.get(t) for t in text1]
                                tokens_context = [vocab.get(t) for t in text1_context]
                            else:
                                tokens = [vocab.get(t) for t in WordPunctTokenizer().tokenize(text.lower())]
                                tokens_context = [vocab.get(t) for t in WordPunctTokenizer().tokenize(text1_context.lower())]

                            for i in all_triples:
                                tokens_triples0 = []
                                for j in i:
                                    l = [vocab_entity.get(j[l]) if l < 2 else vocab_relation.get(j[l]) for l in range(len(j))]
                                    tokens_triples0.append(l)
                                tokens_triples.append(tokens_triples0)

                            if len(tokens) > args.seq_length:
                                tokens = tokens[:args.seq_length]
                            if len(tokens_context) > args.context_length:
                                tokens_context = tokens_context[:args.context_length]
                            length = len(tokens)
                            while len(tokens) < args.seq_length:
                                tokens.append(0)
                            while len(tokens_context) < args.context_length:
                                tokens_context.append(0)
                            #print("LLLLLLLLLLLLLLLL")
                            #print(tokens_context)
                            dataset.append((tokens, tokens_context, tokens_triples, id2_list, label, length, text))
                        else:
                            pass
        return dataset


# Parallel read dataset.
def read_dataset(args,path, columns, vocab, vocab_entity, vocab_relation):
    dataset = Dataset(path)
    t=dataset.build_and_save(args, args.processes_num, columns, vocab, vocab_entity, vocab_relation)
    print(len(t))
    # print(":::::::::::::::::::::::::::")
    # print(t)
    return t

def batch_loader(batch_size, input_ids, context_ids, triples_ids, label_ids, length_ids, id2_ids):
    instances_num = input_ids.size()[0]
    for i in range(instances_num // batch_size):
        input_ids_batch = input_ids[i * batch_size: (i + 1) * batch_size, :]
        context_ids_batch = context_ids[i * batch_size: (i + 1) * batch_size]
        label_ids_batch = label_ids[i * batch_size: (i + 1) * batch_size]
        length_ids_batch = length_ids[i * batch_size: (i + 1) * batch_size]
        triples_ids_batch = triples_ids[i * batch_size: (i + 1) * batch_size]
        id2_ids_batch = id2_ids[i * batch_size: (i + 1) * batch_size]
        yield input_ids_batch, context_ids_batch, triples_ids_batch, id2_ids_batch, label_ids_batch, length_ids_batch

    if instances_num > instances_num // batch_size * batch_size:
        input_ids_batch = input_ids[instances_num // batch_size * batch_size:, :]
        context_ids_batch = context_ids[instances_num // batch_size * batch_size:]
        label_ids_batch = label_ids[instances_num // batch_size * batch_size:]
        length_ids_batch = length_ids[instances_num // batch_size * batch_size:]
        triples_ids_batch = triples_ids[instances_num // batch_size * batch_size:]
        id2_ids_batch = id2_ids[instances_num // batch_size * batch_size:]
        yield input_ids_batch, context_ids_batch, triples_ids_batch, id2_ids_batch, label_ids_batch, length_ids_batch


def get_num_lines(file_path):
    fp = open(file_path, "r+")
    buf = mmap.mmap(fp.fileno(), 0)
    lines = 0
    while buf.readline():
        lines += 1
    return lines


def get_embedding_matrix(args, vocab, normalization=False):
    glove_path = args.pretrained_w2v_model_path
    glove_vectors = {}
    if args.pretrain_model_type == 'w2v':
        #with open(glove_path, encoding='gbk') as glove_file:
        with open(glove_path, encoding='gb18030') as glove_file:
            for line in tqdm(glove_file, total=get_num_lines(glove_path)):
                split_line = line.rstrip().split()
                word = split_line[0]
                if len(split_line) != (args.w2v_embedding_dim + 1) or word not in vocab.w2i:
                    continue
                assert (len(split_line) == args.w2v_embedding_dim + 1)
                vector = np.array([float(x) for x in split_line[1:]], dtype="float32")
                if normalization:
                    vector = vector / np.linalg.norm(vector)
                assert len(vector) == args.w2v_embedding_dim
                glove_vectors[word] = vector
        print("Number of pre-trained word vectors loaded: ", len(glove_vectors))
        all_embeddings = np.array(list(glove_vectors.values()))
        embeddings_mean = float(np.mean(all_embeddings))
        embeddings_stdev = float(np.std(all_embeddings))
        print("Embeddings mean: ", embeddings_mean)
        print("Embeddings stdev: ", embeddings_stdev)
        embedding_matrix = torch.FloatTensor(vocab.size, args.w2v_embedding_dim).normal_(embeddings_mean, embeddings_stdev)
        for i in range(2, vocab.size):
            word = vocab.i2w[i]
            if word in glove_vectors:
                embedding_matrix[i] = torch.FloatTensor(glove_vectors[word])
        if normalization:
            for i in range(vocab.size):
                embedding_matrix[i] = embedding_matrix[i] / float(np.linalg.norm(embedding_matrix[i]))
        embeddings = nn.Embedding(vocab.size, args.w2v_embedding_dim, padding_idx=0)
        embeddings.weight = nn.Parameter(embedding_matrix)
    else:
        embeddings = nn.Embedding(vocab.size, args.w2v_embedding_dim, padding_idx=0)
    return embeddings

def get_triples_embedding_matrix(args, path, vocab, normalization=False):
    triples_vectors = {}
    with open(path, encoding='utf-8') as glove_file:
        for line in tqdm(glove_file, total=get_num_lines(path)):
            split_line = line.replace(","," ").rstrip().split()
            word = split_line[0]
            if len(split_line) != (args.triples_embedding_dim + 1) or word not in vocab.w2i:
                continue
            assert (len(split_line) == args.triples_embedding_dim + 1)
            vector = np.array([float(x) for x in split_line[1:]], dtype="float32")
            if normalization:
                vector = vector / np.linalg.norm(vector)
            assert len(vector) == args.triples_embedding_dim
            triples_vectors[word] = vector
    print("Number of pre-trained word vectors loaded: ", len(triples_vectors))
    all_embeddings = np.array(list(triples_vectors.values()))
    embeddings_mean = float(np.mean(all_embeddings))
    embeddings_stdev = float(np.std(all_embeddings))
    print("triples Embeddings mean: ", embeddings_mean)
    print("triples Embeddings stdev: ", embeddings_stdev)
    embedding_matrix = torch.FloatTensor(vocab.size, args.triples_embedding_dim).normal_(embeddings_mean,embeddings_stdev)

    for i in range(0, vocab.size):
        word = vocab.i2w[i]
        if word in triples_vectors:
            embedding_matrix[i] = torch.FloatTensor(triples_vectors[word])
    if normalization:
        for i in range(vocab.size):
            embedding_matrix[i] = embedding_matrix[i] / float(np.linalg.norm(embedding_matrix[i]))
    if args.concat_mode=="concat":
        embeddings = nn.Embedding(vocab.size, args.triples_embedding_dim, padding_idx=0)
        embeddings.weight = nn.Parameter(embedding_matrix)
        return embeddings
    else:
        return embedding_matrix


class Vocab(object):
    def __init__(self):
        self.w2i = {}
        self.i2w = []
        self.w2c = {}
        self.size = 0
    def load(self, vocab_path, is_quiet=False):
        with open(vocab_path, mode="r", encoding="utf-8") as reader:
            for index, line in enumerate(reader):
                try:
                    w = line.strip().split()[0]
                    if w in self.w2i:
                        print(w)
                    self.w2i[w] = index
                    self.i2w.append(w)
                except:
                    print(w)
                    self.w2i["???" + str(index)] = index
                    self.i2w.append("???" + str(index))
                    if not is_quiet:
                        print("Vocabulary file line " + str(index + 1) + " has bad format token")
            assert len(self.w2i) == len(self.i2w)
        if not is_quiet:
            self.size = len(self.w2i)
            print("Vocabulary Size: ", self.size)
    def get(self, w):
        return self.w2i.get(w)

def get_time_dif(start_time):
    """获取已使用时间"""
    end_time = time.time()
    time_dif = end_time - start_time
    return timedelta(seconds=int(round(time_dif)))

def using_GPU_num(gpu_num):
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu_num

def saveSenResult(x_test, y_test_cls, y_pred_cls, args, weights):
    """获得预测结果"""
    writer_true = open(os.path.join(args.output_result_path,'true_test.txt'), encoding='utf-8', mode='w')
    writer_false = open(os.path.join(args.output_result_path,'false_test.txt'),encoding='utf-8', mode='w')
    writer_true.write("预测\t真实\t句子\n")
    writer_false.write("预测\t真实\t句子\n")
    data_len = len(x_test)
    squ = []
    for i in range(data_len):
        if y_test_cls[i] == y_pred_cls[i]:
            writer_true.write(str(y_pred_cls[i]) + "\t" + str(y_test_cls[i]) + "\t" + str(x_test[i]) + "\n")
        else:
            writer_false.write(str(y_pred_cls[i]) + "\t" + str(y_test_cls[i]) + "\t" + str(x_test[i]) + "\n")
        squ.append(str(x_test[i]).split(' '))
    if args.attention_layer == 'none':
        return
    dic = {}
    dic['sequences'], dic['attention_weights'], dic['rea_labels'], dic[
        'pre_labels'] = squ, weights, y_test_cls, y_pred_cls
    with open(os.path.join(args.output_result_path,"attn_data.json"), 'w', encoding='utf-8') as fw:
        json.dump(dic, fw, ensure_ascii=False, indent=4)
    gh = GenHtml()
    gh.gen(dic, os.path.join(args.output_result_path,'attention.html'), args)

def sort_batch_by_length(tensor: torch.Tensor, sequence_lengths: torch.Tensor):
    # # This is the equivalent of zipping with index, sorting by the original sequence lengths and returning the now sorted indices.
    if not isinstance(tensor, torch.Tensor) or not isinstance(sequence_lengths, torch.Tensor):
        raise ConfigurationError("Both the tensor and sequence lengths must be torch.Tensors.")
    sorted_sequence_lengths, permutation_index = sequence_lengths.sort(0, descending=True)
    sorted_tensor = tensor.index_select(0, permutation_index)
    index_range = torch.arange(0, len(sequence_lengths), device=sequence_lengths.device)
    _, reverse_mapping = permutation_index.sort(0, descending=False)
    restoration_indices = index_range.index_select(0, reverse_mapping)
    return sorted_tensor, sorted_sequence_lengths, restoration_indices, permutation_index

def last_dim_softmax(vector, mask):
    result = torch.nn.functional.softmax(vector*mask, dim = -1)
    result = result * mask
    result = result / (result.sum(dim = 1, keepdim = True) + 1e-13)
    return result
