import torch
import torch.nn as nn
import torch.optim as optim

import os
import json
import time
import h5py
import random
import argparse
from sklearn import metrics

from util import Vocab
from util import mkdir
from util import set_seed
from util import read_dataset
from util import saveSenResult
from util import read_cataloge
from util import batch_loader
from util import get_time_dif
from util import get_query_matrix
from util import get_embedding_matrix
from util import get_triples_embedding_matrix
#from util import get_max_triples_number
import warnings
from model import RNNSequenceClassifier

warnings.filterwarnings("ignore")

print("PyTorch version:")
print(torch.__version__)
print("GPU Detected:")
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
print(torch.cuda.is_available())
using_GPU = True


# Training phase.
def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # Path options.
    parser.add_argument("--pretrained_w2v_model_path", required=True, type=str,
                        help="Path of the tence w2v pretrained model.")
    parser.add_argument("--query_matrix_path", required=True, type=str,
                        help="Path of the query matrix.")
    parser.add_argument("--summary_result_path", required=True, type=str,
                        help="Path of the output model.")
    parser.add_argument("--output_result_path", required=True, type=str,
                        help="Path of the output result.")
    parser.add_argument("--train_path", type=str, required=True,
                        help="Path of the trainset.")
    parser.add_argument("--dev_path", type=str, required=True,
                        help="Path of the devset.")
    parser.add_argument("--test_path", type=str, required=True,
                        help="Path of the testset.")
    parser.add_argument("--vocab_path", type=str, required=True,
                        help="Path of the vocab.")
    parser.add_argument("--triples_vocab_path", type=str, required=True,
                        help="Path of the entity vocab.")
    parser.add_argument("--elmo_path", type=str, required=True,
                        help="Path of the elmo features.")
    parser.add_argument("--context_elmo_path", type=str, required=True,
                        help="Path of the context elmo features.")



    # Model options.
    parser.add_argument("--language_type", type=str, choices=["en", "zh"], required=True,
                        help="Num of the classes.")
    parser.add_argument("--num_classes", type=int, default=3,
                        help="Num of the classes.")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch size.")
    parser.add_argument("--require_improvement", type=int, default=5,
                        help="Require improvement.")
    parser.add_argument("--epochs_num", type=int, default=200,
                        help="Number of epochs.")
    parser.add_argument("--w2v_embedding_dim", type=int, required=True,
                        help="w2v embedding dim.")
    parser.add_argument("--triples_embedding_dim", type=int, default=250,
                        help="triples embedding dim.")
    parser.add_argument("--elmo_embedding_dim", type=int, default=1024,
                        help="elmo embedding dim.")
    parser.add_argument("--input_dim", type=int, default=200,
                        help="input embedding dim.")
    parser.add_argument("--seq_length", type=int, default=128,
                        help="Sequence length.")
    parser.add_argument("--context_length", type=int, default=256,
                        help="Context length.")
    parser.add_argument("--hidden_size", type=int, default=200,
                        help="hidden size.")
    parser.add_argument("--layers_num", type=int, default=2,
                        help="Number of layers.")
    parser.add_argument("--triples_number", type=int, default=7,
                        help="The number of triples corresponding to each word in the text.")
    parser.add_argument("--processes_num", type=int, default=2,
                        help="Number of processes when loading trainsets.")
    parser.add_argument("--attention_query_size", type=int, default=200,
                        help="Size of attention query matrix.")
    parser.add_argument("--attention_layer", choices=["att", "m_a", "m_pre_orl_a", "m_pre_orl_pun_a", "m_pol_untrain_a",
                                                      "mpa", "mpoa","none"],
                        required=True,help="attention type.")
    parser.add_argument("--pretrain_model_type",
                        choices=["w2v", "elmo", "none"],
                        required=True,
                        help="pretrain model type.")
    parser.add_argument("--m_a_type",
                        choices=["rand", "FloatTensor"],
                        required=True,
                        help="m_a type.")
    parser.add_argument("--concat_mode",
                        choices=["concat", "graph_attention","none"],
                        required=True,
                        help="The combination of triples and text.")

    # Optimizer options.
    parser.add_argument("--learning_rate", type=float, default=0.01,
                        help="Learning rate.")
    parser.add_argument("--momentum", type=float, default= 0.95,
                        help="momentum.")

    # Training options.
    parser.add_argument("--dropout", type=float, default=0.4,
                        help="Dropout.")
    parser.add_argument("--is_bidir", type=int, default=True,
                        help="bidir or only one.")
    parser.add_argument("--report_steps", type=int, default=100,
                        help="Specific steps to print prompt.")
    parser.add_argument("--seed", type=int, default=7,
                        help="Random seed.")
    parser.add_argument("--run_type", type=str, required=True,
                        help="usage: python main_vua.py [train / test]")

    args = parser.parse_args()

    #set numpy、random、etc seeds
    set_seed(args.seed)

    #set vocab
    vocab = Vocab() # 实例化一个对象（类）
    vocab_entity = Vocab()
    vocab_relation = Vocab()
    vocab.load(args.vocab_path) # 看词表中的每个词汇是否有错误，将词变成字典和列表格式
    vocab_entity.load(os.path.join(args.triples_vocab_path,"vocab_entity.txt"))
    vocab_relation.load(os.path.join(args.triples_vocab_path, "vocab_relation.txt"))
    label_columns = read_cataloge(args.train_path) # 读取标签的种类

    #set embedding
    embeddings = get_embedding_matrix(args, vocab, normalization=False) # 读取w2v词表中词的向量或随机初始化
    embeddings_entity = get_triples_embedding_matrix(args, os.path.join(args.triples_vocab_path, "entity_transE.txt"), vocab_entity, normalization=False) # 读取三元组中实体的向量
    embeddings_relation = get_triples_embedding_matrix(args, os.path.join(args.triples_vocab_path, "relation_transE.txt"), vocab_relation, normalization=False)
    elmo_embedding = h5py.File(args.elmo_path, 'r') # 读取elmo方式初始化的隐式情感句词向量
    elmo_embedding_context = h5py.File(args.context_elmo_path, 'r') # 读取elmo方式初始化的上下文词向量
    query_matrix = get_query_matrix(args) # 获取查询向量

    model = RNNSequenceClassifier(args, embeddings, embeddings_entity, embeddings_relation, query_matrix)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    best_josn = {'F_macro': 0, 'P_macro': 0, 'R_macro': 0, 'Best_F_macro': 0, 'ACC': 0, 'F_negative': 0,
                 'F_positive': 0, 'Predict': [], 'Label': [], 'Weights': [], 'Last_up_epoch': 0, 'Total_batch_loss': 0,
                 'F_nuetral': 0, 'Time':0, 'Total_orthogonal_loss': 0, 'train_num': 0, 'test_num': 0, 'dev_num':0}

    # verification / testing
    def evaluate(args, is_test):
        model.eval()
        if is_test:
            print("Start testing.")
            dataset = read_dataset(args, args.test_path, label_columns, vocab, vocab_entity, vocab_relation)
            best_josn['test_num'] = len(dataset)
            writer_result = open(os.path.join(args.output_result_path, 'result.txt'), encoding='utf-8', mode='w')
            writer_summary_result = open(os.path.join(args.summary_result_path, 'summary_result.txt'), mode='a')
        else:
            print("Start verification.")
            dataset = read_dataset(args, args.dev_path, label_columns, vocab, vocab_entity, vocab_relation)
            best_josn['dev_num'] = len(dataset)
            random.shuffle(dataset)
        input_ids = torch.LongTensor([example[0] for example in dataset]).cuda()
        context_ids = torch.LongTensor([example[1] for example in dataset]).cuda()
        triples_ids = torch.LongTensor([example[2] for example in dataset]).cuda()
        id2_ids = torch.LongTensor([example[3] for example in dataset]).cuda()
        label_ids = torch.LongTensor([example[4] for example in  dataset]).cuda()
        length_ids = torch.LongTensor([example[5] for example in  dataset]).cuda()
        input = [example[5] for example in dataset]

        if is_test:
            batch_size = 1
        else:
            batch_size = args.batch_size

        for i, (input_ids_batch, context_ids_batch, triples_ids_batch, id2_ids_batch, label_ids_batch, length_ids_batch) in enumerate(
                batch_loader(batch_size, input_ids, context_ids, triples_ids, label_ids, length_ids, id2_ids)):
            model.zero_grad()
            input_ids_batch = input_ids_batch.cuda()
            context_ids_batch = context_ids_batch.cuda()
            label_ids_batch = label_ids_batch.cuda()
            triples_ids_batch = triples_ids_batch.cuda()
            id2_ids_batch = id2_ids_batch.cuda()
            length_ids_batch = length_ids_batch.cuda()

            if args.attention_layer == 'att':
                predicted, weight = model(input_ids_batch, context_ids_batch, triples_ids_batch, length_ids_batch, elmo_embedding,
                                             elmo_embedding_context, id2_ids_batch)
            else:
                predicted, weight, _ = model(input_ids_batch, context_ids_batch, triples_ids_batch, length_ids_batch, elmo_embedding,
                                             elmo_embedding_context, id2_ids_batch)

            best_josn['Weights'] += weight.squeeze(dim=1).cpu().detach().numpy().tolist()
            _, predicted_labels = torch.max(predicted.data, 1)
            best_josn['Predict'] += predicted_labels.cpu().numpy().tolist()
            best_josn['Label'] += label_ids_batch.data.cpu().numpy().tolist()

        if is_test:
            details_result = metrics.classification_report(best_josn['Label'], best_josn['Predict'], digits=4)
            best_josn['P_macro'], best_josn['R_macro'], best_josn['F_macro'], _ = metrics.precision_recall_fscore_support(best_josn['Label'], best_josn['Predict'], average="macro")
            best_josn['ACC'] = metrics.accuracy_score(best_josn['Label'], best_josn['Predict'])
            saveSenResult(input, best_josn['Label'], best_josn['Predict'], args, best_josn['Weights'])
            writer_result.writelines(details_result)
            print("Testing Acc: {:.4f}, F_macro: {:.4f}, P_macro: {:.4f}, R_macro: {:.4f}".format(best_josn['ACC'],
                                                                                                  best_josn['F_macro'],
                                                                                                  best_josn['P_macro'],
                                                                                                  best_josn['R_macro']))
            writer_result.writelines(
                "Testing Acc: {:.4f}, F_macro: {:.4f}, P_macro: {:.4f}, R_macro: {:.4f}".format(best_josn['ACC'],
                                                                                                best_josn['F_macro'],
                                                                                                best_josn['P_macro'],
                                                                                                best_josn['R_macro']))
            writer_summary_result.writelines('保存路径' + args.output_result_path + "\t" + str(args.learning_rate)+"\t" + str(args.momentum) +'\n')
            writer_summary_result.writelines(
                "Testing Acc: {:.4f}, F_macro: {:.4f}, P_macro: {:.4f}, R_macro: {:.4f}\n\n".format(best_josn['ACC'],
                                                                                                best_josn['F_macro'],
                                                                                                best_josn['P_macro'],
                                                                                                best_josn['R_macro']))
            writer_summary_result.writelines(details_result)
        else:
            best_josn['P_macro'], best_josn['R_macro'], best_josn['F_macro'], _ = metrics\
            .precision_recall_fscore_support(best_josn['Label'], best_josn['Predict'],average="macro")
            best_josn['ACC'] = metrics.accuracy_score(best_josn['Label'],best_josn['Predict'])

    # training
    def train():
        print("Start training.")
        mkdir(args.output_result_path)
        writer_process = open(os.path.join(args.output_result_path, 'process.txt'), mode='w')
        writer_process.writelines("Start training.")
        trainset = read_dataset(args, args.train_path, label_columns, vocab, vocab_entity, vocab_relation) # 读数据集
        random.shuffle(trainset)

        best_josn['train_num'] = len(trainset)

        input_ids = torch.LongTensor([example[0] for example in trainset]).cuda()
        context_ids = torch.LongTensor([example[1] for example in trainset]).cuda()
        triples_ids = torch.LongTensor([example[2] for example in trainset]).cuda()
        id2_ids = torch.LongTensor([example[3] for example in trainset]).cuda()
        label_ids = torch.LongTensor([example[4] for example in trainset]).cuda()
        length_ids = torch.LongTensor([example[5] for example in trainset]).cuda()

        print("Batch size: ", args.batch_size)
        print("The number of training instances:", best_josn['train_num'])

        start_time = time.time()
        best_josn['Time'] = get_time_dif(start_time)
        print("Time usage:", best_josn['Time'])

        param_optimizer = list(model.named_parameters())
        nll_criterion = nn.NLLLoss() # 定义损失函数
        if args.attention_layer == 'm_pol_untrain_a': # 定义优化器
            optimizer_grouped_parameters = [{'params': [p for n, p in param_optimizer if ('query_embedding.weight' not in n)], 'weight_decay_rate': 0.01}]
        else:
            optimizer_grouped_parameters = [{'params': [p for n, p in param_optimizer], 'weight_decay_rate': 0.01}]
        optimizer = optim.SGD(optimizer_grouped_parameters, lr = args.learning_rate, momentum = args.momentum)

        for epoch in range(1, args.epochs_num + 1):
            model.train()
            for i, (input_ids_batch, context_ids_batch, triples_ids_batch, id2_ids_batch, label_ids_batch, length_ids_batch) in enumerate(
                    batch_loader(args.batch_size, input_ids, context_ids, triples_ids, label_ids, length_ids, id2_ids)):
                model.zero_grad()
                input_ids_batch = input_ids_batch.cuda()
                context_ids_batch = context_ids_batch.cuda()
                label_ids_batch = label_ids_batch.cuda()
                triples_ids_batch = triples_ids_batch.cuda()
                id2_ids_batch = id2_ids_batch.cuda()
                length_ids_batch = length_ids_batch.cuda()

                #print(input_ids_batch.size())
                if args.attention_layer == 'att':
                    predicted_ids_batch, _ = model(input_ids_batch, context_ids_batch, triples_ids_batch, length_ids_batch, elmo_embedding, elmo_embedding_context, id2_ids_batch)
                else:
                    predicted_ids_batch, _, orthogonal_loss = model(input_ids_batch, context_ids_batch, triples_ids_batch, length_ids_batch, elmo_embedding, elmo_embedding_context, id2_ids_batch)

                    best_josn['Total_orthogonal_loss'] += orthogonal_loss

                batch_loss = nll_criterion(predicted_ids_batch, label_ids_batch)
                best_josn['Total_batch_loss'] += batch_loss
                if args.attention_layer != 'm_pre_orl_pun_a' and args.attention_layer != 'mpoa':
                    optimizer.zero_grad()
                    batch_loss.backward()
                    optimizer.step()
                else:
                    optimizer.zero_grad()
                    (0.1*orthogonal_loss).backward(retain_graph=True)
                    (0.9*batch_loss).backward(retain_graph=True)
                    optimizer.step()
                best_josn['Time'] = get_time_dif(start_time)
                if (i + 1) % args.report_steps == 0:
                    if args.attention_layer == 'att':
                        print("Epoch id: {}, Training steps: {}, Avg batch loss: {:.4f}, Time: {}".format
                              (epoch, i + 1, best_josn['Total_batch_loss'] / args.report_steps, best_josn['Time']))
                        writer_process.writelines("Epoch id: {}, Training steps: {}, Avg batch loss: {:.4f},"
                        " Time: {}".format(epoch, i + 1,best_josn['Total_batch_loss'] / args.report_steps, best_josn['Time']))
                    else:
                        print("Epoch id: {}, Training steps: {}, Avg batch loss: {:.4f}, Avg orthogonal loss:"
                        " {:.4f}, Time: {}".format(epoch, i + 1,best_josn['Total_batch_loss'] / args.report_steps,
                                    best_josn['Total_orthogonal_loss'] / args.report_steps, best_josn['Time']))
                        writer_process.writelines("Epoch id: {}, Training steps: {}, Avg batch loss: {:.4f},"
                        " Avg orthogonal loss: {:.4f}, Time: {}".format(epoch, i + 1,
                                    best_josn['Total_batch_loss'] / args.report_steps,
                                    best_josn['Total_orthogonal_loss'] / args.report_steps, best_josn['Time']))
                    best_josn['Total_batch_loss'] = 0
                    best_josn['Total_orthogonal_loss'] = 0
            evaluate(args, False)
            best_josn['Time'] = get_time_dif(start_time)
            if best_josn['F_macro'] > best_josn['Best_F_macro'] + 0.001:
                best_josn['Best_F_macro'] = best_josn['F_macro']
                best_josn['Last_up_epoch'] = epoch
                torch.save(model, os.path.join(args.output_result_path, 'result.pkl'))
                print("Deving Acc: {:.4f}, F_macro: {:.4f}, Time: {} *".format(best_josn['ACC'], best_josn['F_macro'], best_josn['Time']))
                writer_process.writelines("Deving Acc: {:.4f}, F_macro: {:.4f}, Time: {} *"
                                .format(best_josn['ACC'], best_josn['F_macro'], best_josn['Time']))
            elif epoch - best_josn['Last_up_epoch'] == args.require_improvement:
                print("No optimization for a long time, auto-stopping...")
                writer_process.writelines("No optimization for a long time, auto-stopping...")
                break
            else:
                print("Deving Acc: {:.4f}, F_macro: {:.4f}, Time: {} ".format(best_josn['ACC'], best_josn['F_macro'], best_josn['Time']))
                writer_process.writelines("Deving Acc: {:.4f}, F_macro: {:.4f}, Time: {} ".format(best_josn['ACC'], best_josn['F_macro'], best_josn['Time']))

    if args.run_type == 'train':
        train()
    else:
        model = torch.load(os.path.join(args.output_result_path, 'result.pkl'))
        evaluate(args, True)

if __name__ == "__main__":
    main()
