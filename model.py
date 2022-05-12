import torch.nn as nn
import torch.nn.functional as F
from util import sort_batch_by_length
from util import last_dim_softmax
from torch.nn.utils.rnn import pack_padded_sequence
from torch.nn.utils.rnn import pad_packed_sequence

import torch

class RNNSequenceClassifier(nn.Module):
    def __init__(self, args, embedding, embeddings_entity, embeddings_relation, query_embedding):
        # Always call the superclass (nn.Module) constructor first
        super(RNNSequenceClassifier, self).__init__()
        self.args = args
        if args.is_bidir == True:
            is_bidir_number = 2
        else:
            is_bidir_number = 1

        self.embedding = embedding
        self.embeddings_entity = embeddings_entity
        self.embeddings_relation = embeddings_relation
        self.seq_length = args.seq_length
        self.triples_number = args.triples_number
        self.batch_size = args.batch_size
        self.context_length = args.context_length

        if self.args.pretrain_model_type=="elmo":
            self.input_dim = args.elmo_embedding_dim
        else:
            self.input_dim = args.input_dim

        if self.args.concat_mode=="graph_attention":
            self.triples_embedding_dim=args.triples_embedding_dim * 2
            self.rnn = nn.LSTM(input_size=self.input_dim+self.triples_embedding_dim, hidden_size=args.hidden_size,
                               num_layers=args.layers_num, dropout=args.dropout, batch_first=False, bidirectional=args.is_bidir)
        elif self.args.concat_mode=="concat":
            self.triples_embedding_dim = args.triples_embedding_dim
            self.rnn = nn.LSTM(input_size=self.input_dim + self.triples_embedding_dim, hidden_size=args.hidden_size, num_layers=args.layers_num,
                               dropout=args.dropout, batch_first=True,bidirectional=args.is_bidir)
        else:
            self.rnn = nn.LSTM(input_size=self.input_dim, hidden_size=args.hidden_size,
                               num_layers=args.layers_num,dropout=args.dropout, batch_first=True, bidirectional=args.is_bidir)


        self.entity_transformed = nn.Linear(args.triples_embedding_dim * 2, args.triples_embedding_dim, False)
        self.relation_transformed = nn.Linear(args.triples_embedding_dim, args.triples_embedding_dim, False)
        self.change_average = nn.Linear(args.triples_embedding_dim * 2, args.input_dim+args.triples_embedding_dim * 2)
        if args.attention_layer == 'att':
            self.attention_weights = nn.Linear(args.hidden_size * is_bidir_number, 1)
            self.output_projection = nn.Linear(args.hidden_size * is_bidir_number, args.num_classes)
        else:
            self.query_embedding = query_embedding
            self.proquery_weights_mp = nn.Linear(args.hidden_size * is_bidir_number, args.attention_query_size)
            self.multi_output_projection = nn.Linear(args.hidden_size * is_bidir_number* args.num_classes*2, args.num_classes)
        self.dropout_on_input_to_LSTM = nn.Dropout(args.dropout, inplace=False)
        self.dropout_on_input_to_linear_layer = nn.Dropout(args.dropout, inplace=False)
        self.aaa = nn.Linear(args.hidden_size * is_bidir_number, 1)
        self.bbb = nn.Linear(args.hidden_size * is_bidir_number, args.hidden_size * is_bidir_number* args.num_classes)

    def forward(self, inputs, context, triples, lengths, elmo_embedding, elmo_embedding_context, id2_ids_batch):
        #print("{{{{{{{{{{{{{{{{{{{{{{{{{")
        if self.args.pretrain_model_type == 'elmo':
            elmo_inputs = torch.Tensor().cuda()
            elmo_inputs_context = torch.Tensor().cuda()
            for i in range(len(inputs)):
                elmo_input = torch.from_numpy(elmo_embedding[' '.join(map(str, inputs[i].cpu().numpy()))].value).type(torch.cuda.FloatTensor) # 将隐式情感句输入的id转为elmo向量表示
                elmo_input_context = torch.from_numpy(elmo_embedding_context[' '.join(map(str, context[i].cpu().numpy()))].value).type(torch.cuda.FloatTensor) # 将上下文输入的id转为elmo向量表示

                try:
                    elmo_inputs = torch.cat((elmo_inputs, elmo_input.unsqueeze(dim=0)))
                except:
                    elmo_inputs = torch.cat((elmo_inputs, elmo_input.unsqueeze(dim=0)[:,:128,:]), dim=0)

                elmo_inputs_context = torch.cat((elmo_inputs_context, elmo_input_context.unsqueeze(dim=0)[:,:256,:]), dim=0)
                #elmo_inputs_context = torch.cat((elmo_inputs_context, elmo_input_context.unsqueeze(dim=0)[:, :128, :]),dim=0)
            #print(elmo_inputs.size())
            inputs = elmo_inputs
           # print(inputs.size())
            context = elmo_inputs_context
        else:
            inputs = self.embedding(inputs)
            context = self.embedding(context)

        #print(inputs.size())
        # Introducing external knowledge in different ways.

        if self.args.concat_mode=="graph_attention":
            t = torch.zeros(inputs.size(0), self.seq_length, self.input_dim + self.triples_embedding_dim).cuda()
            # print("{{{{{{{{{{{{{{{{{{{{{{{{{")
            for i in range(len(inputs)):
               # print(i)
                b = torch.full([self.seq_length, self.triples_number], -1, dtype=torch.long).cuda()
                bb = torch.zeros(self.seq_length, self.triples_embedding_dim).cuda()
                if (torch.equal(id2_ids_batch[i], b)):
                    t[i] = torch.cat((inputs[i], bb), dim=-1)
                else:
                    for k in range(len(id2_ids_batch[i])):
                        c = torch.full([self.triples_number], -1, dtype=torch.long).cuda()
                        cc = torch.zeros(self.triples_embedding_dim).cuda()
                        if (torch.equal(id2_ids_batch[i][k], c)):
                            t[i][k] = torch.cat((inputs[i][k], cc), dim=-1)
                        else:
                            list1 = torch.Tensor().cuda()
                            list2 = torch.Tensor().cuda()
                            head_id, tail_id, relation_id = torch.chunk(triples[i][k], 3, dim=1)
                            t2 = self.embeddings_entity[head_id].cuda()
                            t21 = self.embeddings_entity[tail_id].cuda()
                            t22 = self.embeddings_relation[relation_id].cuda()
                            head_tail = torch.cat((t2, t21), dim=2)
                            list1 = torch.cat((list1, head_tail), dim=0)
                            list2 = torch.cat((list2, t22), dim=0)
                            head_tail_transformed = self.entity_transformed(list1)
                            head_tail_transformed_final = F.tanh(head_tail_transformed)
                            relation_transformed1 = list2
                            e_weight = (head_tail_transformed_final * relation_transformed1).sum(dim=2)
                            alpha_weight = F.softmax(e_weight, dim=0)
                            graph_embed = (alpha_weight.unsqueeze(1) * head_tail).sum(dim=0)
                            aa = torch.cat((inputs[i][k], graph_embed.squeeze(0)))
                            t[i][k] = aa
            xx = torch.zeros(inputs.size(0), self.context_length, self.triples_embedding_dim).cuda()
            context_t = torch.cat((context, xx), dim=-1)
        elif self.args.concat_mode=="concat":
            t = torch.zeros(inputs.size(0), self.seq_length, self.input_dim + self.triples_embedding_dim).cuda()
            # print("}}}}}}}}}}}}}}}}}}}}}}}}}}}")
            for i in range(len(inputs)):
                dict = {}
                b = torch.full([self.seq_length, self.triples_number], -1, dtype=torch.long).cuda()
                bb = torch.zeros(self.seq_length, self.triples_embedding_dim).cuda()
                if (torch.equal(id2_ids_batch[i], b)):
                    t[i] = torch.cat((inputs[i], bb), dim=-1)
                else:
                    for k in range(len(id2_ids_batch[i])):
                        a = 0
                        input = torch.Tensor().cuda()
                        c = torch.full([self.triples_number], -1, dtype=torch.long).cuda()
                        cc = torch.zeros(self.triples_embedding_dim).cuda()
                        if (torch.equal(id2_ids_batch[i][k], c)):
                            t[i][k] = torch.cat((inputs[i][k], cc), dim=-1)
                        else:
                            for j in range(len(id2_ids_batch[i][k])):
                                if id2_ids_batch[i][k][j].cpu().numpy() == 1:
                                    inputs_triples = torch.cat(
                                        (inputs[i][k], self.embeddings_entity(triples[i][k][j][1])))
                                elif id2_ids_batch[i][k][j].cpu().numpy() == 2:
                                    inputs_triples = torch.cat(
                                        (inputs[i][k], self.embeddings_entity(triples[i][k][j][0])))
                                else:
                                    continue

                                if a == 0:
                                    a = a + 1
                                    input = torch.cat((inputs_triples, input))
                                else:
                                    a = a + 1
                                    input = input + inputs_triples

                        if a != 0:
                            input = input / a
                            dict[k] = input

                    for k in dict:
                        t[i][k] = dict[k]
        else:
            t = torch.zeros(inputs.size(0), self.seq_length, self.input_dim).cuda()
            # print(">>>>>>>>>>>>>>>>>>.")
            t=inputs
            context_t = context


        # 1. input
        #print(inputs.size(),context.size())
        #print(t.size(0),t.size(1),t.size(-1))
        #cc = torch.cat((t, context_t),dim=1)
        embedded_input = self.dropout_on_input_to_LSTM(t) # 随机失活
        embedded_context = self.dropout_on_input_to_LSTM(context_t)
        # LSTM过程
        (sorted_input, sorted_lengths, input_unsort_indices, _) = sort_batch_by_length(embedded_input, lengths)
        (sorted_input_context, sorted_lengths_context, input_unsort_indices_context, __context) = sort_batch_by_length(embedded_context, lengths)
        packed_input = pack_padded_sequence(sorted_input, sorted_lengths.data.tolist(), batch_first=True)
        packed_input_context = pack_padded_sequence(sorted_input_context, sorted_lengths_context.data.tolist(), batch_first=True)
        packed_sorted_output, _ = self.rnn(packed_input)
        packed_sorted_output_context, __context = self.rnn(packed_input_context)
        sorted_output, _ = pad_packed_sequence(packed_sorted_output, batch_first=True)
        sorted_output_context, __context = pad_packed_sequence(packed_sorted_output_context, batch_first=True)
        #output_ = sorted_output[input_unsort_indices]
        output = sorted_output[input_unsort_indices]
        output_context = sorted_output_context[input_unsort_indices_context]


        # 2. use attention
        if self.args.attention_layer == 'att':
            attention_logits = self.attention_weights(output).squeeze(-1)
            mask_attention_logits = (attention_logits != 0).type(
                torch.cuda.FloatTensor if inputs.is_cuda else torch.FloatTensor)
            softmax_attention_logits = last_dim_softmax(attention_logits, mask_attention_logits)
            softmax_attention_logits0 = softmax_attention_logits.unsqueeze(dim=1)
            input_encoding = torch.bmm(softmax_attention_logits0, output)
            input_encoding0 = input_encoding.squeeze(dim=1)
        else:
            input_encoding = torch.Tensor().cuda()
            querys = self.query_embedding(torch.arange(0,self.args.num_classes,1).cuda())
            attention_weights = torch.Tensor(self.args.num_classes, len(output), len(output[0])).cuda()
            for i in range(self.args.num_classes):
                attention_logits = self.proquery_weights_mp(output)
                attention_logits = torch.bmm(attention_logits, querys[i].unsqueeze(dim=1).repeat(len(output),1,1)).squeeze(dim=-1)
                mask_attention_logits = (attention_logits != 0).type(
                    torch.cuda.FloatTensor if inputs.is_cuda else torch.FloatTensor)
                softmax_attention_logits = last_dim_softmax(attention_logits, mask_attention_logits)
                input_encoding_part = torch.bmm(softmax_attention_logits.unsqueeze(dim=1), output)
                input_encoding = torch.cat((input_encoding,input_encoding_part.squeeze(dim=1)), dim=-1)
                attention_weights[i] = softmax_attention_logits

        # 3. run linear layer
        if self.args.attention_layer == 'att':
            input_encodings = self.dropout_on_input_to_linear_layer(input_encoding0)
            unattized_output = self.output_projection(input_encodings)
            output_distribution = F.log_softmax(unattized_output, dim=-1)
            return output_distribution, softmax_attention_logits.squeeze(dim=1)
        else:
            # 和上下文拼接
            aaaa = torch.tanh(self.aaa(output))# 将output视为查询向量，得权重
            aaaaa = torch.bmm(aaaa.squeeze(-1).unsqueeze(1), output_context) # 上下文表示
            bbbb = self.bbb(aaaaa.squeeze(1))
            input_encoding0 = torch.cat((input_encoding, bbbb), dim=1) # 隐式情感句表示与上下文拼接
            input_encodings = self.dropout_on_input_to_linear_layer(input_encoding0)
            unattized_output = self.multi_output_projection(input_encodings)
            output_distribution = F.log_softmax(unattized_output, dim=-1)
            cos = torch.nn.CosineSimilarity(dim=0, eps=1e-16)
            attention_loss = abs(cos(querys[0], querys[1])) + abs(cos(querys[1], querys[2])) \
                                                            + abs(cos(querys[0], querys[2])) # 正交
            return output_distribution, attention_weights, attention_loss
