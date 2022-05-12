1.使用命令运行文件：
python main_vua.py --w2v_embedding_dim 200 --elmo_embedding_dim 1024 --attention_query_size 200 --run_type train --pretrained_w2v_model_path F:\Wangmin\static_data_context\smp\smp_embedding.txt --elmo_path F:\Wangmin\static_data_context\smp\elmo.hdf5 --context_elmo_path F:\Wangmin\static_data_context\smp\elmo_context_256.hdf5 --train_path F:\Wangmin\static_data_context\smp\clean_train_context_triples_小于七个三元组.txt --dev_path F:\Wangmin\static_data_context\smp\clean_dev_context_triples_小于七个三元组.txt --test_path F:\Wangmin\static_data_context\smp\clean_test_context_triples_小于七个三元组.txt --vocab_path F:\Wangmin\static_data_context\smp\vocab.txt --summary_result_path results --input_dim 200 --output_result_path result/smp/elmo/mpoa/rand --attention_layer mpoa --pretrain_model_type elmo --query_matrix_path F:\Wangmin\static_data_context\emo_vector_zh.json --language_type zh --m_a_type rand --triples_vocab_path F:\Wangmin\static_data\smp\bert_new\300 --epochs_num 200 --triples_embedding_dim 300 --batch_size 32 --dropout 0.4 --learning_rate 0.01 --momentum 0.95 --concat_mode graph_attention
python main_vua.py --w2v_embedding_dim 200 --elmo_embedding_dim 1024 --attention_query_size 200 --run_type train --pretrained_w2v_model_path F:\Wangmin\static_data_context\smp\smp_embedding.txt --elmo_path F:\Wangmin\static_data_context\smp\elmo.hdf5 --context_elmo_path F:\Wangmin\static_data_context\smp\elmo_context_256.hdf5 --train_path F:\Wangmin\static_data_context\smp\1.txt --dev_path F:\Wangmin\static_data_context\smp\1.txt --test_path F:\Wangmin\static_data_context\smp\1.txt --vocab_path F:\Wangmin\static_data_context\smp\vocab.txt --summary_result_path results --input_dim 200 --output_result_path result/smp/elmo/mpoa/rand --attention_layer mpoa --pretrain_model_type elmo --query_matrix_path F:\Wangmin\static_data_context\emo_vector_zh.json --language_type zh --m_a_type rand --triples_vocab_path F:\Wangmin\static_data\smp\bert_new\300 --epochs_num 2 --triples_embedding_dim 300 --batch_size 2 --dropout 0.4 --learning_rate 0.01 --momentum 0.95 --concat_mode graph_attention


2.参数说明：
--vocab_path 词表
--run_type 运行类型,可选[train/test]
--output_result_path 模型保存路径
--summary_result_path 测试集结果保存路径
--attention_layer 注意力机制类型,可选[att/m_a/m_pre_orl_a/m_pre_orl_pun_a/m_pol_untrain_a/mpa/mpoa/none]
--pretrain_model_type 文本词向量预训练方式,可选[w2v/elmo/random]
--query_matrix_path 查询向量路径
--language_type 数据集类型,可选[中/英]
--concat_mode 文本与知识结合方式,可选[concat/graph_attention]
