import re
import os
import jieba

r = '''[/_$&%^*<>+""@|~{}#]+|[——！\\\=“”‘’￥《》【】'"]'''
root = os.getcwd()
vocabs = []
text2label = {}
def process_file(file):
    rfile = os.path.join(root, file)
    wfile = 'clean_'+file
    with open(rfile, 'r', encoding='utf-8') as fr:
        for line in fr:
            docid, text, label = line.replace('\u3000','').split('\t')
            label = label.strip()
            text = re.sub(r, ' ', text.strip())
            text = re.sub('[ ]+', '', text).strip()
            if len(text) == 0:
                print(line)
                continue
            #检查是否有重复，如果有则删除数据
            if text not in text2label:
                text2label[text] = (label, wfile)
            elif text2label[text] != label:
                text2label.pop(text)
            vocabs.extend(list(jieba._lcut(text)))
    return wfile

wfile_list = []
for file in os.listdir(root):
    if re.match(r'^(?!clean|vocab|Tence).*.txt', file):
        wfile = process_file(file)
        wfile_list.append(open(wfile, 'w', encoding='utf-8'))

def write_flie(f):
    id = 0
    f.writelines('id\ttext\tlabel\n')
    for text, (label, wfile) in text2label.items():
        if f.name == wfile:
            f.write('text_'+str(id)+'\t'+text+'\t'+label+'\n')
            id += 1
for f in wfile_list:
    write_flie(f)

for f in wfile_list:
    f.close()
vocabs = list(set(vocabs))
vocab_path = os.path.join(root, 'vocab.txt')
with open(vocab_path, 'w', encoding='utf-8') as fw:
    fw.writelines('<PAD>\n')
    for w in vocabs:
        fw.writelines(w+'\n')
    fw.writelines('<UNK>\n')


