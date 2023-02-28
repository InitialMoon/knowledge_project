# 训练集数据的位置
TRAIN_FILE = '../../../data/train.txt'
CORPUS = '../../../data/corpus.txt'
PREFIX_NUM = 19 # 识别数据对应关系的前19个字符

train_data = open(TRAIN_FILE, 'r', encoding='utf-8')
corpus_data = open(CORPUS, 'r', encoding='utf-8')
# trainline is str type

i = 0
train_line = train_data.readline()
corpus_line = corpus_data.readline()

while True:
    if train_line == "":
        break
    else:
        if train_line[0:PREFIX_NUM] == corpus_line[0:PREFIX_NUM]:
            i = i + 1
            print(train_line)
            print(corpus_line)
            train_line = train_data.readline()
            corpus_line = corpus_data.readline()
        else:
            corpus_line = corpus_data.readline()

print(i)
train_data.close()
corpus_data.close()
