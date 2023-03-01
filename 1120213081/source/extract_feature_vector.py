featureWordFile = "./frequentWords.txt"
TRAIN_FILE = '../../../data/train.txt'
CORPUS = '../../../data/corpus.txt'
PREFIX_NUM = 19  # 识别数据对应关系的前19个字符

featureWords = open(featureWordFile, 'r')
train_data = open(TRAIN_FILE, 'r', encoding='utf-8')
corpus_data = open(CORPUS, 'r', encoding='utf-8')

# read in the words into a list
words = {}
j = 0
for w in featureWords:
    w = w.strip()
    words[w] = words.get(w, 0) + 1
# print(len(words))
train_line = train_data.readline()
corpus_line = corpus_data.readline()

# 存放所有语料的特征向量
x = []

i = 0
f = open("evFrequentWords.txt", "a")
while True:
    if train_line == "":
        break
    else:
        if train_line[0:PREFIX_NUM] == corpus_line[0:PREFIX_NUM]:
            oriWords = corpus_line.split(' ')  # 将读入的原始语料使用空格进行分割
            # 一个句子中所有的词中有没有，没有没在这5000维向量中的
            x.append([])
            j = 0
            # 这个是用原句子在对词典做搜索循环,对5001维做判断
            flag = False
            for ow in oriWords:
                if "/m" not in ow:
                    splitWord = ""
                    for c in ow:
                        splitWord = splitWord + c
                        if c == '/' or c == '{':
                            sow = splitWord[:-1]
                            print(sow)
                            if sow not in words:
                                flag = True
                                break
                    if flag:
                        f.write("1 ")
                        # x[i].append(1) # 这里如果有字典中没有出现的词，我们就将第一维记为1，否则记为0
                        j = j + 1
                        break
            if not flag:
                f.write("0 ")
                # x[i].append(0)
                # j = j + 1
            # 这个是用词典在对原句子中的词做搜索循环,对5000维做判断
            for w in words:
                if w in corpus_line:
                    x[i].append(1)
                    f.write("1 ")
                    j = j + 1
                else:
                    x[i].append(0)
                    f.write("0 ")
                    j = j + 1
            i = i + 1
            print(j)
            train_line = train_data.readline()
            corpus_line = corpus_data.readline()
        else:
            corpus_line = corpus_data.readline()
        f.write('\n')

f.close()
train_data.close()
corpus_data.close()
