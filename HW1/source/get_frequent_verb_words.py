# 训练集数据的位置
TRAIN_FILE = '../../../data/train.txt'
CORPUS = '../../../data/corpus.txt'
PREFIX_NUM = 19  # 识别数据对应关系的前19个字符
VERB_NUM = 4000
ADJ_NUM = 1000

train_data = open(TRAIN_FILE, 'r', encoding='utf-8')
corpus_data = open(CORPUS, 'r', encoding='utf-8')
# trainline is str type

i = 0
train_line = train_data.readline()
corpus_line = corpus_data.readline()
vtdict = {} # 动词字典
adict = {} # 形容词字典

# 找出字典中前n个value大的值函数，来源https://blog.csdn.net/Sun_Raiser/article/details/124076514
def sortedDictValues(adict):
    items = list(adict.items())
    items.sort(key=lambda x: x[1], reverse=True)
    return [(key, value) for key, value in items]


while True:
    if train_line == "":
        break
    else:
        if train_line[0:PREFIX_NUM] == corpus_line[0:PREFIX_NUM]:
            i = i + 1
            words = corpus_line.split(' ')  # 将读入的原始语料使用空格进行分割
            # 找动词,和形容词
            for w in words:
                if "/a" in w or "/vt" in w or "/vn" in w or "/vi" in w:
                    splitWord = ""
                    for c in w:
                        splitWord = splitWord + c
                        if c == '/' or c == '{':
                            w = splitWord[:-1]
                            break
                    vtdict[w] = vtdict.get(w, 0) + 1

            train_line = train_data.readline()
            corpus_line = corpus_data.readline()
        else:
            corpus_line = corpus_data.readline()

train_data.close()
corpus_data.close()

resDict = {}

resDict.update(vtdict)
resDict.update(adict)

sortedDict = sortedDictValues(resDict)
# 分别选择频率高的词汇到词库里
f = open("frequentWords.txt", "w")

for i in range(0, 5000):
    f.write(sortedDict[i][0])
    f.write('\n')

f.close()