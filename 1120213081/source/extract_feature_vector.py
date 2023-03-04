# 注意，再使用本函数的时候，入宫给的输出目标文件名式之前存在的文件名，则请删除掉这个文件，
# 因为使用的是a方式写入的，所以如果原先还保留着错误的数据，则会出现重复现象，这回造成数据的问题
# 使用方式，直接更改全局变量即可，只要更改 SOURCE_FILE 的路径作为输入的目标（train， validation， test），和输出目标文件的新名字即可
# 最后会有一个换行，注意后面使用的过程种如果遇到了说这个向量是0维的，那么可能似乎这里出问题了, 不过按理来说，再后面的程序种我已经加入了跳过空格的措施。

PREFIX_NUM = 19  # 识别数据对应关系的前19个字符
featureWordFile = "./frequentWords.txt"
TRAIN_FILE = '../../../data/train.txt'
VAL_FILE = '../../../data/validation.txt'
TEST_FILE = '../../../data/test.txt'
CORPUS = '../../../data/corpus.txt'
SOURCE_FILE = TRAIN_FILE
TARGET_FILE = './train_x.txt'


def extract_feature_of(word_dict, source, target_file):
    feature_words = open(word_dict, 'r')
    source_data = open(source, 'r', encoding='utf-8')
    corpus_data = open(CORPUS, 'r', encoding='utf-8')

    # read in the words into a list
    words = {}
    for w in feature_words:
        w = w.strip()
        words[w] = words.get(w, 0) + 1

    source_line = source_data.readline()
    corpus_line = corpus_data.readline()

    # 存放所有语料的特征向量
    x = []
    i = 0
    f = open(target_file, "a")

    while True:
        if source_line == "":
            break
        else:
            if source_line[0:PREFIX_NUM] == corpus_line[0:PREFIX_NUM]:
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
                f.write('\n')
                print(j)
                if j != 5001:
                    print("error in " + str(i) + " this line, the vector length is not 5001 demsions")
                    print(source_line)
                    print(corpus_line)
                    # 如果有少量的不满5001维的向量，可以直接忽视
                    source_line = source_data.readline()
                    corpus_line = corpus_data.readline()
                    continue
                i = i + 1
                source_line = source_data.readline()
                corpus_line = corpus_data.readline()
            else:
                corpus_line = corpus_data.readline()

    print("This extract action we get " + str(i) + "vectors")
    f.close()
    source_data.close()
    corpus_data.close()


extract_feature_of(featureWordFile, SOURCE_FILE, TARGET_FILE)
