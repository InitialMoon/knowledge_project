import csv
import random


def find_entity(sentence):
    wordlist = sentence.split(" ")
    sentence_id = int(wordlist[0])
    del wordlist[0]
    e1_start_pos = 0
    e1_end_pos = 0
    e2_start_pos = 0
    e2_end_pos = 0
    i = 0
    for word in wordlist:
        if word[0:4] == "<e1>":
            e1_start_pos = i
        if word[-5:] == "</e1>":
            e1_end_pos = i
        if word[0:4] == "<e2>":
            e2_start_pos = i
        if word[-5:] == "</e2>":
            e2_end_pos = i
        i = i + 1

    return sentence_id, e1_start_pos, e1_end_pos, e2_start_pos, e2_end_pos


def wash_sentence(s):
    s = s.replace("<e1>", "")
    s = s.replace("</e1>", "")
    s = s.replace("<e2>", "")
    s = s.replace("</e2>", "")
    return s


def generate_random_vector(dim):
    vec = []
    for _ in range(dim):
        vec.append((random.random() - 0.5) * 2)
    return vec


if __name__ == "__main__":
    raw_data = []
    wash_path_train = "../raw_data/SemEval2010_task8_training/train_file.txt"
    wash_path_test = "../raw_data/SemEval2010_task8_testing_keys/test_file_full.txt"
    train_num = 8000

    with open(wash_path_train, "r") as fin:
        for line in fin:
            if (line != "\n") and (line[0:7] != "Comment"):
                # 删除标点
                line = line.replace('"', "")
                line = line.replace(".", "")
                line = line.replace("\t", " ")
                line = line.replace("\n", "")
                raw_data.append(line)

    with open(wash_path_test, "r") as fin:
        for line in fin:
            if (line != "\n") and (line[0:7] != "Comment"):
                # 删除标点
                line = line.replace('"', "")
                line = line.replace(".", "")
                line = line.replace("\t", " ")
                line = line.replace("\n", "")
                raw_data.append(line)

    relation_dic = {}
    with open("./relation", "r") as fin:
        for line in fin:
            line = line.replace("\n", "")
            line = line.split(" ")
            relation_dic[line[1]] = line[0]

    word_dic = {}

    sentences = []
    relations = []
    i = 0
    while i < len(raw_data):
        raw_data[i] = raw_data[i].replace(",", " ")
        sentences.append(raw_data[i].lower())
        i += 1
        relations.append(raw_data[i].replace("\n", ""))
        i += 1

    j = 0
    for s in sentences:
        sentences[j] = wash_sentence(s)
        j += 1

    word_num = 0
    max_sentence_len = 0
    for s in sentences:
        flag = 0
        s = s.split(" ")
        if len(s) > max_sentence_len:
            max_sentence_len = len(s)
        for w in s:
            if w not in word_dic:
                if flag != 0:
                    word_dic[w] = word_num
                    word_num = word_num + 1
            flag += 1

    sentences.clear()
    sentences = []
    i = 0
    while i < len(raw_data):
        sentences.append(raw_data[i].lower())
        i += 1
        relations.append(raw_data[i].replace("\n", ""))
        i += 1

    j = 0
    outputs = []
    for _ in range(len(sentences)):
        output = []
        sentence_id, e1_start_pos, e1_end_pos, e2_start_pos, e2_end_pos = find_entity(
            str(sentences[j])
        )
        rela_id = relation_dic[relations[j]]
        sentences[j] = wash_sentence(sentences[j])
        s = sentences[j].split(" ")
        s = s[1:]
        output.append(j + 1)
        output.append(int(rela_id))
        output.append(len(s))
        output.append(e1_start_pos)
        output.append(e1_end_pos)
        output.append(e2_start_pos)
        output.append(e2_end_pos)

        for w in s:
            output.append(word_dic[w])
        if len(s) < max_sentence_len:
            for k in range(max_sentence_len - len(s)):
                output.append(word_num)

        j = j + 1
        outputs.append(output)

    k = 0
    f = open("train.csv", "w", newline="")
    f2 = open("test.csv", "w", newline="")
    writer1 = csv.writer(f)
    writer2 = csv.writer(f2)
    for i in outputs:
        if k >= train_num:
            writer2.writerow(i)
        else:
            writer1.writerow(i)
        k = k + 1
    f2.close()
    f.close()
