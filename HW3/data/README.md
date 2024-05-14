# 处理后数据

## 训练集和测试集

本文件夹存放经过处理后的数据，以 csv 格式存储，每一行对应一句话。

例：对于训练集中这样一段语料：

```plain
1	"The system as described above has its greatest application in an arrayed <e1>configuration</e1> of antenna <e2>elements</e2>."
Component-Whole(e2,e1)
Comment: Not a collection: there is structure here, organisation.
```

处理成 csv 后应为如下格式：

```csv
1,3,16,12,12,15,15,<各单词的 id>
```

csv 各项的解释如下：

- 第一项是该语料的 ID；
- 第二项是该语料关系对应的 ID（各关系对应的 ID 见后文）；
- 第三项是该语料的长度；
- 第四和第五项是 `e1` 的开始和结束词的位置下标，下标从 0 开始计算；
- 第六和第七项是 `e2` 的开始和结束词的位置下标，下标从 0 开始计算；
- 后面若干项按顺序给出原句子每个词对应的 ID（单词 ID 从 0 开始计算）。为了做到各行项数相同，需设定一个语料长度上限 $len$，语料实际包含单词数低于 $len$ 时，需要在最后填充若干个值等于字典总大小的项，使得该部分的项数补足 $len$ 项。

在处理语料时，原句的所有标点符号均会被移除（连字符 `-` 除外），包括 `<e1>` 这样的标记也会移除。需要注意的是 `I'm` 这样的缩写形式，应处理为 `I` 和 `m` 这两个单词。

各关系对应的 ID 如下面所示：

```plain
0 Cause-Effect(e1,e2)
1 Cause-Effect(e2,e1)
2 Component-Whole(e1,e2)
3 Component-Whole(e2,e1)
4 Content-Container(e1,e2)
5 Content-Container(e2,e1)
6 Entity-Destination(e1,e2)
7 Entity-Destination(e2,e1)
8 Entity-Origin(e1,e2)
9 Entity-Origin(e2,e1)
10 Instrument-Agency(e1,e2)
11 Instrument-Agency(e2,e1)
12 Member-Collection(e1,e2)
13 Member-Collection(e2,e1)
14 Message-Topic(e1,e2)
15 Message-Topic(e2,e1)
16 Product-Producer(e1,e2)
17 Product-Producer(e2,e1)
18 Other
```

## 预训练词嵌入表示

预训练数据来自 [glove.6B.50d.txt](https://www.kaggle.com/datasets/devjyotichandra/glove6b50dtxt)，每个单词均表示为一个 50 维向量。

为了使用该数据，将该数据进行预处理，只保留训练集和测试集语料中出现过的单词，并将这些单词按 ID 顺序排好，存储到一个 csv 文件中。csv 文件每行只包含 50 维的向量表示，不包含单词对应的 ID。

预训练数据中未出现的单词，其向量值将被随机初始化，要求初始化后每位上的值位于 $[-1,1]$ 之间。
