import pickle
from collections import Counter

class Dictionary(object):
    def __init__(self, word2idx=None, idx2word=None):
        if word2idx is None:
            word2idx = {}
        if idx2word is None:
            idx2word = []
        self.word2idx = word2idx
        self.idx2word = idx2word

    @property
    def ntoken(self):
        return len(self.word2idx)

    @property
    def padding_idx(self):
        return len(self.word2idx)

    def tokenize(self, sentence, add_word):
        sentence = sentence.lower()
        sentence = sentence.replace(',', '').replace('?', '').replace('\'s', ' \'s')
        words = sentence.split()
        tokens = []
        if add_word:
            for w in words:
                tokens.append(self.add_word(w))
        else:
            for w in words:
                # the least frequent word (`bebe`) as UNK for Visual Genome dataset
                tokens.append(self.word2idx.get(w, self.padding_idx-1))
        return tokens

    def dump_to_file(self, path):
        pickle.dump([self.word2idx, self.idx2word], open(path, 'wb'))
        print('dictionary dumped to %s' % path)

    @classmethod
    def load_from_file(cls, path):
        print('loading dictionary from %s' % path)
        word2idx, idx2word = pickle.load(open(path, 'rb'))
        d = cls(word2idx, idx2word)
        return d

    def add_word(self, word):
        if word not in self.word2idx:
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word) - 1
        return self.word2idx[word]

    def __len__(self):
        return len(self.idx2word)


# 定义样本数据
samples = [('category1', 'this is the first question'),
           ('category2', 'this is the second question'),
           ('category1', 'this is another question'),
           ('category3', 'yet another question')]

# 构建词表
counter = Counter()
for _, question in samples:
    words = question.strip().split()  # 将问题划分成单词
    counter.update(words)  # 统计单词出现的次数

# 根据单词出现次数降序排序，将最常见的单词放在前面
sorted_words = sorted(counter, key=counter.get, reverse=True)

# 构建word2idx和idx2word词典
word2idx = {}
idx2word = {}
for idx, word in enumerate(sorted_words):
    word2idx[word] = idx
    idx2word[idx] = word

# 保存词典到pkl文件
with open('vocab.pkl', 'wb') as f:
    pickle.dump((word2idx, idx2word), f)

dictionary = Dictionary.load_from_file('vocab.pkl')

tokens = dictionary.tokenize(samples[0][1], False)