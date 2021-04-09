from fromTxtToVec.corpus_build import Corpus
from fromTxtToVec.pad import Pad
from fromTxtToVec.BERT_feat import ExtractBertEmb
from fromTxtToVec.train_vector import Embedding
import numpy as np



class To_vec:

    def __init__(self, mode, sent_maxlen):
        self.mode = mode
        self.sent_maxlen = sent_maxlen

    def vector(self):
        sents, labels = Corpus().read_txt()
        pad_sents, pad_labels = Pad(self.sent_maxlen).pad_seq(sents, labels)

        if self.mode == 'w2v':
            sents_, labels_ = pad_sents, pad_labels

        elif self.mode == 'bert':
            path = input('请输入BERT模型的绝对路径or相对路径...')
            extractor = ExtractBertEmb(bert_path=path)
            granu = input('请输入抽取的粒度: token or cls')
            if granu == 'token':
                bert_sents = extractor.extract(sentences=[''.join(i) for i in sents], granularity=granu)
                sents_ = []
                for s in bert_sents:
                    if len(s) >= int(self.sent_maxlen):
                        matrix = s[:int(self.sent_maxlen)]
                    else:
                        matrix = np.zeros((int(self.sent_maxlen), 768))
                        for idx, i in enumerate(s):
                            matrix[idx] = i
                    sents_.append(matrix)
            elif granu == 'token':
                sents_ = extractor.extract(sentences=[''.join(i) for i in sents], granularity=granu)
            labels_ = pad_labels

        return np.array(sents_), labels_

    def w2v_matrix(self, emb_size):

        sents, labels = Corpus().read_txt()
        matrix = Embedding(emb_size=emb_size).w2v(corpus=sents)

        return matrix






