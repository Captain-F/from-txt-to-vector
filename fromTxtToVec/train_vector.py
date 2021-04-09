import os
from gensim.models import Word2Vec
from fromTxtToVec.corpus_build import Corpus
import numpy as np
import logging


class Embedding:

    def __init__(self, emb_size):

        self.emb_size = emb_size


    def w2v(self, corpus):

        id2token = Corpus().token_dict()
        matrix = np.zeros((len(id2token) + 1, self.emb_size))
        model = Word2Vec(corpus, size=self.emb_size, window=5, min_count=3)

        count = 0
        for idx, token in id2token.items():
            try:
                vector = model.wv[token]
                matrix[idx] = vector
            except:
                matrix[idx] = np.random.uniform(-0.25, 0.25, self.emb_size)
                count += 1

        print('嵌入模型训练完成...')
        print('共有{}未登录词...'.format(count))

        return matrix



