from keras_bert import extract_embeddings
import numpy as np


class ExtractBertEmb:

    # extract the features of sentences;
    # shape for one sentence (extracted by BERT): (length + 2, 768);
    # the output shape: (length, 768);

    def __init__(self, bert_path):

        self.model_path = bert_path

    def extract(self, sentences, granularity):

        feats = extract_embeddings(self.model_path, sentences)
        if granularity == 'token':
            feats = np.array([feat[1:-1] for feat in feats])
        elif granularity == 'cls':
            feats = np.array([feat[0] for feat in feats])

        return feats

'''
    Example:
        txts = ['我爱南京', '我爱北京']
        extractor = ExtractBertEmb(bert_path=r'bert_path')
        feats = extractor.extract(txts, 'cls')
        feats = extractor.extract(txts, 'token')

        print(feats.shape)
'''

