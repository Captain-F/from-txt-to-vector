from fromTxtToVec.corpus_build import Corpus
from keras.preprocessing.sequence import pad_sequences
from keras.utils import np_utils



class Pad:

    def __init__(self, sent_maxlen):
        self.sent_maxlen = sent_maxlen

    def pad_func(self, x):

        xPad = pad_sequences(x, maxlen=self.sent_maxlen, padding='post', truncating='post')

        return xPad

    def trans(self):

        id2token = Corpus().token_dict()
        label2id = Corpus().label_dict()
        token2id = {token: idx for idx, token in id2token.items()}

        return token2id, label2id

    def pad_seq(self, sents, labels):

        token2id, label2id = self.trans()
        token2id_sents, label2id_labels = [], []
        for sent in sents:
            s = []
            for token in sent:
                s.append(token2id[token])
            token2id_sents.append(s)
        for label in labels:
            l = []
            for lab in label:
                l.append(label2id[lab])
            label2id_labels.append(l)

        pad_sents = self.pad_func(token2id_sents)
        pad_labels = self.pad_func(label2id_labels)
        pad_labels = np_utils.to_categorical(pad_labels, len(label2id))


        return pad_sents, pad_labels