from pathlib import Path
import re

'''
    input:
        txt:
            我\tB
            爱\tI
            南\tI
            京\tI
            。\tO
            ...
    output:
        dict:{'0': 0, 'B': 1, ...}
        sents:[['我','爱', '南', '京', ...,], ..., ]
        labels:[['B', 'I', 'I', 'O', ...,], ..., ]
'''


class Corpus:

    def __init__(self):

        self.root = list(Path().glob('*.txt'))

    def read_txt(self):
        '''
        input:
            txt:
                我\tB
                爱\tI
                南\tI
                京\tI
                。\tO
                ...
        return:
            sents:[['我','爱', '南', '京', ...,], ..., ]
            labels:[['B', 'I', 'I', 'O', ...,], ..., ]
        '''

        with open(self.root[0], 'r', encoding='utf8')as f:
            token_labs = f.readlines()

        sent, sents, label, labels = [], [], [], []
        for t_l in token_labs:

            if t_l != '\n':
                token = t_l.strip().split('\t')[0]
                lab = t_l.strip().split('\t')[1]
                sent.append(token)
                label.append(lab)
            else:
                sents.append(sent)
                labels.append(label)
                sent = []
                label = []

        return sents, labels


    def label_dict(self):

        _, labels = self.read_txt()
        labels_set = list(set([j for i in labels for j in i]))
        B_label = [i for i in labels_set if i.split('-')[0] == 'B']
        I_label = [i for i in labels_set if i.split('-')[0] == 'I']
        B_I_list = []
        for i in B_label:
            for jdx, j in enumerate(I_label):
                if i.split('-')[1] == j.split('-')[1]:
                    B_I_list.extend([i, j])
        #B_I_list_ = list(set(B_I_list)).sort(key=B_I_list.index)
        B_I_dict = {i: idx + 1 for idx, i in enumerate(B_I_list)}
        ini_dict = {'0': 0, 'O': len(labels_set)}
        lab2id = dict(B_I_dict, **ini_dict)

        return lab2id


    def token_dict(self):

        sents, _ = self.read_txt()
        raw_tokens = [j for i in sents for j in i]
        tokens = list(set(raw_tokens))
        tokens.sort(key=raw_tokens.index)
        id2token = {idx + 1: token for idx, token in enumerate(tokens)}

        return id2token


    def get_path(self):

        paths = list(Path().glob('*.txt'))
        names = [re.split(r'[\\.]', str(path))[1] for path in paths]

        return paths, names