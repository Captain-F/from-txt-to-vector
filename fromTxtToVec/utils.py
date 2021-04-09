import joblib


def read_pickle(path):
    with open(path, 'rb')as f:
        feats = joblib.load(f)

    return feats

def save_pickle(path, feats):
    with open(path, 'wb')as f:
        joblib.dump(feats, f)