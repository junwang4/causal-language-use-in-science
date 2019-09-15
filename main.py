import os, sys, time, re, json
import numpy as np
import pandas as pd
from sklearn import metrics
from IPython.display import display
from sklearn.metrics import classification_report, precision_recall_fscore_support, confusion_matrix
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.utils.class_weight import compute_class_weight
from bert_sklearn import BertClassifier, load_model

NVIDIA_TITAN_XP = "0"
NVIDIA_1080TI = "1"
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = NVIDIA_TITAN_XP
os.environ["CUDA_VISIBLE_DEVICES"] = NVIDIA_1080TI

pd.options.display.max_colwidth = 500
pd.options.display.width = 1000
pd.options.display.precision = 3
np.set_printoptions(precision=3)


BERT_MODEL_NAME = 'bert'
BERT_MODEL_NAME = 'biobert'
BERT_NAME_2_MODEL = {'bert' : 'bert-base-cased',
                     'biobert' : 'biobert-base-cased'
                    }
BERT_MODEL = BERT_NAME_2_MODEL[BERT_MODEL_NAME]

PRJ = 'EMNLP'
NUM_CLASSES = 4

K = 5
EPOCHS = 5

DATA_DIR = 'data'
MODEL_DIR = f'model/{PRJ}_{BERT_MODEL_NAME}'



##################################
#
# functions
#
def get_train_data_csv_fpath(): return f'{DATA_DIR}/pubmed_causal_language_use.csv'

def clean_str(string): return string.strip() # for BioBert or cased-Bert

def get_model_bin_file(K, fold):
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)
    return f'{MODEL_DIR}/K{K}_epochs{EPOCHS}_{fold}.bin'


def get_train_dev_data(df, K=10, fold=0):
    df.columns=['text', 'label']
    df['text'] = df.text.apply(clean_str)
    random_state = 0  

    skf = StratifiedKFold(n_splits=K, shuffle=True, random_state=random_state)
    for i, (train_index, test_index) in enumerate(skf.split(df.text, df.label)):
        if i == fold:
            break
    train = df.iloc[train_index]
    dev = df.iloc[test_index]

    print(f"ALL: {len(df)}   TRAIN: {len(train)}   TEST: {len(dev)}")
    label_list = np.unique(train.label)
    return train, dev, label_list


def train_model(train, savefile, epochs=3, val_frac=0.1, class_weight=None):
    X_train = train['text']
    y_train = train['label']

    max_seq_length, train_batch_size, lr = 128, 32, 2e-5

    model = BertClassifier(bert_model=BERT_MODEL, random_state=0, class_weight=class_weight, max_seq_length=max_seq_length, train_batch_size=train_batch_size, learning_rate=lr, epochs=epochs, validation_fraction=val_frac)

    print(model)

    model.fit(X_train, y_train)

    model.save(savefile)
    print(f'- model saved to: {savefile}')
    return model


def train_KFold_model():
    y_test_all, y_pred_all = [], []

    train_data_csv_fpath = get_train_data_csv_fpath()

    df = pd.read_csv(train_data_csv_fpath, usecols=['sentence', 'label'], encoding = 'utf8', keep_default_na=False)
    print('- label value counts:')
    print(df.label.value_counts())

    results = []
    for fold in range(K):
        train_data, dev_data, label_list = get_train_dev_data(df, K, fold)

        model_file = get_model_bin_file(K, fold)
        if 1:
            class_weight = [x for x in compute_class_weight("balanced", range(NUM_CLASSES), train_data['label'])]
            print('- auto-computed class weight:', class_weight)

            val_frac = 0.0
            model = train_model(train_data, model_file, epochs=EPOCHS, val_frac=val_frac, class_weight=class_weight)

        X_test = dev_data['text']
        y_test = dev_data['label']
        y_test_all += y_test.tolist()

        y_pred = model.predict(X_test)
        del model
        y_pred_all += y_pred.tolist()

        acc = metrics.accuracy_score(y_pred, y_test)
        res = precision_recall_fscore_support(y_test, y_pred, average='macro')
        print(f'\nAcc: {acc:.3f}      F1:{res[2]:.3f}       P: {res[0]:.3f}   R: {res[1]:.3f} \n')

        item = {'Acc': acc, 'weight': len(dev_data)/len(df), 'size': len(dev_data)}
        item.update({'P':res[0], 'R':res[1], 'F1':res[2]})
        for cls in np.unique(y_test):
            res = precision_recall_fscore_support(y_test, y_pred, average=None, labels=[cls])
            for i, scoring in enumerate('P R F1'.split()):
                item['{}_{}'.format(scoring, cls)] = res[i][0]
        results.append(item)

        acc_all = np.mean(np.array(y_pred_all) == np.array(y_test_all))
        res = precision_recall_fscore_support(y_test_all, y_pred_all, average='macro')
        print( f'\nAVG of {fold+1} folds  |  Acc: {acc_all:.3f}    F1:{res[2]:.3f}       P: {res[0]:.3f}   R: {res[1]:.3f} \n')

    if 1:
        df_2 = pd.DataFrame(list(results)).transpose()
        df_2['avg'] = df_2.mean(axis=1)
        df_2 = df_2.transpose()
        df_2['size'] = df_2['size'].astype(int)
        display(df_2)


def main():
    train_KFold_model()


if __name__ == "__main__":
    tic = time.time()
    main()
    print(f'time used: {time.time()-tic:.0f}s')

