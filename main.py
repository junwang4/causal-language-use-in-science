import os, sys, time, re, json
import numpy as np
import pandas as pd
from IPython.display import display
from sklearn.metrics import accuracy_score, classification_report, precision_recall_fscore_support, confusion_matrix
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


BERT_NAME_2_MODEL = {'bert' : 'bert-base-cased',
                     'biobert' : 'biobert-base-cased'
                    }

BERT_MODEL_NAME = 'bert'
BERT_MODEL_NAME = 'biobert'

BERT_MODEL = BERT_NAME_2_MODEL[BERT_MODEL_NAME]

PRJ = 'EMNLP'
label_name = {0:'none', 1:'causal', 2:'cond', 3:'corr'}
NUM_CLASSES = len(label_name)

K = 5
EPOCHS = 5

DATA_DIR = 'data'
MODEL_DIR = f'model/{PRJ}_{BERT_MODEL_NAME}'

RANDOM_STATE = 0


##################################
#
# functions
#
def get_train_data_csv_fpath(): return f'{DATA_DIR}/pubmed_causal_language_use.csv'
def read_train_data(): return pd.read_csv(get_train_data_csv_fpath(), usecols=['sentence', 'label'], encoding = 'utf8', keep_default_na=False)
def clean_str(s): return s.strip() # BioBert or cased-Bert works better with cased letters, so don't use s.lower()

def get_class_weight(labels):
    class_weight = [x for x in compute_class_weight("balanced", range(len(set(labels))), labels)]
    print('- auto-computed class weight:', class_weight)
    return class_weight

def get_model_bin_file(fold=0):
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)
        print(f'\ncreate a new folder for storing BERT model: "{MODEL_DIR}"\n')
    if fold>=0:
        return f'{MODEL_DIR}/K{K}_epochs{EPOCHS}_{fold}.bin'
    elif fold==-1:
        return f'{MODEL_DIR}/full_epochs{EPOCHS}.bin'
    else:
        print('Wrong value of fold:', fold)
        sys.exit()

def get_pred_csv_file(mode='train'):
    pred_folder = f'./pred/{PRJ}_{BERT_MODEL_NAME}_{mode}'
    if not os.path.exists(pred_folder):
        os.makedirs(pred_folder)
        print('\ncreate new folder for prediction results:', pred_folder, '\n')
    if mode == 'train':
        return f'{pred_folder}/K{K}_epochs{EPOCHS}.csv'
    elif mode == 'apply':
        return f'{pred_folder}/epochs{EPOCHS}.csv'
    else:
        print('- wrong mode:', mode, '\n')

def get_train_test_data(df, fold=0):
    df['sentence'] = df.sentence.apply(clean_str)
    skf = StratifiedKFold(n_splits=K, shuffle=True, random_state=RANDOM_STATE)
    for i, (train_index, test_index) in enumerate(skf.split(df.sentence, df.label)):
        if i == fold:
            break
    train = df.iloc[train_index]
    test = df.iloc[test_index]

    print(f"ALL: {len(df)}   TRAIN: {len(train)}   TEST: {len(test)}")
    label_list = np.unique(train.label)
    return train, test, label_list

def train_model(train, model_file_to_save, epochs=3, val_frac=0.1, class_weight=None):
    X_train = train['sentence']
    y_train = train['label']

    max_seq_length, train_batch_size, lr = 128, 32, 2e-5

    model = BertClassifier(bert_model=BERT_MODEL, random_state=RANDOM_STATE, \
                            class_weight=class_weight, max_seq_length=max_seq_length, \
                            train_batch_size=train_batch_size, learning_rate=lr, \
                            epochs=epochs, validation_fraction=val_frac)
    print(model)
    model.fit(X_train, y_train)
    model.save(model_file_to_save)
    print(f'\n- model saved to: {model_file_to_save}\n')
    return model


def train_one_full_model():
    df_train = read_train_data()
    class_weight = get_class_weight(df_train['label'])

    model_file_to_save = get_model_bin_file(fold=-1) # -1: for one full model
    train_model(df_train, model_file_to_save, epochs=EPOCHS, val_frac=0.15, class_weight=None)


def train_KFold_model():
    df = read_train_data()
    print('- label value counts:')
    print(df.label.value_counts())

    y_test_all, y_pred_all = [], []
    results = []
    df_out_proba = None
    for fold in range(K):
        train_data, test_data, label_list = get_train_test_data(df, fold)

        model_file = get_model_bin_file(fold)
        use_class_weight_for_unbalanced_data = True
        class_weight = get_class_weight(df['label']) if use_class_weight_for_unbalanced_data else None

        val_frac = 0.05
        model = train_model(train_data, model_file, epochs=EPOCHS, val_frac=val_frac, class_weight=class_weight)

        X_test = test_data['sentence']
        y_test = test_data['label']
        y_test_all += y_test.tolist()

        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)
        del model

        y_pred_all += y_pred.tolist()

        tmp = pd.DataFrame(data=y_proba, columns=[f'c{i}' for i in range(NUM_CLASSES)])
        tmp['confidence'] = tmp.max(axis=1)
        tmp['winner'] = tmp.idxmax(axis=1)
        tmp['sentence'] = X_test.tolist()
        tmp['label'] = y_test.tolist()
        df_out_proba = tmp if df_out_proba is None else pd.concat((df_out_proba, tmp))

        acc = accuracy_score(y_pred, y_test)
        res = precision_recall_fscore_support(y_test, y_pred, average='macro')
        print(f'\nAcc: {acc:.3f}      F1:{res[2]:.3f}       P: {res[0]:.3f}   R: {res[1]:.3f} \n')

        item = {'Acc': acc, 'weight': len(test_data)/len(df), 'size': len(test_data)}
        item.update({'P':res[0], 'R':res[1], 'F1':res[2]})
        for cls in np.unique(y_test):
            res = precision_recall_fscore_support(y_test, y_pred, average=None, labels=[cls])
            for i, scoring in enumerate('P R F1'.split()):
                item['{}_{}'.format(scoring, cls)] = res[i][0]
        results.append(item)

        acc_all = np.mean(np.array(y_pred_all) == np.array(y_test_all))
        res = precision_recall_fscore_support(y_test_all, y_pred_all, average='macro')
        print( f'\nAVG of {fold+1} folds  |  Acc: {acc_all:.3f}    F1:{res[2]:.3f}       P: {res[0]:.3f}   R: {res[1]:.3f} \n')

    # show an overview of the performance
    df_2 = pd.DataFrame(list(results)).transpose()
    df_2['avg'] = df_2.mean(axis=1)
    df_2 = df_2.transpose()
    df_2['size'] = df_2['size'].astype(int)
    display(df_2)

    # put together the results of all 5-fold tests and save
    output_pred_csv_file_train = get_pred_csv_file(mode='train')
    df_out_proba.to_csv(output_pred_csv_file_train, index=False, float_format="%.3f")
    print(f'\noutput all {K}-fold test results to: "{output_pred_csv_file_train}"\n')


def apply_one_full_model_to_new_sentences():
    fpath_unseen_data = "data/sample_new_sentences.csv"
    columns = ['pmid', 'sentence']
    df = pd.read_csv(fpath_unseen_data, usecols=columns)

    print(f'all: {len(df):,}    unique sentences: {len(df.sentence.unique()):,}     papers: {len(df.pmid.unique()):,}')

    output_pred_file = get_pred_csv_file('apply')
    print(output_pred_file)

    model_file = get_model_bin_file(fold=-1)  # -1: indicating this is the model trained on all data
    print(f'\n- use trained model: {model_file}\n')

    model = load_model(model_file)

    model.eval_batch_size = 32
    y_prob = model.predict_proba(df.sentence)

    df_out = pd.DataFrame(data=y_prob, columns=[f'c{i}' for i in range(NUM_CLASSES)])
    df_out['confidence'] = df_out.max(axis=1)
    df_out['winner'] = df_out.idxmax(axis=1)
    for col in columns:
        df_out[col] = df[col]

    df_out.to_csv(output_pred_file, index=False, float_format="%.3f")
    print(f'\n- output prediction to: {output_pred_file}\n')


def evaluate_and_error_analysis():
    df = pd.read_csv(get_pred_csv_file(mode='train')) # -2: a flag indicating putting together the results on all folds
    df['pred'] = df['winner'].apply(lambda x:int(x[1])) # from c0->0, c1->1, c2->2, c3->3

    print('\nConfusion Matrix:\n')
    cm = confusion_matrix(df.label, df.pred)
    print(cm)

    print('\n\nClassification Report:\n')
    print(classification_report(df.label, df.pred))

    out = ["""
<style>
    * {font-family:arial}
    body {width:900px;margin:auto}
    .wrong {color:red;}
    .hi1 {font-weight:bold}
</style>
<table cellpadding=10>
"""]


    row = f'<tr><th><th><th colspan=4>Predicted</tr>\n<tr><td><td>'
    for i in range(NUM_CLASSES):
        row += f"<th>{label_name[i]}"
    for i in range(NUM_CLASSES):
        row += f'''\n<tr>{'<th rowspan=4>Actual' if i==0 else ''}<th align=right>{label_name[i]}'''
        for j in range(NUM_CLASSES):
            row += f'''<td align=right><a href='#link{i}{j}'>{cm[i][j]}</a></td>'''
    out.append(row + "</table>")

    for i in range(NUM_CLASSES):
        for j in range(NUM_CLASSES):
            row = f"<div id=link{i}{j}><h2>{label_name[i]} => {label_name[j]}</h2><table cellpadding=10>"
            row += f'<tr><th><th>Sentence<th>Label<th>{label_name[0]}<th>{label_name[1]}<th>{label_name[2]}<th>{label_name[3]}<th>mark</tr>'
            out.append(row)

            df_ = df[(df.label==i) & (df.pred==j)]
            df_ = df_.sort_values('confidence', ascending=False)

            cnt = 0
            for c0, c1, c2, c3, sentence, label, pred in zip(df_.c0, df_.c1, df_.c2, df_.c3, df_.sentence, df_.label, df_.pred):
                cnt += 1
                mark = "" if label == pred else "<span class=wrong>oops</span>"
                item = f"""<tr><th valign=top>{cnt}.
                        <td valign=top width=70%>{sentence}
                        <td valign=top>{label_name[label]}
                        <td valign=top class=hi{int(c0>max(c1,c2,c3))}>{c0:.2f}
                        <td valign=top class=hi{int(c1>max(c0,c2,c3))}>{c1:.2f}
                        <td valign=top class=hi{int(c2>max(c0,c1,c3))}>{c2:.2f}
                        <td valign=top class=hi{int(c3>max(c0,c1,c2))}>{c3:.2f}
                        <td valign=top>{mark}</tr>"""
                out.append(item)
            out.append('</table></div>')

    html_file_output = '/var/www/html/a.html'
    html_file_output = '/tmp/a.html'
    with open(html_file_output, 'w') as fout:
        fout.write('\n'.join(out))
        print(f'\n- HTML file output to: "{html_file_output}"\n')


def main():
    task_func = {
        'train_kfold': train_KFold_model,
        'train_one_full_model': train_one_full_model,
        'evaluate_and_error_analysis': evaluate_and_error_analysis,
        'apply_one_full_model_to_new_sentences': apply_one_full_model_to_new_sentences
    }

    task = 'evaluate_and_error_analysis' # error analysis: outpu an HTML file for checking errors
    task = 'apply_one_full_model_to_new_sentences'
    task = 'train_one_full_model' # use all data (with about 10 percent used as validation) to train a model, which will be used to new sentences
    task = 'train_kfold'

    task_func[task]()


if __name__ == "__main__":
    tic = time.time()
    main()
    print(f'time used: {time.time()-tic:.0f} seconds')
