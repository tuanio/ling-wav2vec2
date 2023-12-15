from metric import Correct_Rate, Accuracy, Align, insertions, deletions, substitutions
import pandas as pd
import argparse
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--subset', type=str, default='private_test')
parser.add_argument('--out-file', type=str, default='')

args = parser.parse_args()

predict_df = pd.read_csv(args.subset + '_submission.csv')
truth_df = pd.read_csv('md_d_metric/' + args.subset + '_phones.csv')

out_file = open(args.out_file, 'a')

def process(text):
    return ' '.join(text.split(' $ ')).split(' ')

predict_data = dict(zip(predict_df.id, predict_df.predict))
truth_data = {key: {'canonical': process(can), 'transcript': process(trans)} for key, can, trans in zip(truth_df.id, truth_df.canonical, truth_df.transcript)}

canon_pred_f = open('md_d_metric/aligned/ref_our_detail', 'w')
canon_trans_f = open('md_d_metric/aligned/ref_human_detail', 'w')
trans_pred_f = open('md_d_metric/aligned/human_our_detail', 'w')

def cal_IDS(s1, s2):
    a1, a2 = Align(s1, s2)
    l = len(a1)
    I = insertions(a1, a2)
    D = deletions(a1, a2)
    S = substitutions(a1, a2)

    false = I + D[0] + S

    return l - false, I, D, S, a1, a2

def get_op(seq1, seq2):
    op = []
    for i in range(len(seq1)):
        if seq1[i]!="<eps>" and seq2[i]=="<eps>":
            op.append('D')
        elif seq1[i] == "<eps>" and seq2[i]!="<eps>" :
            op.append('I')
        elif (seq1[i]!=seq2[i]) and seq2[i]!="<eps>" and seq1[i]!="<eps>":
            op.append("S")
        else:
            op.append("C")
    return op

def get_align(k, s1, s2):
    a1, a2 = Align(s1, s2)

    I = insertions(a1, a2)
    D = deletions(a1, a2)[0]
    S = substitutions(a1, a2)
    C = len(a1) - I - D - S

    return [
        k + ' ref ' + ' '.join(a1),
        k + ' hyp ' + ' '.join(a2),
        k + ' op ' + ' '.join(get_op(a1, a2)),
        k + f' #csid {C} {S} {I} {D}'
    ]

list_correct_rate = []
list_accuracy = []
list_len = []

for data_id in predict_data:
    pred = predict_data[data_id].split(' ')
    trans = truth_data[data_id]['transcript']
    canon = truth_data[data_id]['canonical']
    canon_pred_f.write('\n'.join(get_align(data_id, canon, pred)) + '\n')
    canon_trans_f.write('\n'.join(get_align(data_id, canon, trans)) + '\n')
    trans_pred_f.write('\n'.join(get_align(data_id, trans, pred)) + '\n')

    correct_rate, len_, _ = Correct_Rate(trans, pred)
    acc, len_ = Accuracy(trans, pred)

    list_correct_rate.append((len_ - correct_rate) / len_)
    list_accuracy.append((len_ - acc) / len_)
    list_len.append(len_)

list_correct_rate = np.array(list_correct_rate)
list_accuracy = np.array(list_accuracy)
list_len = np.array(list_len)

corr_rate = (list_correct_rate * list_len).sum() / list_len.sum()
acc = (list_accuracy * list_len).sum() / list_len.sum()

corr_rate = round(corr_rate, 4)
acc = round(acc, 4)

print('*' * 2 + f" MD&D for [{args.subset}] " + '*' * 2, file=out_file)
print('Correct Rate:', corr_rate, file=out_file)
print("Accuracy:", acc, file=out_file)