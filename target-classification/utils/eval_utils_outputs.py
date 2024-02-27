import pandas as pd, numpy as np
from sklearn.metrics import precision_recall_fscore_support

def calculateF1(model, seed):
    file = '../output/Stance_Merge_Unrelated/predictions_{}_seed_{}.csv'.format(model, seed)
    gt_target = pd.read_csv(file, usecols=[1], encoding='ISO-8859-1')
    mapped_target = pd.read_csv(file, usecols=[2], encoding='ISO-8859-1')

    df = pd.concat([gt_target, mapped_target], axis=1)
    df.columns = ['Target', 'Mapped Target']

    labels = ['Joe Biden', 'Bernie Sanders', 'Donald Trump', 'abortion', 'cloning', 'death penalty', 'gun control', 'marijuana legalization', 'minimum wage', 'nuclear energy', 'school uniforms', 'Atheism', 'Feminist Movement', 'Hillary Clinton', 'face masks', 'fauci', 'stay at home orders', 'school closures']
    labels_to_idx = {v.lower():k for k, v in enumerate(labels)}
    labels_to_idx['unrelated'] = len(labels_to_idx)

    f1_target = [[] for _ in range(5)]
    preds = []; gt_truths = []
    for idx, row in df.iterrows():
        
        if row['Target'] == 'Legalization of Abortion':
            row['Target'] = 'abortion'
        pred  = labels_to_idx[row['Mapped Target'].lower()]
        y_true = labels_to_idx[row['Target'].lower()]
        preds.append(pred); gt_truths.append(y_true)

    f1_target_avg1 = 100.0 * precision_recall_fscore_support(gt_truths[:1080], preds[:1080], average='micro')[2]
    f1_target_avg2 = 100.0 * precision_recall_fscore_support(gt_truths[1080:1880], preds[1080:1880], average='micro')[2]
    f1_target_avg3 = 100.0 * precision_recall_fscore_support(gt_truths[1880:6989], preds[1880:6989], average='micro')[2]
    f1_target_avg4 = 100.0 * precision_recall_fscore_support(gt_truths[6989:9146], preds[6989:9146], average='micro')[2]
    f1_target_avg5 = 100.0 * precision_recall_fscore_support(gt_truths, preds, average='micro')[2]
    print('\n\n********Statistics for model {}, seed {}********'.format(model, seed))
    print('F1 for {} = {}'.format(model, f1_target_avg5))
    print('F1 for {} = {}'.format('SemEval', f1_target_avg1))
    print('F1 for {} = {}'.format('AM', f1_target_avg3))
    print('F1 for {} = {}'.format('PStance', f1_target_avg4))
    print('F1 for {} = {}'.format('Covid19', f1_target_avg2))

for seed in [0, 112, 342]:
    calculateF1('BiLSTM', seed)
    calculateF1('Bert', seed)
    calculateF1('Bertweet', seed)

# after getting the results for three seeds, just average them to get the results for Table 3