from sklearn import metrics


def prec_rec_f1(y_true, y_pred):
    if max(y_true) == 0:
        print('Warning: only normal data in test part')
        return 0, 0, 0
    precision, recall, fscore, _ = metrics.precision_recall_fscore_support(y_true, y_pred, pos_label=1, average='binary')
    return precision, recall, fscore
