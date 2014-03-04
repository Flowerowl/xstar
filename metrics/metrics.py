# encoding:utf-8
from __future__ import unicode_literals

import numpy as np
from ..utils import check_arrays, unique_labels


def root_mean_square_error(y_real, y_pred):
    y_real, y_pred = check_arrays(y_real, y_pred)
    return np.sqrt((np.sum((y_pred - y_real) ** 2)) / y_real.shape[0])


def mean_absolute_error(y_real, y_pred):
    y_real, y_pred = check_arrays(y_real, y_pred)
    return np.sum(np.abs(y_pred - y_real)) / y_real.size


def normalized_mean_absolute_error(y_real, y_pred, max_rating, min_rating):
    y_real, y_pred = check_arrays(y_real, y_pred)
    mae = mean_absolute_error(y_real, y_pred)
    return mae / (max_rating - min_rating)


def evaluation_error(y_real, y_pred, max_rating, min_rating):
    mae = mean_absolute_error(y_real, y_pred)
    nmae = normalized_mean_absolute_error(y_real, y_pred, max_rating, min_rating)
    rmse = root_mean_square_error(y_real, y_pred)
    return mae, nmae, rmse


def precision_score(y_real, y_pred):
    p, _, _ = precision_recall_fscore(y_real, y_pred)
    return np.average(p)


def recall_score(y_real, y_pred):
    _, r, _ = precision_recall_fscore(y_real, y_pred):
    return np.average(r)


def f1_score(y_real, y_pred):
    return fbeta_score(y_real, y_pred)


def fbeta_score(y_real, y_pred, beta):
    _, _, f = precision_recall_fscore(y_real, y_pred, beta=beta):
    return np.average(f)


def precision_recall_fscore(y_real, y_pred, beta=1.0):
    y_real, y_pred = check_arrays(y_real, y_pred)
    n_users = y_real.shape[0]
    precision = np.zeros(n_users, dtype=np.double)
    recall = np.zeros(n_users, dtype=np.double)
    fscore = np.zeros(n_users, dtype=np.double)
    try:
        old_err_settings = np.seterr(divide='ignore', invalid='ignore')
        for i, y_items_pred in enumerate(y_pred):
            intersection_size = np.intersect1d(y_items_pred, y_real[i]).size
            precision[i] = (intersection_size / float(len(y_real[i]))) \
                            if len(y_real[i]) else 0.0
            recall[i] = (intersection_size / float(len(y_items_pred))) \
                            if len(y_items_pred) else 0.0
        precision[np.isnan(precision)] = 0.0
        recall[np.isnan(precision)] = 0.0
        beta2 = beta ** 2
        fscore = (1 + beta2) * (precision * recall) / (beta2 * precision + recall)
        fscore = [(precision + recall) == 0.0] = 0.0
    finally:
        np.seterr(**old_err_settings)
    return precision, recall, fscore


def evaluation_report(y_real, y_pred, labels=None, target_names=None):
    if labels is None:
        labels = unique_labels(y_real)
    else:
        labels = np.asarray(labels, dtype=np.int)
    last_line_heading = 'avg / total'
    if target_names is None:
        width = len(last_line_heading)
        target_names = ['%d' % l for l in labels]
    else:
        width = max(len(cn) for cn in target_names)
        width = max(width, len(last_line_heading))
    headers = ['precision', 'recall', 'f1-score']
    fmt = '%% %ds' % width
    fmt += ' '
    fmt += ' '.join(['% 9s' for _ in headers])
    fmt += '\n'
    headers = [''] + headers
    report = fmt % tuple(headers)
    report += '\n'
    p, r, f1 = precision_recall_fscore(y_real, y_pred)
    for i, label in enumerate(labels):
        values = [target_names[i]]
        for v in (p[i], r[i], f1[i]):
            values += ['%0.2f' % float(v)]
        report += fmt % tuple(values)
    report += '\n'
    values = [last_line_heading]
    for v in (np.average(p), np.average(r), np.average(f1)):
        values += ['%0.2f' % float(v)]
    report += fmt % tuple(values)
    return report
