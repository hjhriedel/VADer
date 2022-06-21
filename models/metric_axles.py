#%%
from dis import dis
import scipy.signal as sig
import scipy.stats as stats
import numpy as np
import tensorflow as tf

def error_axle_positions(prediction, true_values, threshold=20, distance=20):

    # defining number and position of true axles:
    idx_axle_true = np.where(true_values == 1)[0]
    num_axle_true = np.count_nonzero(true_values)

    # defining number and position of predicted axles:
    idx_axle_pred, _ = sig.find_peaks(prediction, height=0.25, prominence=0.15, distance=distance)
    num_axle_pred = idx_axle_pred.shape[0]

    num_axle_correct, num_axle_false = 0, 0
    false_samples = np.zeros_like(idx_axle_true)
    axle_correct = np.zeros_like(true_values)
    
    for i in range(num_axle_true):
        if idx_axle_pred.any():
            idx_min_distance = np.argmin(abs(idx_axle_pred - idx_axle_true[i]))
            if min(abs(idx_axle_pred - idx_axle_true[i])) <= threshold:
                false_samples[i-num_axle_false] = idx_axle_true[i] - idx_axle_pred[idx_min_distance]
                num_axle_correct += 1
                idx_axle_pred = np.delete(idx_axle_pred, idx_min_distance)
                axle_correct[idx_axle_true[i]] = 1
            else:
                false_samples = np.delete(false_samples, i-num_axle_false, axis=0)
                num_axle_false += 1
        else:
            false_samples = np.delete(false_samples, i - num_axle_false, axis=0)
            num_axle_false += 1

    return num_axle_true, num_axle_pred, num_axle_correct, false_samples


def error_statistics(prediction, true_values):
    num_axle_true, num_axle_pred, num_axle_correct, false_samples = 0, 0, 0, []
    for pred, tv in zip(prediction, true_values):
        _num_axle_true, _num_axle_pred, _num_axle_correct, _false_samples = error_axle_positions(pred, tv)
        num_axle_true += _num_axle_true
        num_axle_pred += _num_axle_pred
        num_axle_correct += _num_axle_correct
        false_samples.append(_false_samples)
    false_samples = np.hstack(false_samples)
    
    avg_num_neg, avg_num_pos, avg_num_null = 0, 0, 0

    if false_samples.any():
        mean_false_samples = np.sum(abs(false_samples)) / num_axle_correct if num_axle_correct > 0 else 999
        max_false_samples = np.max(abs(false_samples))
        # min_false_samples = np.min(abs(false_samples))
        var_false_samples = np.var(false_samples)
        std_false_samples = np.std(false_samples)
    else:        
        mean_false_samples, max_false_samples, var_false_samples, std_false_samples = 999, 999, 999, 999

    avg_num_neg = np.sum(false_samples<0) / num_axle_correct
    avg_num_pos = np.sum(false_samples>0) / num_axle_correct
    avg_num_null = np.sum(false_samples==0) / num_axle_correct
    
    recall = num_axle_correct / num_axle_true
    precision = num_axle_correct / num_axle_pred if num_axle_pred > 0 else 0

    return precision, max_false_samples, mean_false_samples, var_false_samples, std_false_samples, avg_num_neg, avg_num_null, avg_num_pos, recall



def _max_fs(prediction, true_values): 
    false_samples = []
    for pred, tv in zip(prediction, true_values):
        _, _, _, _false_samples = error_axle_positions(pred, tv)
        false_samples.append(_false_samples)
    false_samples = np.hstack(false_samples)
    max_fs = np.max(abs(false_samples)) if false_samples.any() else 999
    return max_fs

def max_fs(prediction, true_values):
    return tf.py_function(_max_fs, [true_values, prediction], [tf.float32])


def _mean_fs(prediction, true_values): 
    false_samples, num_axle_correct = [], 0
    for pred, tv in zip(prediction, true_values):
        _, _, _num_axle_correct, _false_samples = error_axle_positions(pred, tv)
        num_axle_correct += _num_axle_correct
        false_samples.append(_false_samples)
    false_samples = np.hstack(false_samples)
    if false_samples.any() and num_axle_correct > 0:
        mean_fs = np.sum(abs(false_samples)) / num_axle_correct
    else:
        mean_fs = 0
    return mean_fs

def mean_fs(prediction, true_values):
    return tf.py_function(_mean_fs, [true_values, prediction], [tf.float32])


def _var_fs(prediction, true_values): 
    false_samples = []
    for pred, tv in zip(prediction, true_values):
        _, _, _, _false_samples = error_axle_positions(pred, tv)
        false_samples.append(_false_samples)
    false_samples = np.hstack(false_samples)
    var_fs = np.var(false_samples) if false_samples.any() else 0
    return var_fs
    
def var_fs(prediction, true_values):
    return tf.py_function(_var_fs, [true_values, prediction], [tf.float32])


def _std_fs(prediction, true_values): 
    false_samples = []
    for pred, tv in zip(prediction, true_values):
        _, _, _, _false_samples = error_axle_positions(pred, tv)
        false_samples.append(_false_samples)
    false_samples = np.hstack(false_samples)
    std_fs = np.std(false_samples) if false_samples.any() else 0
    return std_fs

def std_fs(prediction, true_values):
    return tf.py_function(_std_fs, [true_values, prediction], [tf.float32])


def _percent_neg(prediction, true_values): 
    num_axle_correct, false_samples = 0, []
    for pred, tv in zip(prediction, true_values):
        _, _, _num_axle_correct, _false_samples = error_axle_positions(pred, tv)
        num_axle_correct += _num_axle_correct
        false_samples.append(_false_samples)
    false_samples = np.hstack(false_samples)
    percent_neg = np.sum(false_samples < 0) / num_axle_correct if num_axle_correct > 0 else 0
    return percent_neg

def percent_neg(prediction, true_values):
    return tf.py_function(_percent_neg, [true_values, prediction], [tf.float32])


def _percent_null(prediction, true_values): 
    num_axle_correct, false_samples = 0, []
    for pred, tv in zip(prediction, true_values):
        _, _, _num_axle_correct, _false_samples = error_axle_positions(pred, tv)
        num_axle_correct += _num_axle_correct
        false_samples.append(_false_samples)
    false_samples = np.hstack(false_samples)
    percent_null = np.sum(false_samples == 0) / num_axle_correct if num_axle_correct > 0 else 0
    return percent_null

def percent_null(prediction, true_values):
    return tf.py_function(_percent_null, [true_values, prediction], [tf.float32])


def _percent_pos(prediction, true_values): 
    num_axle_correct, false_samples = 0, []
    for pred, tv in zip(prediction, true_values):
        _, _, _num_axle_correct, _false_samples = error_axle_positions(pred, tv)
        num_axle_correct += _num_axle_correct
        false_samples.append(_false_samples)
    false_samples = np.hstack(false_samples)
    percent_pos = np.sum(false_samples>0) / num_axle_correct if num_axle_correct > 0 else 0
    return percent_pos

def percent_pos(prediction, true_values):
    return tf.py_function(_percent_pos, [true_values, prediction], [tf.float32])


def _avg_pos(prediction, true_values): 
    num_axle_correct, false_samples = 0, []
    for pred, tv in zip(prediction, true_values):
        _, _, _num_axle_correct, _false_samples = error_axle_positions(pred, tv)
        num_axle_correct += _num_axle_correct
        false_samples.append(_false_samples)
    false_samples = np.hstack(false_samples)
    criteria = false_samples > 0
    avg_pos = np.sum(false_samples[criteria]) / np.sum(criteria) if np.sum(criteria) > 0 else 0
    return avg_pos

def avg_pos(prediction, true_values):
    return tf.py_function(_avg_pos, [true_values, prediction], [tf.float32])


def _avg_neg(prediction, true_values): 
    num_axle_correct, false_samples = 0, []
    for pred, tv in zip(prediction, true_values):
        _, _, _num_axle_correct, _false_samples = error_axle_positions(pred, tv)
        num_axle_correct += _num_axle_correct
        false_samples.append(_false_samples)
    false_samples = np.hstack(false_samples)
    criteria = false_samples < 0
    avg_neg = np.sum(false_samples[criteria]) / np.sum(criteria) if np.sum(criteria) > 0 else 0
    return avg_neg

def avg_neg(prediction, true_values):
    return tf.py_function(_avg_neg, [true_values, prediction], [tf.float32])


def _f1(prediction, true_values): 
    num_axle_pred, num_axle_correct, num_axle_true = 0, 0, 0
    for pred, tv in zip(prediction, true_values):
        _num_axle_true, _num_axle_pred, _num_axle_correct, _ = error_axle_positions(pred, tv)
        num_axle_true += _num_axle_true
        num_axle_correct += _num_axle_correct
        num_axle_pred += _num_axle_pred
    if num_axle_correct == 0 or num_axle_pred == 0:
        return 0
    else:
        precision = num_axle_correct / num_axle_pred
        recall = num_axle_correct / num_axle_true
        return 2 * (precision * recall) / (precision + recall)

def f1(prediction, true_values):
    return tf.py_function(_f1, [true_values, prediction], [tf.float32])


def _recall(prediction, true_values): 
    num_axle_true, num_axle_correct = 0, 0
    for pred, tv in zip(prediction, true_values):
        _num_axle_true, _, _num_axle_correct, _ = error_axle_positions(pred, tv)
        num_axle_true += _num_axle_true
        num_axle_correct += _num_axle_correct
    recall = num_axle_correct / num_axle_true
    return recall

def recall(prediction, true_values):
    return tf.py_function(_recall, [true_values, prediction], [tf.float32])

def _precision(prediction, true_values): 
    num_axle_pred, num_axle_correct = 0, 0
    for pred, tv in zip(prediction, true_values):
        _, _num_axle_pred, _num_axle_correct, _ = error_axle_positions(pred, tv)
        num_axle_pred += _num_axle_pred
        num_axle_correct += _num_axle_correct
    precision = num_axle_correct / num_axle_pred if num_axle_pred > 0 else 0
    return precision

def precision(prediction, true_values):
    return tf.py_function(_precision, [true_values, prediction], [tf.float32])


def _recall3(prediction, true_values): 
    num_axle_true, num_axle_correct = 0, 0
    for pred, tv in zip(prediction, true_values):
        _num_axle_true, _, _num_axle_correct, _ = error_axle_positions(pred, tv, threshold=3)
        num_axle_true += _num_axle_true
        num_axle_correct += _num_axle_correct
    recall = num_axle_correct / num_axle_true
    return recall

def recall3(prediction, true_values):
    return tf.py_function(_recall3, [true_values, prediction], [tf.float32])

def _precision3(prediction, true_values): 
    num_axle_pred, num_axle_correct = 0, 0
    for pred, tv in zip(prediction, true_values):
        _, _num_axle_pred, _num_axle_correct, _ = error_axle_positions(pred, tv, threshold=3)
        num_axle_pred += _num_axle_pred
        num_axle_correct += _num_axle_correct
    precision = num_axle_correct / num_axle_pred if num_axle_pred > 0 else 0
    return precision

def precision3(prediction, true_values):
    return tf.py_function(_precision3, [true_values, prediction], [tf.float32])

def _f1_3(prediction, true_values): 
    num_axle_pred, num_axle_correct, num_axle_true = 0, 0, 0
    for pred, tv in zip(prediction, true_values):
        _num_axle_true, _num_axle_pred, _num_axle_correct, _ = error_axle_positions(pred, tv, threshold=3)
        num_axle_true += _num_axle_true
        num_axle_correct += _num_axle_correct
        num_axle_pred += _num_axle_pred
    if num_axle_correct == 0 or num_axle_pred == 0:
        return 0
    else:
        precision = num_axle_correct / num_axle_pred
        recall = num_axle_correct / num_axle_true
        return 2 * (precision * recall) / (precision + recall)

def f1_3(prediction, true_values):
    return tf.py_function(_f1_3, [true_values, prediction], [tf.float32])


def _recall7(prediction, true_values): 
    num_axle_true, num_axle_correct = 0, 0
    for pred, tv in zip(prediction, true_values):
        _num_axle_true, _, _num_axle_correct, _ = error_axle_positions(pred, tv, threshold=7)
        num_axle_true += _num_axle_true
        num_axle_correct += _num_axle_correct
    recall = num_axle_correct / num_axle_true
    return recall

def recall7(prediction, true_values):
    return tf.py_function(_recall7, [true_values, prediction], [tf.float32])

def _precision7(prediction, true_values): 
    num_axle_pred, num_axle_correct = 0, 0
    for pred, tv in zip(prediction, true_values):
        _, _num_axle_pred, _num_axle_correct, _ = error_axle_positions(pred, tv, threshold=7)
        num_axle_pred += _num_axle_pred
        num_axle_correct += _num_axle_correct
    precision = num_axle_correct / num_axle_pred if num_axle_pred > 0 else 0
    return precision

def precision7(prediction, true_values):
    return tf.py_function(_precision7, [true_values, prediction], [tf.float32])

def _f1_7(prediction, true_values): 
    num_axle_pred, num_axle_correct, num_axle_true = 0, 0, 0
    for pred, tv in zip(prediction, true_values):
        _num_axle_true, _num_axle_pred, _num_axle_correct, _ = error_axle_positions(pred, tv, threshold=7)
        num_axle_true += _num_axle_true
        num_axle_correct += _num_axle_correct
        num_axle_pred += _num_axle_pred
    if num_axle_correct == 0 or num_axle_pred == 0:
        return 0
    else:
        precision = num_axle_correct / num_axle_pred
        recall = num_axle_correct / num_axle_true
        return 2 * (precision * recall) / (precision + recall)

def f1_7(prediction, true_values):
    return tf.py_function(_f1_7, [true_values, prediction], [tf.float32])
