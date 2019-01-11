import sys
sys.path.append('../')

import os
from dl_backbone.config import cfg
import math
import operator
from functools import reduce
from collections import Counter
import numpy as np
import pandas as pd
from ml_stratifiers import MultilabelStratifiedShuffleSplit


def combine_dataset(root):
    df_train = pd.read_csv(os.path.join(root, "train.csv"))
    df_external = pd.read_csv(os.path.join(root, "HPAv18RGBY_WithoutUncertain_wodpl.csv"))
    df_combined = pd.concat([df_train, df_external])
    df_combined.reset_index(drop=True, inplace=True)
    print("train: %d, external: %d, combined: %d" % (len(df_train), len(df_external), len(df_combined)))
    df_combined['target_vec'] = df_combined['Target'].apply(str2vec)
    return df_combined


def str2vec(s):
    tags = list(map(int, s.split(' ')))
    vec = np.zeros(28)
    vec[tags] = 1
    return vec.tolist()


def train_test_split(df, n_splits=4) -> ([pd.DataFrame], [pd.DataFrame]):
    df_backup = df.copy()
    X = df['Id'].tolist()
    y = df['target_vec'].tolist()
    msss = MultilabelStratifiedShuffleSplit(n_splits=n_splits, test_size=0.1, random_state=42)
    train_dfs, valid_dfs = [], []
    for train_index, valid_index in msss.split(X,y):
        train_df = df_backup.iloc[train_index]
        train_dfs.append(train_df[['Id', 'Target']])
        valid_df = df_backup.iloc[valid_index]
        valid_dfs.append(valid_df[['Id', 'Target']])
    return train_dfs, valid_dfs


def count_distrib(df):
    tag_list = df['Target'].tolist()
    tag_list = reduce(operator.add, map(lambda x: list(map(int, x.split(' '))), tag_list))
    return Counter(tag_list)


def create_class_weight(labels_dict, mu=0.5):
    """
    this is for calculating weighted BCE loss only
    :param labels_dict:
    :param mu:
    :return:
    """
    total = sum(labels_dict.values())
    keys = labels_dict.keys()
    class_weight = dict()
    class_weight_log = dict()

    for key in keys:
        score = total / float(labels_dict[key])
        score_log = math.log(mu * total / float(labels_dict[key]))
        class_weight[key] = round(score, 2) if score > 1.0 else round(1.0, 2)
        class_weight_log[key] = round(score_log, 2) if score_log > 1.0 else round(1.0, 2)

    class_weight_vec = np.zeros(len(class_weight))
    class_weight_log_vec = np.zeros(len(class_weight))
    for k in class_weight:
        class_weight_vec[k] = class_weight[k]
    for k in class_weight_log:
        class_weight_log_vec[k] = class_weight_log[k]
    return class_weight_vec, class_weight_log_vec


def create_sample_weight(df, save_name, mu=1.0):
    """
    assign sample weights for each sample. rare sample have higher weights(linearly)
    refer to
    :param df:
    :param save_name:
    :param mu:
    :return:
    """
    label_list = df['Target'].tolist()
    import pickle
    import operator
    from functools import reduce
    from collections import Counter
    freq_count = dict(Counter(
        reduce(operator.add,
               map(lambda x: list(map(int, x.split(' '))),
                   label_list
                   )
               )
    ))
    total = sum(freq_count.values())
    keys = freq_count.keys()
    assert sorted(list(keys)) == list(range(len(keys)))
    class_weight = dict()
    class_weight_log = dict()
    for key in range(len(keys)):
        score = total / float(freq_count[key])
        score_log = math.log(mu * total / float(freq_count[key]))
        class_weight[key] = round(score, 2) if score > 1.0 else round(1.0, 2)
        class_weight_log[key] = round(score_log, 2) if score_log > 1.0 else round(1.0, 2)

    rareness = [x[0] for x in sorted(freq_count.items(), key=operator.itemgetter(1))]

    weights = []
    sample_labels = list(map(lambda x: list(map(int, x.split(' '))), label_list))
    for labels in sample_labels:
        for rare_label in rareness:
            if rare_label in labels:
                weights.append(class_weight[rare_label])
                break

    assert len(weights) == len(label_list)
    with open(save_name, 'wb') as f:
        pickle.dump(weights, f)
    print("%d weights saved into %s" % (len(label_list), save_name))


def calc_statistics(loader='train'):
    """
    calculate mean and std for data sets
    please close normalize in dataset's transform manually
    :return:
    """
    from dl_backbone.data.build import make_data_loader
    if loader == 'train':
        data_loader = make_data_loader(cfg, cfg.DATASETS.TRAIN, is_train=True)
    elif loader == 'valid':
        data_loader = make_data_loader(cfg, cfg.DATASETS.VALID, is_train=False)
    elif loader == 'test':
        data_loader = make_data_loader(cfg, cfg.DATASETS.TEST, is_train=False)
    else:
        raise KeyError('loader must be specified')
    mean = 0.
    std = 0.
    nb_samples = 0.
    for iteration, (images, targets, indices) in enumerate(data_loader):
        batch_size = images.size(0)
        images = images.view(batch_size, images.size(1), -1)
        mean += images.mean(2).sum(0)
        std += images.std(2).sum(0)
        nb_samples += batch_size
        running_mean = mean / nb_samples
        running_std = std / nb_samples
        print("iter %d running mean: " % iteration, running_mean)
        print("iter %d running std : " % iteration, running_std)


if __name__ == "__main__":
    df_combined = combine_dataset("/unsullied/sharefs/ouxiaoxuan/isilon/kaggle/")
    train_dfs, valid_dfs = train_test_split(df_combined, 4)
    for i in range(4):
        train_df = train_dfs[i]
        train_df.to_csv("train_split_%d.csv"%i, index=False)
        valid_df = valid_dfs[i]
        valid_df.to_csv("valid_split_%d.csv"%i, index=False)

        create_sample_weight(train_df, "train_split_%d_weights.pickle"%i)
    class_weight_vec, class_weight_log_vec = create_class_weight(dict(count_distrib(df_combined)), mu=0.5)