import os
from tqdm import tqdm
from multiprocessing.pool import Pool
import pandas as pd
import json
from PIL import Image

import imagehash


def calc_hash(pid, paths, hashfunc, postfix):
    d = {}
    for img_path in tqdm(paths, postfix=pid):
        img_name = img_path.split('/')[-1]
        d[img_name] = hashfunc(Image.open(img_path))

    with open('%d_%s.json' % (pid, postfix), 'w') as f:
        json.dump(d, f)


def main(paths, hash_func, postfix):
    process_num = 24
    list_len = len(paths)
    p = Pool(process_num)
    for i in range(process_num):
        start = int(i * list_len / process_num)
        end = int((i + 1) * list_len / process_num)
        process_paths = paths[start:end]
        p.apply_async(
            calc_hash, args=(str(i), process_paths, hash_func, postfix)
        )
    print('Waiting for all subprocesses done...')
    p.close()
    p.join()
    print('All subprocesses done.')


def merge_json(postfix):
    d = {}
    json_files = [part for part in os.listdir('.') if part.endswith("%s.json" % postfix)]
    print("merging..")
    for json_file in json_files:
        with open(json_file, 'r') as f:
            data = json.load(f)
            d.update(data)

    with open('merged_%s.json' % postfix, 'w') as f:
        json.dump(d, f)


if __name__ == '__main__':
    train_csv_path = "/unsullied/sharefs/ouxiaoxuan/isilon/kaggle/train_split.csv"
    valid_csv_path = "/unsullied/sharefs/ouxiaoxuan/isilon/kaggle/valid_split.csv"
    test_csv_path = "/unsullied/sharefs/ouxiaoxuan/isilon/kaggle/sample_submission.csv"
    train_df = pd.read_csv(train_csv_path)
    valid_df = pd.read_csv(valid_csv_path)

    train_df_all = pd.concat((train_df, valid_df))
    test_df = pd.read_csv(test_csv_path)

    train_img_root = "/unsullied/sharefs/ouxiaoxuan/isilon/kaggle/train"
    test_img_root = "/unsullied/sharefs/ouxiaoxuan/isilon/kaggle/test"

    def get_paths(df, root):
        paths = []
        for img_id in df['Id'].tolist():
            for color in ['red', 'green', 'blue', 'yellow']:
                img_name = img_id + "_" + color + ".png"
                paths.append(os.path.join(root, img_name))
        return paths

    train_paths = get_paths(train_df_all, train_img_root)
    test_paths = get_paths(test_df, test_img_root)

    hash_dict = {
        'ahash': imagehash.average_hash,
        'phash': imagehash.phash,
        'dhash': imagehash.dhash
    }

    main(train_paths, imagehash.average_hash, 'train_ahash')
    merge_json('train_ahash')

    main(test_paths, imagehash.average_hash, 'test_ahash')
    merge_json('test_ahash')