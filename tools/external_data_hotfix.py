import os
from tqdm import tqdm
from multiprocessing.pool import Pool
import pandas as pd
import numpy as np
from PIL import Image


def hotfix(pid, img_list, img_root):

    for img_name in tqdm(img_list, postfix=pid):
        for color in ['red', 'green', 'blue', 'yellow']:
            img_path = os.path.join(img_root, img_name+'_'+color+'.png')
            np_img = np.array(Image.open(img_path))
            np_img = (np_img * 0.75).astype(np.uint8)
            img = Image.fromarray(np_img)
            img.save(img_path, 'PNG')


if __name__ == '__main__':
    csv_path = "/unsullied/sharefs/ouxiaoxuan/isilon/kaggle/HPAv18RBGY_wodpl.csv"
    img_root = "/unsullied/sharefs/ouxiaoxuan/isilon/kaggle/train"
    img_list = pd.read_csv(csv_path)['Id'].tolist()
    process_num = 24
    list_len = len(img_list)
    p = Pool(process_num)
    for i in range(process_num):
        start = int(i * list_len / process_num)
        end = int((i + 1) * list_len / process_num)
        process_images = img_list[start:end]
        p.apply_async(
            hotfix, args=(str(i), process_images, img_root)
        )
    print('Waiting for all subprocesses done...')
    p.close()
    p.join()
    print('All subprocesses done.')
