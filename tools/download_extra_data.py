import os
import errno
from multiprocessing.pool import Pool
from tqdm import tqdm
import requests
import pandas as pd
import numpy as np
from PIL import Image


def download(pid, image_list, base_url, save_dir, image_size=(512, 512)):
    colors = ['red', 'green', 'blue', 'yellow']
    for i in tqdm(image_list, postfix=pid):
        img_id = i.split('_', 1)
        for color in colors:
            img_path = img_id[0] + '/' + img_id[1] + '_' + color + '.jpg'
            img_name = i + '_' + color + '.png'
            img_url = base_url + img_path
            save_path = os.path.join(save_dir, img_name)
            if not os.path.exists(save_path):

                # Get the raw response from the url
                r = requests.get(img_url, allow_redirects=True, stream=True)
                r.raw.decode_content = True

                # Use PIL to resize the image and to convert it to L
                # (8-bit pixels, black and white)
                im = Image.open(r.raw)
                im = im.resize(image_size, Image.LANCZOS)
                np_im = np.array(im)
                if color == 'red':
                    r_channel = np_im[:, :, 0]
                    im = Image.fromarray(r_channel.astype(np.uint8))
                elif color == 'green':
                    g_channel = np_im[:, :, 1]
                    im = Image.fromarray(g_channel.astype(np.uint8))
                elif color == 'blue':
                    b_channel = np_im[:, :, 2]
                    im = Image.fromarray(b_channel.astype(np.uint8))
                elif color == 'red':
                    y_channel = 0.5*(np_im[:, :, 0]) + 0.5*(np_im[:, :, 1])
                    im = Image.fromarray(y_channel.astype(np.uint8))
                im.save(save_path, 'PNG')


if __name__ == '__main__':
    # Parameters
    process_num = 24
    image_size = (512, 512)
    url = 'http://v18.proteinatlas.org/images/'
    csv_path = "/unsullied/sharefs/ouxiaoxuan/isilon/kaggle/HPAv18RBGY_wodpl.csv"
    save_dir = "/unsullied/sharefs/ouxiaoxuan/isilon/kaggle/external_data_fixed"

    # Create the directory to save the images in case it doesn't exist
    try:
        os.makedirs(save_dir)
    except OSError as exc:
        if exc.errno != errno.EEXIST:
            raise
        pass

    print('Parent process %s.' % os.getpid())
    img_list = pd.read_csv(csv_path)['Id']
    list_len = len(img_list)
    p = Pool(process_num)
    for i in range(process_num):
        start = int(i * list_len / process_num)
        end = int((i + 1) * list_len / process_num)
        process_images = img_list[start:end]
        p.apply_async(
            download, args=(str(i), process_images, url, save_dir, image_size)
        )
    print('Waiting for all subprocesses done...')
    p.close()
    p.join()
    print('All subprocesses done.')