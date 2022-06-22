import json
import h5py
import cv2
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import mat73

parser = argparse.ArgumentParser(
    description='Generate keyshots, keyframes and score bar.')
parser.add_argument('--h5_path', type=str,
                    help='path to hdf5 file that contains information of a dataset.', default='tvsum.h5')
parser.add_argument('-j', '--json_path', type=str,
                    help='path to json file that stores pred score output by model, it should be saved in score_dir.', default='score_dir_tvsum/epoch-4.json')
parser.add_argument('-r', '--data_root', type=str,
                    help='path to directory of original dataset.', default='ydata-tvsum50-v1_1')
parser.add_argument('-s', '--save_dir', type=str,
                    help='path to directory where generating results should be saved.', default='Results_tvsum')
parser.add_argument('-d', '--dataset', type=str,
                    help='which dataset videos you want to summarize?', default='tvsum')

args = parser.parse_args()
h5_path = args.h5_path
json_path = args.json_path
data_root = args.data_root
save_dir = args.save_dir
dataset = args.dataset.lower()

if dataset == 'tvsum':
    video_dir = os.path.join(data_root, 'ydata-tvsum50-video', 'video')
    matlab_path = os.path.join(
        data_root, 'ydata-tvsum50-matlab', 'matlab', 'ydata-tvsum50.mat')
    d = mat73.loadmat(matlab_path)
    map_dict = {}
    for i in range(len(d['tvsum50']['video'])):
        map_dict[i+1] = d['tvsum50']['video'][i]
        
elif dataset == 'summe':
    video_dir = os.path.join(data_root, 'videos')
    matlab_path = os.path.join(data_root, 'GT')
    map_dict = {}
    i = 0
    for filename in os.listdir(matlab_path):
        map_dict[i+1] = filename[:-4]
        i += 1
f_data = h5py.File(h5_path)
with open(json_path) as f:
    json_dict = json.load(f)
    ids = json_dict.keys()


def get_keys(id):
    video_info = f_data['video_' + id]
    video_path = os.path.join(video_dir, map_dict[int(id)]+'.mp4')
    cps = video_info['change_points'][()]
    pred_score = json_dict[id]['pred_score']
    pred_selected = json_dict[id]['pred_selected']

    video = cv2.VideoCapture(video_path)
    frames = []
    success, frame = video.read()
    while success:
        frames.append(frame)
        success, frame = video.read()
    frames = np.array(frames)
    keyshots = []
    for sel in pred_selected:
        for i in range(cps[sel][0], cps[sel][1]):
            keyshots.append(frames[i])
    keyshots = np.array(keyshots)

    write_path = os.path.join(save_dir, id, 'summary.avi')
    video_writer = cv2.VideoWriter(
        write_path, cv2.VideoWriter_fourcc(*'XVID'), 24, keyshots.shape[2:0:-1])
    for frame in keyshots:
        video_writer.write(frame)
    video_writer.release()

    keyframe_idx = [np.argmax(
        pred_score[cps[sel][0]: cps[sel][1]]) + cps[sel][0] for sel in pred_selected]
    keyframes = frames[keyframe_idx]

    keyframe_dir = os.path.join(save_dir, id, 'keyframes')
    os.mkdir(keyframe_dir)
    for i, img in enumerate(keyframes):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        plt.axis('off')
        plt.imshow(img)
        plt.savefig(os.path.join(keyframe_dir, '{}.jpg'.format(i)))


def gen_summary():
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    for id in ids:
        os.mkdir(os.path.join(save_dir, id))
        get_keys(id)


if __name__ == '__main__':
    plt.switch_backend('agg')
    gen_summary()


f_data.close()
