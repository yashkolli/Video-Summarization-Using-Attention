import os
import mat73
import cv2
import h5py
from tqdm import tqdm
from utils import *


def gen_user_summary_tvsum(index, vidlen, cps, weight):
    n = len(d['tvsum50']['user_anno'][index][0])
    user_summary = np.zeros((n, vidlen))
    user_anno = np.zeros((n, vidlen))
    for i in range(vidlen):
        for j in range(n):
            user_anno[j][i] = d['tvsum50']['user_anno'][index][i][j]
    for i in range(n):
        gt_value = np.array([user_anno[cp[0]:cp[1]].mean() for cp in cps])
        _, selected = knapsack(gt_value, weight, int(0.15 * vidlen))
        selected = selected[::-1]
        key_labels = np.zeros((vidlen,))
        for j in selected:
            key_labels[cps[j][0]:cps[j][1]] = 1
        user_summary[i, ] = key_labels
    return user_summary


data_root = ''  # path to original dataset
video_dir = os.path.join(data_root, 'ydata-tvsum50-video', 'video')
matlab_path = os.path.join(
    data_root, 'ydata-tvsum50-matlab', 'matlab', 'ydata-tvsum50.mat')
d = mat73.loadmat(matlab_path)
map_dict = {}
for i in range(len(d['tvsum50']['video'])):
    map_dict[i+1] = d['tvsum50']['video'][i]
with h5py.File("tvsum.h5", "w") as f:
    for i in tqdm(range(len(d['tvsum50']['video'])), desc="Video", ncols=80, leave=False):
        video_path = os.path.join(video_dir, map_dict[i+1]+'.mp4')
        group_name = 'video_' + str(i + 1)
        grp = f.create_group(group_name)
        video = cv2.VideoCapture(video_path)
        tqdm.write('Processing video ' + str(i+1))
        length = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = video.get(cv2.CAP_PROP_FPS)
        ratio = length//320
        l = (length - ratio*320)//2
        fea = []
        fea_for_train = []
        picks = []
        j = 0
        pc = 0
        success, frame = video.read()
        while success:
            frame_feature = extract_features(frame)

            if (j+1) > l and pc < 320:
                if (j+1-l) % ratio == 0:
                    picks.append(j)
                    pc += 1

                    fea_for_train.append(frame_feature)

            fea.append(frame_feature)
            j += 1

            success, frame = video.read()

        video.release()

        fea = torch.stack(fea).numpy()
        fea_for_train = torch.stack(fea_for_train).numpy()

        change_points, n_frame_per_seg = get_change_points(fea, length, fps)

        user_summary = gen_user_summary_tvsum(
            i, length, change_points, n_frame_per_seg)

        oracle_summary = get_oracle_summary(user_summary)
        label = [oracle_summary[picked_frame] for picked_frame in picks]

        grp['feature'] = fea_for_train
        grp['label'] = label
        grp['length'] = length
        grp['change_points'] = change_points
        grp['n_frame_per_seg'] = n_frame_per_seg
        grp['picks'] = np.array(list(picks))
        grp['user_summary'] = user_summary
