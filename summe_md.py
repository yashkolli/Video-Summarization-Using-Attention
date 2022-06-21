import os
import scipy.io
import cv2
import h5py
from utils import *


def gen_user_summary_summe(filename):
    d = scipy.io.loadmat(os.path.join(matlab_path, filename))
    vidlen = int(d['nFrames'])
    n = len(d['segments'][0])
    user_summary = np.zeros((n, vidlen))
    for i in range(vidlen):
        for j in range(n):
            if d['user_score'][i][j] > 0:
                user_summary[j][i] = 1
    return user_summary


data_root = ''  # path to the original dataset
video_dir = os.path.join(data_root, 'videos')
matlab_path = os.path.join(data_root, 'GT')
i = 0
with h5py.File("summe.h5", "w") as f:
    for filename in os.listdir(matlab_path):
        video_path = os.path.join(video_dir, filename[:-4] + '.mp4')
        group_name = 'video_' + str(i + 1)
        grp = f.create_group(group_name)
        video = cv2.VideoCapture(video_path)
        print('Processing video ' + str(i+1))
        length = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = video.get(cv2.CAP_PROP_FPS)
        ratio = length//320
        l = (length - ratio*320)//2
        fea_for_train = []
        fea = []
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

        user_summary = gen_user_summary_summe(filename)

        change_points, n_frame_per_seg = get_change_points(fea, length, fps)

        oracle_summary = get_oracle_summary(user_summary)
        label = [oracle_summary[picked_frame] for picked_frame in picks]

        grp['feature'] = fea_for_train
        grp['label'] = label
        grp['length'] = length
        grp['change_points'] = change_points
        grp['n_frame_per_seg'] = n_frame_per_seg
        grp['picks'] = np.array(list(picks))
        grp['user_summary'] = user_summary

        i += 1
