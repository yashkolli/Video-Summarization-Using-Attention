import numpy as np
from PIL import Image
import math
from torchvision import transforms, models
import torch
from torch import nn
from KTS.cpd_auto import cpd_auto


def knapsack(v, w, W):
    r = len(v) + 1
    c = W + 1

    v = np.r_[[0], v]
    w = np.r_[[0], w]

    dp = [[0 for i in range(c)] for j in range(r)]

    for i in range(1, r):
        for j in range(1, c):
            if w[i] <= j:
                dp[i][j] = max(v[i] + dp[i-1][j-w[i]], dp[i-1][j])
            else:
                dp[i][j] = dp[i-1][j]

    chosen = []
    i = r - 1
    j = c - 1
    while i > 0 and j > 0:
        if dp[i][j] != dp[i-1][j]:
            chosen.append(i-1)
            j = j - w[i]
            i = i - 1
        else:
            i = i - 1

    return dp[r-1][c-1], chosen


class Rescale:
    def __init__(self, *output_size):
        self.output_size = output_size

    def __call__(self, image):
        new_h, new_w = self.output_size
        new_h, new_w = int(new_h), int(new_w)
        img = image.resize((new_w, new_h), resample=Image.BILINEAR)
        return img


def extract_features(frame):
    return fea_net(transform(Image.fromarray(frame)).cuda().unsqueeze(0)).squeeze().detach().cpu()


def get_change_points(video_feat, n_frame, fps):
    n = n_frame / fps
    m = int(math.ceil(n/2.0))
    K = np.dot(video_feat, video_feat.T)
    change_points, _ = cpd_auto(K, m, 1)
    change_points = np.concatenate(([0], change_points, [n_frame-1]))

    temp_change_points = []
    for idx in range(len(change_points)-1):
        segment = [change_points[idx], change_points[idx+1]-1]
        if idx == len(change_points)-2:
            segment = [change_points[idx], change_points[idx+1]]

        temp_change_points.append(segment)
    change_points = np.array(list(temp_change_points))

    temp_n_frame_per_seg = []
    for change_points_idx in range(len(change_points)):
        n_frame = change_points[change_points_idx][1] - \
            change_points[change_points_idx][0]
        temp_n_frame_per_seg.append(n_frame)
    n_frame_per_seg = np.array(list(temp_n_frame_per_seg))

    return change_points, n_frame_per_seg


def sum_fscore(overlap_arr, true_sum_arr, oracle_sum):
    fscores = []
    for overlap, true_sum in zip(overlap_arr, true_sum_arr):
        precision = overlap / (oracle_sum + 1e-8)
        recall = overlap / (true_sum + 1e-8)
        if precision == 0 and recall == 0:
            fscore = 0
        else:
            fscore = 2 * precision * recall / (precision + recall)
        fscores.append(fscore)
    return sum(fscores) / len(fscores)


def get_oracle_summary(user_summary):
    n_user, n_frame = user_summary.shape
    oracle_summary = np.zeros(n_frame)
    overlap_arr = np.zeros(n_user)
    oracle_sum = 0
    true_sum_arr = user_summary.sum(axis=1)
    priority_idx = np.argsort(-user_summary.sum(axis=0))
    best_fscore = 0
    for idx in priority_idx:
        oracle_sum += 1
        for usr_i in range(n_user):
            overlap_arr[usr_i] += user_summary[usr_i][idx]
        cur_fscore = sum_fscore(overlap_arr, true_sum_arr, oracle_sum)
        if cur_fscore > best_fscore:
            best_fscore = cur_fscore
            oracle_summary[idx] = 1
        else:
            break
    print('Overlap: '+str(overlap_arr))
    print('True summary n_key: '+str(true_sum_arr))
    print('Oracle smmary n_key: '+str(oracle_sum))
    print('Final F-score: '+str(best_fscore))
    return oracle_summary


transform = transforms.Compose([
    Rescale(224, 224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

net = models.googlenet(pretrained=True).float().cuda()
net.eval()
fea_net = nn.Sequential(*list(net.children())[:-2])
