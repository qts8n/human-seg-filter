import numpy as np


def human_seg_tracking(pre_gray, cur_gray, prev_cfd, dl_weights, disflow):
    check_thres = 8
    h, w = pre_gray.shape[:2]
    track_cfd = np.zeros_like(prev_cfd)
    is_track = np.zeros_like(pre_gray)
    flow_fw = disflow.calc(pre_gray, cur_gray, None)
    flow_bw = disflow.calc(cur_gray, pre_gray, None)
    flow_fw = np.round(flow_fw).astype(np.int_)
    flow_bw = np.round(flow_bw).astype(np.int_)
    y_list = np.array(range(h))
    x_list = np.array(range(w))
    yv, xv = np.meshgrid(y_list, x_list)
    yv, xv = yv.T, xv.T
    cur_x = xv + flow_fw[:, :, 0]
    cur_y = yv + flow_fw[:, :, 1]

    # 超出边界不跟踪
    not_track = (cur_x < 0) + (cur_x >= w) + (cur_y < 0) + (cur_y >= h)
    flow_bw[~not_track] = flow_bw[cur_y[~not_track], cur_x[~not_track]]
    not_track += (np.square(flow_fw[:, :, 0] + flow_bw[:, :, 0]) +
                  np.square(flow_fw[:, :, 1] + flow_bw[:, :, 1])) >= check_thres
    track_cfd[cur_y[~not_track], cur_x[~not_track]] = prev_cfd[~not_track]

    is_track[cur_y[~not_track], cur_x[~not_track]] = 1

    not_flow = np.all(np.abs(flow_fw) == 0,
                      axis=-1) * np.all(np.abs(flow_bw) == 0, axis=-1)
    dl_weights[cur_y[not_flow], cur_x[not_flow]] = 0.05
    return track_cfd, is_track, dl_weights


def human_seg_track_fuse(track_cfd, dl_cfd, dl_weights, is_track):
    fusion_cfd = dl_cfd.copy()
    is_track = is_track.astype(np.bool_)
    fusion_cfd[is_track] = dl_weights[is_track] * dl_cfd[is_track] + (
        1 - dl_weights[is_track]) * track_cfd[is_track]
    index_certain = ((dl_cfd > 0.9) + (dl_cfd < 0.1)) * is_track
    index_less01 = (dl_weights < 0.1) * index_certain
    fusion_cfd[index_less01] = 0.3 * dl_cfd[index_less01] + 0.7 * track_cfd[
        index_less01]
    index_larger09 = (dl_weights >= 0.1) * index_certain
    fusion_cfd[index_larger09] = 0.4 * dl_cfd[index_larger09] + 0.6 * track_cfd[
        index_larger09]
    return fusion_cfd


def threshold_mask(img, thresh_bg, thresh_fg):
    dst = (img / 255.0 - thresh_bg) / (thresh_fg - thresh_bg)
    dst[np.where(dst > 1)] = 1
    dst[np.where(dst < 0)] = 0
    return dst.astype(np.float32)


def optic_flow_process(cur_gray, scoremap, prev_gray, pre_cfd, disflow, is_init):
    h, w = scoremap.shape
    cur_cfd = scoremap.copy()

    if is_init:
        if h <= 64 or w <= 64:
            disflow.setFinestScale(1)
        elif h <= 160 or w <= 160:
            disflow.setFinestScale(2)
        else:
            disflow.setFinestScale(3)
        fusion_cfd = cur_cfd
    else:
        weights = np.ones((h, w), np.float32) * 0.3
        track_cfd, is_track, weights = human_seg_tracking(
            prev_gray, cur_gray, pre_cfd, weights, disflow)
        fusion_cfd = human_seg_track_fuse(track_cfd, cur_cfd, weights, is_track)

    return fusion_cfd
