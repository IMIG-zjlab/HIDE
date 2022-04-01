import os
from skimage.io import imread, imsave
import skimage
from seg_evals import *

def cal_metrics_png(seg_path, gt_path, metric_file_name, metric_dir):
    gt_labels = []
    pred_labels = []
    dice_sum = 0
    case_count = 0
    sliceIdxs = []
    seg_list = os.listdir(seg_path)
    seg_list.sort()
    pre_case = seg_list[0][4:7]
    f = open(os.path.join(metric_dir, metric_file_name), 'w')
    for file in seg_list:
        if file.endswith(".png"):
            cur_case = file[4:7]
            if cur_case != pre_case:
                arr_pred_labels = np.array(pred_labels)
                arr_gt_labels = np.array(gt_labels)
                arr_pred_labels[arr_pred_labels >= 0.4] = 1
                arr_pred_labels[arr_pred_labels < 0.4] = 0
                k_dice_c = calDSC(arr_pred_labels, arr_gt_labels)
                print(pre_case, ' ', k_dice_c)
                f.write('\n {caseNum}, DICE: {dice_v}'
                        .format(caseNum=pre_case, dice_v=str(k_dice_c)))

                dice_sum += k_dice_c
                case_count += 1
                #reset
                pred_labels = []
                gt_labels = []
                sliceIdxs = []
                pre_case = cur_case
            pred = imread(os.path.join(seg_path, file))
            gt = imread(os.path.join(gt_path, file))
            sliceIdxs.append(int(file.split('_')[1][-3:]))
            pred = skimage.img_as_float(pred)
            gt = skimage.img_as_float(gt)
            pred_labels.append(pred)
            gt_labels.append(gt)

    arr_pred_labels = np.array(pred_labels)
    arr_gt_labels = np.array(gt_labels)
    arr_pred_labels[arr_pred_labels >= 0.4] = 1
    arr_pred_labels[arr_pred_labels < 0.4] = 0
    k_dice_c = calDSC(arr_pred_labels, arr_gt_labels)
    print(pre_case, ' ', k_dice_c)
    f.write('\n {caseNum}, DICE: {dice_v}'
            .format(caseNum=pre_case, dice_v=str(k_dice_c)))
    dice_sum += k_dice_c
    case_count += 1
    mean_dice = dice_sum/case_count
    print(mean_dice)
    f.write('\n mean_DICE: {mean_dice}'.format(mean_dice=str(mean_dice)))
    f.close()
    print('total number of results: ', len(seg_list))


if __name__ == '__main__':
    pred_path = "/data/quantised_results/"
    gt_path = "/data/label/test/"
    metric_file_name = pred_path.split('/')[-4]+'.txt'
    cal_metrics_png(pred_path, gt_path, metric_file_name)