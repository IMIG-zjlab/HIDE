import numpy as np

# dice value
def dice_coef(pred_label, gt_label):
    # list of classes
    c_list = np.unique(gt_label)

    dice_c = []
    for c in range(1,len(c_list)): # dice not for bg
        # intersection
        ints = np.sum(((pred_label == c_list[c]) * 1) * ((gt_label == c_list[c]) * 1))
        # sum
        sums = np.sum(((pred_label == c_list[c]) * 1) + ((gt_label == c_list[c]) * 1)) + 0.0001
        dice_c.append((2.0 * ints) / sums)

    return dice_c


# cal dice
def calDSC(pred_label, gt_label):
    ints = np.sum(((pred_label == 1) * 1) * ((gt_label == 1) * 1))
    sums = np.sum(((pred_label == 1) * 1) + ((gt_label == 1) * 1)) + 0.0001
    dice_value = (2.0 * ints) / sums

    return dice_value




