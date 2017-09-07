import numpy as np
import copy
import nibabel as nib

# calculate evaluation metrics for segmentation
def seg_eval_metric(pred_label, gt_label):
    class_n = np.unique(gt_label)
    # dice
    dice_c = dice_n_class(move_img=pred_label, refer_img=gt_label)
    # # conformity
    # conform_c = conform_n_class(move_img=pred_label, refer_img=gt_label)
    # # jaccard
    # jaccard_c = jaccard_n_class(move_img=pred_label, refer_img=gt_label)
    # # precision and recall
    # precision_c, recall_c = precision_recall_n_class(move_img=pred_label, refer_img=gt_label)

    # return dice_c, conform_c, jaccard_c, precision_c, recall_c
    return dice_c

# dice value
def dice_n_class(move_img, refer_img):
    # list of classes
    c_list = np.unique(refer_img)

    dice_c = []
    for c in range(len(c_list)):
        # intersection
        ints = np.sum(((move_img == c_list[c]) * 1) * ((refer_img == c_list[c]) * 1))
        # sum
        sums = np.sum(((move_img == c_list[c]) * 1) + ((refer_img == c_list[c]) * 1)) + 0.0001
        dice_c.append((2.0 * ints) / sums)

    return dice_c


# conformity value
def conform_n_class(move_img, refer_img):
    # list of classes
    c_list = np.unique(refer_img)

    conform_c = []
    for c in range(len(c_list)):
        # intersection
        ints = np.sum(((move_img == c_list[c]) * 1) * ((refer_img == c_list[c]) * 1))
        # sum
        sums = np.sum(((move_img == c_list[c]) * 1) + ((refer_img == c_list[c]) * 1)) + 0.0001
        # dice
        dice_temp = (2.0 * ints) / sums
        # conformity
        conform_temp = (3*dice_temp - 2) / dice_temp

        conform_c.append(conform_temp)

    return conform_c


# Jaccard index
def jaccard_n_class(move_img, refer_img):
    # list of classes
    c_list = np.unique(refer_img)

    jaccard_c = []
    for c in range(len(c_list)):
        move_img_c = (move_img == c_list[c])
        refer_img_c = (refer_img == c_list[c])
        # intersection
        ints = np.sum(np.logical_and(move_img_c, refer_img_c)*1)
        # union
        uni = np.sum(np.logical_or(move_img_c, refer_img_c)*1) + 0.0001

        jaccard_c.append(ints / uni)

    return jaccard_c


# precision and recall
def precision_recall_n_class(move_img, refer_img):
    # list of classes
    c_list = np.unique(refer_img)

    precision_c = []
    recall_c = []
    for c in range(len(c_list)):
        move_img_c = (move_img == c_list[c])
        refer_img_c = (refer_img == c_list[c])
        # intersection
        ints = np.sum(np.logical_and(move_img_c, refer_img_c)*1)
        # precision
        prec = ints / (np.sum(move_img_c*1) + 0.001)
        # recall
        recall = ints / (np.sum(refer_img_c*1) + 0.001)

        precision_c.append(prec)
        recall_c.append(recall)

    return precision_c, recall_c





# a_nii_path = '/home/xinyang/project_xy/mmwhs2017/dataset/ct_output/test_0.nii'
# b_nii_path = '/home/xinyang/project_xy/mmwhs2017/dataset/ct_train_hist/ct_train_1001_label.nii.gz'
#
# a_nii = nib.load(a_nii_path).get_data().copy()
# b_nii = nib.load(b_nii_path).get_data().copy()
#
# dice_c, conform_c, jaccard_c, precision_c, recall_c = seg_eval_metric(a_nii, b_nii)
#
# # dice_c = dice_n_class(a_nii, b_nii)
# # conform_c = conform_n_class(a_nii, b_nii)
# # jaccard_c = jaccard_n_class(a_nii, b_nii)
# # precision_c, recall_c = precision_recall_n_class(a_nii, b_nii)
#
# print dice_c
# print conform_c
# print jaccard_c
# print precision_c
# print recall_c