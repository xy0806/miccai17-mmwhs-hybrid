from __future__ import division
import numpy as np
import nibabel as nib
import copy
import cv2
import scipy
import scipy.ndimage
from scipy.ndimage import rotate
from skimage.transform import resize
from scipy.ndimage import measurements


##############################
# process .nii data
def load_data_pairs(pair_list, resize_r, rename_map):
    """load all volume pairs"""
    img_clec = []
    label_clec = []

    # rename_map = [0, 205, 420, 500, 550, 600, 820, 850]
    for k in range(0, len(pair_list), 2):
        img_path = pair_list[k]
        lab_path = pair_list[k+1]
        img_data = nib.load(img_path).get_data().copy()
        lab_data = nib.load(lab_path).get_data().copy()

        ###preprocessing
        # resize
        resize_dim = (np.array(img_data.shape) * resize_r).astype('int')
        img_data = resize(img_data, resize_dim, order=1, preserve_range=True)
        lab_data = resize(lab_data, resize_dim, order=0, preserve_range=True)
        lab_r_data = np.zeros(lab_data.shape, dtype='int32')

        # rename labels
        for i in range(len(rename_map)):
            lab_r_data[lab_data == rename_map[i]] = i

        # for s in range(img_data.shape[2]):
        #     cv2.imshow('img', np.concatenate(((img_data[:,:,s]).astype('uint8'), (lab_r_data[:,:,s]*30).astype('uint8')), axis=1))
        #     cv2.waitKey(20)

        img_clec.append(img_data)
        label_clec.append(lab_r_data)

    return img_clec, label_clec


def get_batch_patches(img_clec, label_clec, patch_dim, batch_size, chn=1, flip_flag=True, rot_flag=True):
    """generate a batch of paired patches for training"""
    batch_img = np.zeros([batch_size, patch_dim, patch_dim, patch_dim, chn]).astype('float32')
    batch_label = np.zeros([batch_size, patch_dim, patch_dim, patch_dim]).astype('int32')

    for k in range(batch_size):
        # randomly select an image pair
        rand_idx = np.arange(len(img_clec))
        np.random.shuffle(rand_idx)
        rand_img = img_clec[rand_idx[0]]
        rand_label = label_clec[rand_idx[0]]
        rand_img = rand_img.astype('float32')
        rand_label = rand_label.astype('int32')

        # randomly select a box anchor
        l, w, h = rand_img.shape
        l_rand = np.arange(l - patch_dim)
        w_rand = np.arange(w - patch_dim)
        h_rand = np.arange(h - patch_dim)
        np.random.shuffle(l_rand)
        np.random.shuffle(w_rand)
        np.random.shuffle(h_rand)
        pos = np.array([l_rand[0], w_rand[0], h_rand[0]])
        # crop
        img_temp = copy.deepcopy(rand_img[pos[0]:pos[0]+patch_dim, pos[1]:pos[1]+patch_dim, pos[2]:pos[2]+patch_dim])
        # normalization
        img_temp = img_temp/255.0
        mean_temp = np.mean(img_temp)
        dev_temp = np.std(img_temp)
        img_norm = (img_temp - mean_temp) / dev_temp

        label_temp = copy.deepcopy(rand_label[pos[0]:pos[0]+patch_dim, pos[1]:pos[1]+patch_dim, pos[2]:pos[2]+patch_dim])

        # possible augmentation
        # rotation
        if rot_flag and np.random.random() > 0.65:
            # print 'rotating patch...'
            rand_angle = [-25, 25]
            np.random.shuffle(rand_angle)
            img_norm = rotate(img_norm, angle=rand_angle[0], axes=(1, 0), reshape=False, order=1)
            label_temp = rotate(label_temp, angle=rand_angle[0], axes=(1, 0), reshape=False, order=0)

        batch_img[k, :, :, :, chn-1] = img_norm
        batch_label[k, :, :, :] = label_temp

    return batch_img, batch_label


# calculate the cube information
def fit_cube_param(vol_dim, cube_size, ita):
    dim = np.asarray(vol_dim)
    # cube number and overlap along 3 dimensions
    fold = dim / cube_size + ita
    ovlap = np.ceil(np.true_divide((fold * cube_size - dim), (fold - 1)))
    ovlap = ovlap.astype('int')

    fold = np.ceil(np.true_divide((dim + (fold - 1)*ovlap), cube_size))
    fold = fold.astype('int')

    return fold, ovlap


# decompose volume into list of cubes
def decompose_vol2cube(vol_data, batch_size, cube_size, n_chn, ita):
    cube_list = []
    # get parameters for decompose
    fold, ovlap = fit_cube_param(vol_data.shape, cube_size, ita)
    dim = np.asarray(vol_data.shape)
    # decompose
    for R in range(0, fold[0]):
        r_s = R*cube_size - R*ovlap[0]
        r_e = r_s + cube_size
        if r_e >= dim[0]:
            r_s = dim[0] - cube_size
            r_e = r_s + cube_size
        for C in range(0, fold[1]):
            c_s = C*cube_size - C*ovlap[1]
            c_e = c_s + cube_size
            if c_e >= dim[1]:
                c_s = dim[1] - cube_size
                c_e = c_s + cube_size
            for H in range(0, fold[2]):
                h_s = H*cube_size - H*ovlap[2]
                h_e = h_s + cube_size
                if h_e >= dim[2]:
                    h_s = dim[2] - cube_size
                    h_e = h_s + cube_size
                # partition multiple channels
                cube_temp = vol_data[r_s:r_e, c_s:c_e, h_s:h_e]
                cube_batch = np.zeros([batch_size, cube_size, cube_size, cube_size, n_chn]).astype('float32')
                cube_batch[0, :, :, :, 0] = copy.deepcopy(cube_temp)
                # save
                cube_list.append(cube_batch)

    return cube_list


# compose list of label cubes into a label volume
def compose_label_cube2vol(cube_list, vol_dim, cube_size, ita, class_n):
    # get parameters for compose
    fold, ovlap = fit_cube_param(vol_dim, cube_size, ita)
    # create label volume for all classes
    label_classes_mat = (np.zeros([vol_dim[0], vol_dim[1], vol_dim[2], class_n])).astype('int32')
    idx_classes_mat = (np.zeros([cube_size, cube_size, cube_size, class_n])).astype('int32')

    p_count = 0
    for R in range(0, fold[0]):
        r_s = R*cube_size - R*ovlap[0]
        r_e = r_s + cube_size
        if r_e >= vol_dim[0]:
            r_s = vol_dim[0] - cube_size
            r_e = r_s + cube_size
        for C in range(0, fold[1]):
            c_s = C*cube_size - C*ovlap[1]
            c_e = c_s + cube_size
            if c_e >= vol_dim[1]:
                c_s = vol_dim[1] - cube_size
                c_e = c_s + cube_size
            for H in range(0, fold[2]):
                h_s = H*cube_size - H*ovlap[2]
                h_e = h_s + cube_size
                if h_e >= vol_dim[2]:
                    h_s = vol_dim[2] - cube_size
                    h_e = h_s + cube_size
                # histogram for voting
                for k in range(class_n):
                    idx_classes_mat[:, :, :, k] = (cube_list[p_count] == k)
                # accumulation
                label_classes_mat[r_s:r_e, c_s:c_e, h_s:h_e, :] = label_classes_mat[r_s:r_e, c_s:c_e, h_s:h_e, :] + idx_classes_mat

                p_count += 1
    # print 'label mat unique:'
    # print np.unique(label_mat)

    compose_vol = np.argmax(label_classes_mat, axis=3)
    # print np.unique(label_mat)

    return compose_vol


# compose list of probability cubes into a probability volumes
def compose_prob_cube2vol(cube_list, vol_dim, cube_size, ita, class_n):
    # get parameters for compose
    fold, ovlap = fit_cube_param(vol_dim, cube_size, ita)
    # create label volume for all classes
    map_classes_mat = (np.zeros([vol_dim[0], vol_dim[1], vol_dim[2], class_n])).astype('float32')
    cnt_classes_mat = (np.zeros([vol_dim[0], vol_dim[1], vol_dim[2], class_n])).astype('float32')

    p_count = 0
    for R in range(0, fold[0]):
        r_s = R*cube_size - R*ovlap[0]
        r_e = r_s + cube_size
        if r_e >= vol_dim[0]:
            r_s = vol_dim[0] - cube_size
            r_e = r_s + cube_size
        for C in range(0, fold[1]):
            c_s = C*cube_size - C*ovlap[1]
            c_e = c_s + cube_size
            if c_e >= vol_dim[1]:
                c_s = vol_dim[1] - cube_size
                c_e = c_s + cube_size
            for H in range(0, fold[2]):
                h_s = H*cube_size - H*ovlap[2]
                h_e = h_s + cube_size
                if h_e >= vol_dim[2]:
                    h_s = vol_dim[2] - cube_size
                    h_e = h_s + cube_size
                # accumulation
                map_classes_mat[r_s:r_e, c_s:c_e, h_s:h_e, :] = map_classes_mat[r_s:r_e, c_s:c_e, h_s:h_e, :] + cube_list[p_count]
                cnt_classes_mat[r_s:r_e, c_s:c_e, h_s:h_e, :] = cnt_classes_mat[r_s:r_e, c_s:c_e, h_s:h_e, :] + 1.0

                p_count += 1

    # elinimate NaN
    nan_idx = (cnt_classes_mat == 0)
    cnt_classes_mat[nan_idx] = 1.0
    # average
    compose_vol = map_classes_mat / cnt_classes_mat

    return compose_vol

# Remove small connected components
def remove_minor_cc(vol_data, rej_ratio, rename_map):
    """Remove small connected components refer to rejection ratio"""
    """Usage
        # rename_map = [0, 205, 420, 500, 550, 600, 820, 850]
        # nii_path = '/home/xinyang/project_xy/mmwhs2017/dataset/ct_output/test/test_4.nii'
        # vol_file = nib.load(nii_path)
        # vol_data = vol_file.get_data().copy()
        # ref_affine = vol_file.affine
        # rem_vol = remove_minor_cc(vol_data, rej_ratio=0.2, class_n=8, rename_map=rename_map)
        # # save
        # rem_path = 'rem_cc.nii'
        # rem_vol_file = nib.Nifti1Image(rem_vol, ref_affine)
        # nib.save(rem_vol_file, rem_path)

        #===# possible be parallel in future
    """

    rem_vol = copy.deepcopy(vol_data)
    class_n = len(rename_map)
    # retrieve all classes
    for c in range(1, class_n):
        print 'processing class %d...' % c

        class_idx = (vol_data==rename_map[c])*1
        class_vol = np.sum(class_idx)
        labeled_cc, num_cc = measurements.label(class_idx)
        # retrieve all connected components in this class
        for cc in range(1, num_cc+1):
            single_cc = ((labeled_cc==cc)*1)
            single_vol = np.sum(single_cc)
            # remove if too small
            if single_vol / (class_vol*1.0) < rej_ratio:
                rem_vol[labeled_cc==cc] = 0

    return rem_vol






