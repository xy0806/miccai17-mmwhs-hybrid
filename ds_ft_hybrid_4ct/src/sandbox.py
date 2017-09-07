import numpy as np
import nibabel as nib
import cv2
from scipy.ndimage import rotate


nii_path = '/home/xinyang/project_xy/mmwhs2017/dataset/ct_train_hist/ct_train_1001_image.nii.gz'

nii_fid = nib.load(nii_path)
vol_data = nii_fid.get_data()
vol_data = vol_data.astype('uint8')
vol_dim = np.asarray(vol_data.shape)

# for k in range(vol_dim[2]):
#     cv2.imshow('slice', vol_data[:, :, k])
#     cv2.waitKey(0)


# rotate
rot_vol_data = rotate(vol_data, angle=25, axes=(1, 0), reshape=False, order=1)
rot_vol_dim = np.asarray(rot_vol_data.shape)

for k in range(rot_vol_dim[2]):
    cv2.imshow('rot_slice', np.concatenate((vol_data[:, :, k], rot_vol_data[:, :, k]), axis=1))
    cv2.waitKey(0)



