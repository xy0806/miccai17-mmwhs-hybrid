import os
from nii2avi import save_nii2avi

nii_folder = '/home/xinyang/project_xy/mmwhs2017/dataset/ct_output/25_ds'
img_n = 27

for k in range(img_n):
    print 'converting No. %d volume...' % (k)
    nii_path = os.path.join(nii_folder, ('test_' + str(k) + '.nii.gz'))
    video_path = os.path.join(nii_folder, ('test_' + str(k) + '.mp4'))
    save_nii2avi(niipath=nii_path, Save_file_name=video_path, Time_Loop=40, Avi_rate=5, Angle=20)
