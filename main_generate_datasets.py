"""
tensorflow 을 활용해 retinanet 에 입력될 data 을 전처리합니다.
"""
import os
import numpy as np
from encode import LabelEncoder
from main_rice_data_download import pascal2coco
from utils import all_file_paths, filter_xml_paths, all_image_paths
from tqdm import tqdm

# 라벨링 정보 확인
label_dir = os.path.join(os.getcwd(), 'sample_data', 'Riceseedlingdetection', 'Annotations')
paths = all_file_paths(label_dir)
xml_paths = filter_xml_paths(paths)
xml_paths = sorted(xml_paths)
print('Label 개 수 : {}'.format(len(xml_paths)))

# 이미지 정보 확인
image_dir = os.path.join(os.getcwd(), 'sample_data', 'Riceseedlingdetection', 'JPEGImages')
img_paths = sorted(all_image_paths(image_dir))

# coco 형식으로 변환
coco_datasets = pascal2coco(xml_paths, img_paths, 2, coord_type='xywh')

# set dataset
imgs = []
annos = []
labels = []

for coco_dataset in tqdm(coco_datasets[:]):
    imgs.append(coco_dataset['image'])
    annos.append(coco_dataset['objects']['bbox'])
    labels.append(coco_dataset['objects']['label'])

# generate encode dataset (for resnet backbone)
imgs = np.array(imgs)
label_encoder = LabelEncoder()
preproc_imgs, encode_labels = label_encoder.encode_batch(imgs, annos, labels)

# check shape
print('label shape : {}'.format(encode_labels.shape))
print('image shape : {}'.format(preproc_imgs.shape))

# save dir
save_encode_dir = os.path.join(os.getcwd(), 'sample_data', 'Riceseedlingdetection', 'Encode')
os.makedirs(save_encode_dir, exist_ok=True)

# save encoded label data
labels_path = os.path.join(save_encode_dir, 'labels.npy')
np.save(labels_path, encode_labels)

# save encoded label data
images_path = os.path.join(save_encode_dir, 'preproc_resnet_images.npy')
np.save(images_path, preproc_imgs)