"""
tensorflow 을 활용해 retinanet 에 입력될 data 을 전처리합니다.
"""
import os
from anchor import AnchorBox
from coordinate import convert_to_corners
from decode import decode_box_predictions
from encode import LabelEncoder
from main_rice_data_download import pascal2coco
from utils import all_file_paths, filter_xml_paths, all_image_paths
import numpy as np
from visualize import visualize_bboxes

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


# get anchors (for test)
anchor_box = AnchorBox()
anchor_bboxes = anchor_box.get_anchors(640, 640)

# set dataset
imgs = np.array([coco_datasets[0]['image'], coco_datasets[1]['image']])
annos = [coco_datasets[0]['objects']['bbox'], coco_datasets[1]['objects']['bbox']]
labels = [coco_datasets[0]['objects']['label'], coco_datasets[1]['objects']['label']]

# 데이터가 정상적으로 변환 되었는지를 확인하기 위한 코드 (테스트 용)
for i in range(len(annos)):
    sample_xyxy = convert_to_corners(annos[i])
    visualize_bboxes(imgs[i].astype(int), sample_xyxy)

# label encode
label_encoder = LabelEncoder()
preproc_imgs, labels = label_encoder.encode_batch(imgs, annos, labels)

# generate anchor( 테스트 용)
anchor_boxes = label_encoder.anchor_boxes
anchor_boxes_xyxy = convert_to_corners(anchor_boxes)

# decode
bboxes = decode_box_predictions(anchor_boxes_xyxy, labels)

# mask
pos_mask = labels[0][:, -1] == 1
neg_mask = labels[0][:, -1] == -1
igr_mask = labels[0][:, -1] == -2

# pos anchor
pos_anchor_xywh = anchor_bboxes[pos_mask].numpy().astype(int)
pos_anchor_xyxy = convert_to_corners(pos_anchor_xywh)
# visualize_bboxes(imgs[0].astype(int), pos_anchor_xyxy)

# neg anchor
neg_anchor_xywh = anchor_bboxes[neg_mask].numpy().astype(int)
neg_anchor_xyxy = convert_to_corners(neg_anchor_xywh)
# visualize_bboxes(imgs[0].astype(int), neg_anchor_xyxy)

# ignore anchor
igr_anchor_xywh = anchor_bboxes[igr_mask].numpy().astype(int)
igr_anchor_xyxy = convert_to_corners(igr_anchor_xywh)
visualize_bboxes(imgs[0].astype(int), igr_anchor_xyxy)
