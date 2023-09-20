import os
import cv2
import numpy as np
from tensorflow import keras
import tarfile
from PIL import Image
import matplotlib.pyplot as plt
import tensorflow as tf
import xmltodict
from utils import all_image_paths, get_names, all_file_paths, filter_xml_paths


def download_rice_drone_dataset(save_dir):
    name = 'Riceseedlingdetection'
    url = "https://rebrand.ly/{}".format(name)
    filename = '{}.tgz'.format(name)
    filepath = os.path.join(save_dir, filename)
    if not os.path.exists(filepath):
        print('{}에 다운받습니다.'.format(filepath))
        keras.utils.get_file(filepath, url)


def extraxct_tarfile(filepath, savedir):
    # extract tgz file
    if filepath.endswith("tar.gz") or filepath.endswith("tgz"):
        tar = tarfile.open(filepath, "r:gz")
        os.makedirs(savedir, exist_ok=True)
        tar.extractall(savedir)
        tar.close()


def pascal2coco(pascal_paths, image_paths, scale_factor=1, coord_type='xywh'):
    """
    Description:
    pascal dataset 을 coco json 형태로 변환해 반환한다.

    Args:
        :param list pascal_paths: [path0, path1, path2 ..., path N]
    :dict return:
    """
    assert get_names(pascal_paths, ext=False) == get_names(image_paths, ext=False), '두 리스트 이름 순서와 개 수는 같아야 합니다. '
    coco_datasets = []
    for ind, pascal_path in enumerate(pascal_paths):
        img_path = image_paths[ind]
        img = Image.open(img_path)
        np_img = np.array(img)
        if not scale_factor == 1:
            np_img = cv2.resize(np_img, dsize=None, fx=scale_factor, fy=scale_factor)

        f = open(pascal_path, 'r')
        xml_str = f.read()
        xml_dict = xmltodict.parse(xml_str)

        # parse xml
        coco_dict = {}
        filename = xml_dict['annotation']['filename']
        objects = xml_dict['annotation']['object']
        size = xml_dict['annotation']['size']
        width, height = float(size['width']) * scale_factor, float(size['height']) * scale_factor

        areas = []
        bboxes = []
        for id, obj in enumerate(objects):
            xmin = float(obj['bndbox']['xmin']) * scale_factor
            ymin = float(obj['bndbox']['ymin']) * scale_factor
            xmax = float(obj['bndbox']['xmax']) * scale_factor
            ymax = float(obj['bndbox']['ymax']) * scale_factor

            # add bboxes
            if coord_type == 'xyxy':
                bbox = (xmin, ymin, xmax, ymax)

            elif coord_type == 'rel_xyxy':
                bbox = (xmin / width, ymin / height, xmax / width, ymax / height)

            elif coord_type == 'yxyx':
                bbox = (ymin, xmin, ymax, xmax)

            elif coord_type == 'rel_yxyx':
                bbox = (ymin / height, xmin / width, ymax / height, xmax / width)
            elif coord_type == 'xywh':
                bbox = ((xmax + xmin) / 2, (ymax + ymin) / 2, (xmax - xmin), (ymax - ymin))

            else:
                raise NotImplementedError

            bboxes.append(bbox)

            # add bbox area
            x_gap = xmax - xmin
            y_gap = ymax - ymin
            area = x_gap * y_gap
            areas.append(area)

        n_objects = len(bboxes)
        labels = [0] * n_objects
        is_crowds = [0] * n_objects
        ids = list(range(n_objects))

        # Generate coco dict
        image_ind = 0
        coco_dict['image/filename'] = filename
        coco_dict['image/id'] = image_ind
        coco_dict['image'] = np.array(np_img)
        coco_dict['objects'] = {}

        coco_dict['objects']['area'] = np.array(areas)
        coco_dict['objects']['bbox'] = np.array(bboxes)
        coco_dict['objects']['id'] = np.array(ids)
        coco_dict['objects']['is_crowd'] = np.array(is_crowds)
        coco_dict['objects']['label'] = np.array(labels)

        coco_datasets.append(coco_dict)

    return coco_datasets


def preprocess_label(bbox, image_shape, corner=True):
    """좌표형태를 x1 y1 x2 y2 형태로 변환합니다.
    기존의 형태는 상대 좌표로 되어 있으며 y1, x1, y2, x2 순서로 되어 있습니다.

    Args:
        :param numpy.ndarray bbox: 상대(relative) y1, x1, y2, x2
            relative y1 = y1 / h
            relative x1 = x1 / w
            relative y2 = y2 / h
            relative x2 = x2 / w
        :param numpy.ndarray image_shape: [h, w]
        :param bool corner: x1 y1 x2 y2 형태로 반환, False 이면 cx cy w h 형태로 반환
    """

    def _swap_xy(boxes):
        return np.stack([boxes[:, 1], boxes[:, 0], boxes[:, 3], boxes[:, 2]], axis=-1)

    def _convert_to_xywh(boxes):
        """Changes the box format to center, width and height.

        Arguments:
        boxes: A tensor of rank 2 or higher with a shape of `(..., num_boxes, 4)`
            representing bounding boxes where each box is of the format
            `[xmin, ymin, xmax, ymax]`.

        Returns:
        converted boxes with shape same as that of boxes.
        """
        return tf.concat(
            [(boxes[..., :2] + boxes[..., 2:]) / 2.0, boxes[..., 2:] - boxes[..., :2]],
            axis=-1,
        )

    bbox = _swap_xy(bbox)
    bbox = np.stack(
        [
            bbox[:, 0] * image_shape[1],
            bbox[:, 1] * image_shape[0],
            bbox[:, 2] * image_shape[1],
            bbox[:, 3] * image_shape[0],
        ],
        axis=-1,
    )
    if not corner:
        bbox = _convert_to_xywh(bbox)
    return bbox


if __name__ == '__main__':
    # dataset download
    save_dir = os.path.join(os.getcwd(), 'sample_data')
    os.makedirs(save_dir, exist_ok=True)
    download_rice_drone_dataset(save_dir=save_dir)

    # extract dataset
    filepath = os.path.join(save_dir, 'Riceseedlingdetection.tgz', )
    save_dir = os.path.join(save_dir, 'Riceseedlingdetection')
    extraxct_tarfile(filepath, save_dir)

    # 샘플 사진 확인
    image_dir = os.path.join(save_dir, 'JPEGImages')
    img_paths = sorted(all_image_paths(image_dir))
    print('Image 개 수 : {}'.format(len(img_paths)))

    # 샘플 사진 확인
    img_index = 0
    img_names = get_names(img_paths, ext=False)
    img_name = img_names[img_index]
    img_path = os.path.join(image_dir, img_name + '.jpg')
    img = Image.open(img_path)
    np_img = np.array(img)
    print('샘플 사진')
    plt.imshow(img)
    plt.show()

    # 라벨링 정보 확인
    label_dir = os.path.join(save_dir, 'Annotations')
    paths = all_file_paths(label_dir)
    xml_paths = filter_xml_paths(paths)
    xml_paths = sorted(xml_paths)
    print('Label 개 수 : {}'.format(len(xml_paths)))

    # coco 형식으로 변환
    coco_datasets = pascal2coco(xml_paths, img_paths, 2)
    names = [data['image/filename'] for data in coco_datasets]

    # 데이터 변환 확인
    sample_dict = coco_datasets[img_index]
    img = sample_dict['image']
    bboxes = sample_dict['objects']['bbox']
    abs_bboxes = preprocess_label(np.array(bboxes), img.shape[:2])

    # 데이터 시각화
    for sample_coord in abs_bboxes.astype(int):
        color = (0, 255, 0)  # 초록색 (BGR 색상 코드)
        thickness = 1  # 선 두께
        image = cv2.rectangle(img, (sample_coord[0], sample_coord[1]), (sample_coord[2], sample_coord[3]), color,
                              thickness)
    plt.imshow(image)
    plt.show()
