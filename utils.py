import re
import os
from PIL import Image
import shutil
import random
from xml.etree.ElementTree import parse
import os.path
import numpy as np
import cv2
import tifffile as tifi
import math
import matplotlib.pyplot as plt
import json
import pandas as pd
from glob import glob
# Parse PASCAL VOC dataset
import xmltodict



def tifs2exts(src_dir, dst_dir, ext):
    """
    Description:
        여러장의 tif 파일을 지정된 확장자로 변환후 저장합니다.(저장만 지원합니다)

    :param str src_dir:
    :param str dst_dir:
    :param str ext:
    """

    src_paths = all_image_paths(src_dir)
    assert len(src_paths) >= 0, '개수가 0보다 커야 합니다.'
    os.makedirs(dst_dir, exist_ok=True)
    for src_path in src_paths:
        try:
            _ = tif2ext(src_path, dst_dir, ext)
        except:
            print('Error image: {}'.format(src_path))


def tif2ext(src_path, dst_dir, ext):
    """
    tif 파일을 지정된 확장자로 변환해 저장합니다.

    :param str src_path:
    :param str dst_dir:
    :param str ext: jpg or .jpg
    """

    # dst path 을 생성합니다.
    ext = ext.replace('.', '').lower()
    name = get_name(src_path, ext=False)
    os.makedirs(dst_dir, exist_ok=True)
    dst_path = os.path.join(dst_dir, name + '.' + ext)

    # TIF 이미지 열기
    with Image.open(src_path) as img:
        # 변환된 이미지 저장
        # cv2.imwrite(dst_path, np.array(img))
        if ext == 'png':
            img.save(dst_path, format='PNG')
        elif ext == 'jpg' or ext == 'jpeg':
            img.save(dst_path, format='JPEG')
        else:
            print('지원하지 않는 확장자 입니다.')


def subdivide(src_dir, dst_dir, n_files, copy, shuffle=True):
    """
    Description
        폴더 안에 들어 있는 데이터를 소분합니다.

    Args:
        :param str src_dir: 폴더 별 파일 개 수
        :param str dst_dir: 저장 할 폴더
        :param int n_files: 서브 폴더 별 파일 개 수
         :param bool copy: 복사 여부
            True 시 파일 복사
            False 시 파일 이동(move)
        :param bool shuffle: 경로를 무작위로 섞음

    Usage:
        import os
        import shutil
        from utility import subdivide

        # 데이터를 코드별로 분류 하기
        root_src_dir = '../../datasets/unify_imgs_256'
        root_dst_dir = '../../datasets/codes/unify_imgs_256_subdivide'

        os.makedirs(root_dst_dir, exist_ok=True)
        codes = os.listdir(root_src_dir)

        for code in codes[:]:
            src_dir = os.path.join(root_src_dir, code)
            dst_dir = os.path.join(root_dst_dir, code)
            subdivide(src_dir, dst_dir, 1000, False, True)
    """

    file_paths = all_file_paths(src_dir)
    n_paths = len(file_paths)
    if shuffle:
        random.shuffle(file_paths)

    n_subfolders = int(np.ceil(n_paths / n_files))

    for ind in range(n_subfolders):

        # 서브 폴더를 생성합니다. 서브 폴더 이름은 인덱스로 합니다.
        subfolder_name = str(ind)
        subdir = os.path.join(dst_dir, subfolder_name)
        os.makedirs(os.path.join(dst_dir, str(subfolder_name)), exist_ok=True)

        # 전체 경로를 소분합니다.
        slc = slice(n_files * ind, n_files * (ind + 1))
        sliced_file_paths = file_paths[slc]

        # 파일을 서브폴더로 이동합니다.
        sliced_names = get_names(sliced_file_paths, ext=True)
        for src_path, name in zip(sliced_file_paths, sliced_names):
            dst_path = os.path.join(subdir, name)
            if copy:
                shutil.copy(src_path, dst_path)
            else:
                shutil.move(src_path, dst_path)


def get_names(paths, ext=True):
    names = [get_name(path, ext) for path in paths]
    return names


def get_name(path, ext=True):
    if ext:
        return os.path.split(path)[-1]
    else:
        return os.path.splitext(os.path.split(path)[-1])[0]


def get_ext(path):
    return os.path.splitext(path)[-1]


def get_exts(paths):
    return [get_ext(path) for path in paths]


def all_image_paths(image_dir):
    """
    Description:
        입력 된 폴더 하위 경로의 모든 이미지 파일 경로를 찾아내 반환합니다.
    Args:
        :param str image_dir: 이미지 폴더

    Returns: list, [str, str ... str]
    """
    file_paths = all_file_paths(image_dir)
    img_paths = filter_img_paths(file_paths)
    return img_paths


def all_file_paths(image_dir):
    """
    Description:
        입력 된 폴더 하위 경로의 모든 파일 경로를 찾아내 반환합니다.
    Args:
        :param str image_dir: 파일 폴더

    Returns: list, [str, str ... str]
    """
    paths = []
    for folder, subfolders, files in os.walk(image_dir):
        for file in files:
            path = os.path.join(folder, file)
            paths.append(path)
    return paths


def filter_img_paths(paths):
    """
    Description:
        입력 경로중 '.gif|.jpg|.jpeg|.tiff|.png' 해당 확장자를 가진 파일만 반환합니다.
    Args:
    :param: paths, [str, str, str ... str]
    :return: list, [str, str, str ... str]
    """
    regex = re.compile("(.*)(\w+)(.gif|.jpg|.jpeg|.tiff|.png|.bmp|.JPG|.HEIC|.tif)")
    img_paths = []
    for path in paths:
        if regex.search(path):
            img_paths.append(path)
    return img_paths


def get_pascal_bbox(filepath):
    """
    Description:
        pascal voc format 에서 bounding box 정보를 추출합니다.
    Args:
    :list return:
        [[x1, y1, x2, y2], [x1, y1, x2, y2] ... [x1, y1, x2, y2]]
    """
    tree = parse(filepath)
    root = tree.getroot()
    coords = []
    for obj in root.iter('object'):
        xmin = int(obj.find('bndbox').findtext('xmin'))
        xmax = int(obj.find('bndbox').findtext('xmax'))
        ymin = int(obj.find('bndbox').findtext('ymin'))
        ymax = int(obj.find('bndbox').findtext('ymax'))
        print(xmin, xmax, ymin, ymax)
        coords.append([xmin, ymin, xmax, ymax])
    return coords


def read_tiff_with_resize(filepath, resize_ratio=1.0):
    """
    Description:
        tiff file 을 읽어 반환합니다. 지정된 ratio 형태로 이미지를 resize 하여 반환합니다.
    Args:
        :param filepath:
        :param resize_ratio:
    :return numpy.ndarray img_resized: ndarray 형태로 반환합니다.
    """
    # tifffile open
    img = tifi.imread(filepath)
    dsize = (np.array(img.shape[:2]) * resize_ratio).astype(int)
    if not resize_ratio == 1.0:
        img_resized = cv2.resize(img, dsize=dsize[::-1])
    else:
        img_resized = img
    return img_resized


def save_image(obj, savepath):
    """
    Description:
    numpy 이미지를 저장합니다.

    Args:
        :param str savepath: 저장 경로

    """
    # save resized image
    cv2.imwrite(filename=savepath, img=obj)

    if not os.path.exists(savepath):
        print('파일이 저장되지 않았습니다. 지정된 경로 : {}'.format(savepath))


def plot_images(imgs, names=None, random_order=False, savepath=None, show=True):
    h = math.ceil(math.sqrt(len(imgs)))
    fig = plt.figure()
    plt.gcf().set_size_inches((20, 20))
    for i in range(len(imgs)):
        ax = fig.add_subplot(h, h, i + 1)
        if random_order:
            ind = random.randint(0, len(imgs) - 1)
        else:
            ind = i
        img = imgs[ind]
        plt.axis('off')
        plt.imshow(img)
        # save image
        if not names is None:
            ax.set_title(str(names[ind]), c='red')
    if not savepath is None:
        plt.savefig(savepath)
    if show:
        plt.tight_layout()
        plt.show()


def paths2imgs(paths, resize=None, error=None):
    """
    Description:
        경로을 numpy 로 변환해 반환합니다.
    Args:
        :param paths: list, [str, str, ... str], 이미지 저장 경로
        :param resize: tuple, (h, w)
        :param error: 잘못된 경로를 반환합니다.
    :list return:
    """
    imgs = []
    error_paths = []
    for path in paths:
        try:
            img = path2img(path, resize)
            imgs.append(img)
        except:
            print(os.path.exists(path))
            print("{} 해당 경로에 파일이 존재하지 않습니다.".format(path))
            error_paths.append(path)

    if error == 'error_return':
        return imgs, error_paths

    return imgs


def path2img(path, resize=None):
    """
    Description:
        경로에 있는 이미지를 RGB 컬러 형식으로 불러옵니다
        resize 에 값을 주면 해당 크기로 이미지를 불러옵니다.
    Args:
        :param path: str
        :param resize: tuple or list , (W, H)
    :return np.ndarray img: shape=(h, w, 3)
    """

    # 경로 중 한글명이 있어 cv2 만으로는 읽을 수 없었기 때문에, numpy 로 파일을 읽은 후 이를 cv2.imdecode 를 통해 이미지로 변환합니다.
    # HEIC 경로인 경우 아래와 경로로 작업 합니다.
    if path.endswith('heic') or path.endswith('HEIC'):
        img = np.array(Image.open(path))

    else:
        img = cv2.imread(path)

        # BGR 을 RGB 로 변환합니다.
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    if resize:
        h, w = resize
        img = cv2.resize(img, (w, h))

    return img


def polygon_center_xy(*polygon_points):
    """
    Description:
        polygon 내 center 좌표  값을 찾아 반환합니다.
    Args:
        :param polygon_points:
    :return:
    """
    x_coordinates, y_coordinates = zip(*polygon_points)
    center_x = sum(x_coordinates) / len(polygon_points)
    center_y = sum(y_coordinates) / len(polygon_points)
    return center_x, center_y


def write_msg_in_center(img, text, *polygon_points):
    """
    Description:
        폴리곤 중앙에 지정된 메시지를 그립니다.

    Args:
        :param str img: 이미지에 입력할 텍스트
        :param str text: 이미지에 입력할 텍스트
        :param list polygon_points: [(x1, y1),(x2, y2) ... (x3, y3)]

    :return:
    """

    cx, cy = polygon_center_xy(*polygon_points)
    position = int(cx), int(cy)

    # 텍스트를 쓸 위치와 내용을 정의합니다.
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 5  # 글꼴 크기
    font_color = (0, 0, 255)  # 글꼴 색상 (BGR 형식)
    font_thickness = 10  # 글꼴 두께

    img = cv2.putText(img, text, position, font, font_scale, font_color, font_thickness)
    return img


def extract_pixel_in_polygon(image, coords):
    """
    Description:
        polygon 내 존재하는 모든 픽셀 정보를 가져와 반환합니다.

    Args:
        :param np.ndarray image:
        :param list coords: [(x1, y1),(x2, y2) ... (x3, y3)] or [(x1, y1 ,x2, y2, ... ,x_n, y_n]]
    :return:
    """
    canvas = np.zeros(shape=image.shape[:2], dtype=np.uint8)
    resized_coords = np.array(coords)
    resized_coords = resized_coords.reshape((-1, 1, 2))
    mask = cv2.fillPoly(canvas, [resized_coords], color=1)

    # polyline 을 포함하는 가장 작은 직사각형의 좌표를 추출합니다.
    """
    np.argwhere(mask) =  [[ 752 2298]
                          [ 752 2299]
                             ...
                          [ 752 2300]]
    """
    l_y = np.argwhere(mask)[:, 0].min()
    r_y = np.argwhere(mask)[:, 0].max()
    l_x = np.argwhere(mask)[:, 1].min()
    r_x = np.argwhere(mask)[:, 1].max()

    # 원본 이미지에서 직사각형 부분만을 가져옵니다.
    cropped_polylined_img = image[l_y: r_y, l_x: r_x]

    # 마스크에서 직사각형 부분만을 가져옵니다.
    cropped_mask = mask[l_y: r_y, l_x: r_x]

    # 원본 이미지에서 마스크 영역만 추출합니다.
    masked_cropped_polylined_img = cropped_polylined_img * np.expand_dims(cropped_mask, -1)
    return masked_cropped_polylined_img


# 라벨링 확인
def filter_xml_paths(paths):
    """
    Description:
        입력 경로중 '.gif|.jpg|.jpeg|.tiff|.png' 해당 확장자를 가진 파일만 반환합니다.
    Args:
    :param: paths, [str, str, str ... str]
    :return: list, [str, str, str ... str]
    """
    regex = re.compile("(.*)(\w+)(.xml)")
    xml_paths = []
    for path in paths:
        if regex.search(path):
            xml_paths.append(path)
    return xml_paths

