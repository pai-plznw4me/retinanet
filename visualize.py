import numpy as np
import matplotlib.pyplot as plt
import cv2


def visualize_detections(
    image, boxes, classes, scores, figsize=(7, 7), linewidth=1, color=[0, 0, 1]
):
    """Visualize Detections"""
    image = np.array(image, dtype=np.uint8)
    plt.figure(figsize=figsize)
    plt.axis("off")
    plt.imshow(image)
    ax = plt.gca()
    for box, _cls, score in zip(boxes, classes, scores):
        text = "{}: {:.2f}".format(_cls, score)
        x1, y1, x2, y2 = box
        w, h = x2 - x1, y2 - y1
        patch = plt.Rectangle(
            [x1, y1], w, h, fill=False, edgecolor=color, linewidth=linewidth
        )
        ax.add_patch(patch)
        ax.text(
            x1,
            y1,
            text,
            bbox={"facecolor": color, "alpha": 0.4},
            clip_box=ax.clipbox,
            clip_on=True,
        )
    plt.show()
    return ax


def visualize_bboxes(image, bboxes):
    # 데이터 시각화
    bboxes = np.array(bboxes).astype(int)
    for sample_coord in bboxes:
        sample_coord.astype('int')
        color = (0, 255, 0)  # 초록색 (BGR 색상 코드)
        thickness = 1  # 선 두께
        image = cv2.rectangle(image.astype(np.uint8), (sample_coord[0], sample_coord[1]),
                              (sample_coord[2], sample_coord[3]), color,
                              thickness)
    plt.imshow(image)
    plt.show()
    return image
