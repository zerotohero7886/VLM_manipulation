import matplotlib.pyplot as plt
from PIL import Image
from typing import List


def calculate_iou(box1, box2):
    # Calculate the intersection area
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    intersection_area = max(0, x2 - x1) * max(0, y2 - y1)

    # Calculate the union area
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = box1_area + box2_area - intersection_area

    # Calculate the IoU
    iou = intersection_area / union_area

    return iou


def show_images(
    images: List[Image.Image], titles: List[str] = None, figsize: tuple = (15, 5)
) -> None:
    """
    Display a list of images in a row.

    Parameters:
    - images: List[Image.Image] - List of PIL Image objects to be displayed.
    - titles: List[str] - List of titles for each image. Defaults to None.
    - figsize: tuple - Size of the figure. Defaults to (15, 5).
    """
    fig, axs = plt.subplots(1, len(images), figsize=figsize)

    if len(images) == 1:
        axs = [axs]

    for i, (ax, img) in enumerate(zip(axs, images)):
        ax.imshow(img)
        ax.axis("off")
        if titles and i < len(titles):
            ax.set_title(titles[i])

    plt.tight_layout()
    plt.show()
