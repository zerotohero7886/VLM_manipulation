import matplotlib.pyplot as plt
from PIL import Image
from typing import List
import torch


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

def class_agnostic_nms_single_image(detection_result, iou_threshold=0.5):
    """
    Perform class-agnostic non-maximum suppression (NMS) on a single image's detection results.
    
    Args:
        detection_result (dict): A dictionary containing 'scores', 'labels', and 'boxes'.
        iou_threshold (float): IoU threshold for NMS.
    
    Returns:
        dict: Filtered detection results with 'scores', 'labels', and 'boxes'.
    """
    scores = detection_result['scores']
    boxes = detection_result['boxes']
    labels = detection_result['labels']

    # Sort indices by scores in descending order
    indices = scores.argsort(descending=True)
    keep = []

    while indices.numel() > 0:
        idx = indices[0].item()
        keep.append(idx)

        if indices.numel() == 1:
            break

        ious = torch.tensor([calculate_iou(boxes[idx].tolist(), boxes[i].tolist()) for i in indices[1:]])

        indices = indices[1:][ious <= iou_threshold]

    filtered_boxes = boxes[keep]
    filtered_scores = scores[keep]
    filtered_labels = labels[keep]

    return {
        'scores': filtered_scores,
        'labels': filtered_labels,
        'boxes': filtered_boxes
    }

def class_agnostic_nms(detection_results, iou_threshold=0.5):
    """
    Perform class-agnostic non-maximum suppression (NMS) on detection results for multiple images.
    
    Args:
        detection_results (List[Dict]): A list of dictionaries, each containing 'scores', 'labels', and 'boxes'.
        iou_threshold (float): IoU threshold for NMS.
    
    Returns:
        List[Dict]: Filtered detection results for each image.
    """
    return [class_agnostic_nms_single_image(result, iou_threshold) for result in detection_results]


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
