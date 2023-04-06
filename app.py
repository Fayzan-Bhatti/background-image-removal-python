import cv2
import numpy as np
import urllib.request

# Load the pre-trained Mask R-CNN model
net = cv2.dnn.readNetFromTensorflow('mask_rcnn_inception_v2_coco_2018_01_28/frozen_inference_graph.pb',
                                    'mask_rcnn_inception_v2_coco_2018_01_28/mask_rcnn_inception_v2_coco_2018_01_28.pbtxt')

# Function to remove the background from an image
def remove_background(img_url):
    # Load the image from the given URL
    img_resp = urllib.request.urlopen(img_url)
    img_array = np.array(bytearray(img_resp.read()), dtype=np.uint8)
    img = cv2.imdecode(img_array, -1)

    # Prepare the input blob for the Mask R-CNN model
    blob = cv2.dnn.blobFromImage(img, swapRB=True, crop=False)
    net.setInput(blob)

    # Perform object detection and segmentation using the Mask R-CNN model
    detections = net.forward()
    masks = detections[:, :, 5:]
    classes = detections[:, :, 1:2]

    # Find the mask with the highest score for the person class
    person_mask = None
    max_score = 0
    for i in range(masks.shape[2]):
        if classes[0, i, 0] == 1:
            mask = masks[:, :, i]
            score = np.sum(mask)
            if score > max_score:
                person_mask = mask
                max_score = score

    # Apply the mask to the original image to remove the background
    foreground = cv2.bitwise_and(img, img, mask=person_mask)
    background = np.zeros_like(img)
    background[:] = (255, 255, 255)
    alpha = person_mask.astype(float) / 255.0
    blended = cv2.addWeighted(foreground.astype(float), alpha, background.astype(float), 1 - alpha, 0)
    result = blended.astype(np.uint8)

    return result

# Example usage
img_url = 'https://example.com/image.jpg'
result = remove_background(img_url)
cv2.imwrite('result.jpg', result)
