
import cv2

from category_wise_boxes import category_wise_boxes


img_seg_path = "./labelling_release/train/GT/05_000095.png"
img_seg = cv2.imread(img_seg_path)


real_img_path = "./labelling_release/train/images/05_000095.png"
real_img = cv2.imread(real_img_path)

boxes = category_wise_boxes(real_img, img_seg)
# boxes is a list of lists
