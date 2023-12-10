from model.MTCNNDetector import MTCNNDetector
import torch
import cv2
import os
workers = 0 if os.name == 'nt' else 4
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('Running on device: {}'.format(device))
mtcnn = MTCNNDetector(
    image_size=160, margin=0, min_face_size=20,
    thresholds=[0.8, 0.9, 0.95], resize_factor=0.709, post_process=True,
    device=device
)
for i in os.listdir('data/sample'):
    if i == '.DS_Store':
        continue
    path = os.path.join('data/sample', i)
    img = cv2.imread(path)
    print(img.shape)
    print(type(img))
    boxes, probs, points = mtcnn.detect(img[:, :, ::-1].copy(), landmarks=True)
    print(boxes.shape, probs.shape, points.shape)
    for box in boxes:
        x1, y1, x2, y2 = box
        cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), color=(255, 255, 0), thickness=2)
    for point in points:
        for po in point:
            cv2.circle(img, (int(po[0]), int(po[1])), 1, (0, 0, 255), 4)
    cv2.imshow('image', img)
    name = path.split('/')[-1]
    print(name)
    cv2.imwrite(f'results/{i}', img)
    cv2.waitKey(0)
