from model.MTCNNDetector import MTCNNDetector
from matchers.networks.resnet import res_encoder
from matchers.framework.prototypical_loss_fn import euclidean_distance
from matchers.common.utils import encoder_predict
from matchers.networks.simple_net import ProtoNet
import torchvision.transforms as T
import torch
import cv2
import os

workers = 0 if os.name == 'nt' else 4

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

detector = MTCNNDetector(
    image_size=160, margin=0, min_face_size=20,
    thresholds=[0.8, 0.9, 0.95], resize_factor=0.709, post_process=True,
    device=device
)


matcher = res_encoder(1, 128)


DB_PATH = "matchers/datasets/CS2003-dataset/CS2003-DB"
BEST_MODEL_PATH = "matchers/output/resnet/best_model.pth"
matcher.load_state_dict(torch.load(BEST_MODEL_PATH))
resize = 96
input_transform = T.Compose([
    T.ToTensor(),
    T.Resize((resize, resize)),
    T.Grayscale(num_output_channels=1)
])


c = 0
capture = cv2.VideoCapture(0)
while capture.isOpened():
    no_throw, frame = capture.read()
    if no_throw:
        time_frame = 1
        if c % time_frame == 0:
            boxes, _, landmarks = detector.detect(frame[:, :, ::-1].copy(), landmarks=True)
            if boxes is not None:
                for box in boxes:
                    x1, y1, x2, y2 = box
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color=(255, 255, 0), thickness=2)
                    cv2.imwrite("temp.jpg", frame)
                    ans = encoder_predict(matcher, DB_PATH, "temp.jpg", input_transform, euclidean_distance)
                    cv2.putText(frame, ans, (int(x1), int(y1)), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)
            if landmarks is not None:
                for point in landmarks:
                    for po in point:
                        cv2.circle(frame, (int(po[0]), int(po[1])), 1, (0, 0, 255), 4)
        c += 1
        cv2.imshow('realtime detection test', frame)
        if cv2.waitKey(1) & 0xFF == ord('e'):
            break
    else:
        break
capture.release()
cv2.destroyAllWindows()
