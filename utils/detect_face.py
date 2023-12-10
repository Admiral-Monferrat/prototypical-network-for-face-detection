from torchvision.ops.boxes import batched_nms
from utils.utils import *


def detect_face(images, minimum_size, pnet, rnet, onet, threshold, resize_factor, device):
    pnet.eval()
    rnet.eval()
    onet.eval()
    if not isinstance(images, (list, tuple)):
        images = [images]
    if any(img.size != images[0].size for img in images):
        raise Exception('detect_face()')
    images = np.stack([np.uint8(img) for img in images])
    images = torch.as_tensor(images, device=device)
    images = images.permute(0, 3, 1, 2).float()
    batch_size = images.shape[0]
    h, w = images.shape[2:]
    m = 12.0 / minimum_size
    minimum_length = min(h, w)
    minimum_length = minimum_length * m
    current_scale = m
    scales = []
    while minimum_length >= 12:
        scales.append(current_scale)
        current_scale = current_scale * resize_factor
        minimum_length = minimum_length * resize_factor
    boxes = []
    image_indexes = []
    all_indexes = []
    num_images = 0
    for scale in scales:
        resized_image = imgresample(images, (int(h * scale + 1), int(w * scale + 1)))
        resized_image = (resized_image - 127.5) * 0.0078125
        conf, reg = pnet(resized_image)
        boxes_scale, image_inds_scale = generate_bounding_box(conf[0, :], reg, scale, threshold[0])
        boxes.append(boxes_scale)
        image_indexes.append(image_inds_scale)
        all_indexes.append(num_images + image_inds_scale)
        num_images += batch_size
    boxes = torch.cat(boxes, dim=0)
    image_indexes = torch.cat(image_indexes, dim=0).cpu()
    all_indexes = torch.cat(all_indexes, dim=0)
    picked = batched_nms(boxes[:, :4], boxes[:, 4], all_indexes, 0.5)
    boxes, image_indexes = boxes[picked], image_indexes[picked]
    picked = batched_nms(boxes[:, :4], boxes[:, 4], image_indexes, 0.7)
    boxes, image_indexes = boxes[picked], image_indexes[picked]
    regw = boxes[:, 2] - boxes[:, 0]
    regh = boxes[:, 3] - boxes[:, 1]
    qq1 = boxes[:, 0] + boxes[:, 5] * regw
    qq2 = boxes[:, 1] + boxes[:, 6] * regh
    qq3 = boxes[:, 2] + boxes[:, 7] * regw
    qq4 = boxes[:, 3] + boxes[:, 8] * regh
    boxes = torch.stack([qq1, qq2, qq3, qq4, boxes[:, 4]]).permute(1, 0)
    boxes = to_square(boxes)
    y, ey, x, ex = pad(boxes, w, h)
    if len(boxes) > 0:
        resized_images = []
        for k in range(len(y)):
            if ey[k] > (y[k] - 1) and ex[k] > (x[k] - 1):
                img = images[image_indexes[k], :, (y[k] - 1):ey[k], (x[k] - 1):ex[k]].unsqueeze(0)
                resized_images.append(imgresample(img, (24, 24)))
        resized_images = torch.cat(resized_images, dim=0)
        resized_images = (resized_images - 127.5) / 127.5
        out = rnet(resized_images)
        print(resized_images.shape)
        out0 = out[0].permute(1, 0)
        out1 = out[1].permute(1, 0)
        score = out0[0, :]
        ipass = score > threshold[1]
        boxes = torch.cat((boxes[ipass, :4], score[ipass].unsqueeze(1)), dim=1)  # (len(ipass), 5)
        image_indexes = image_indexes[ipass]
        mv = out1[:, ipass].permute(1, 0)
        picked = batched_nms(boxes[:, :4], boxes[:, 4], image_indexes, 0.7)
        boxes, image_indexes, mv = boxes[picked], image_indexes[picked], mv[picked]
        boxes = bbreg(boxes, mv)
        boxes = to_square(boxes)
    points = torch.zeros(0, 5, 2, device=device)
    if len(boxes) > 0:
        y, ey, x, ex = pad(boxes, w, h)
        resized_images = []
        for k in range(len(y)):
            if ey[k] > (y[k] - 1) and ex[k] > (x[k] - 1):
                img = images[image_indexes[k], :, (y[k] - 1):ey[k], (x[k] - 1):ex[k]].unsqueeze(0)
                resized_images.append(imgresample(img, (48, 48)))
        resized_images = torch.cat(resized_images, dim=0)
        resized_images = (resized_images - 127.5) / 127.5
        out = onet(resized_images)
        out0 = out[0].permute(1, 0)
        out1 = out[1].permute(1, 0)
        out2 = out[2].permute(1, 0)
        score = out0[0, :]
        points = out2
        ipass = score > threshold[2]
        points = points[:, ipass]
        boxes = torch.cat((boxes[ipass, :4], score[ipass].unsqueeze(1)), dim=1)
        image_indexes = image_indexes[ipass]
        mv = out1[:, ipass].permute(1, 0)
        w_i = boxes[:, 2] - boxes[:, 0] + 1
        h_i = boxes[:, 3] - boxes[:, 1] + 1
        points = torch.stack((points[0, :], points[2, :], points[4, :], points[6, :], points[8, :], points[1, :],
                             points[3, :], points[5, :], points[7, :], points[9, :]), dim=0)
        points_x = w_i.repeat(5, 1) * points[:5, :] + boxes[:, 0].repeat(5, 1) - 1
        points_y = h_i.repeat(5, 1) * points[5:, :] + boxes[:, 1].repeat(5, 1) - 1
        points = torch.stack((points_x, points_y)).permute(2, 1, 0)
        boxes = bbreg(boxes, mv)
        picked = batched_nms_numpy(boxes[:, :4], boxes[:, 4], image_indexes, 0.7, 'Min')
        boxes, image_indexes, points = boxes[picked], image_indexes[picked], points[picked]
    boxes = boxes.cpu().numpy()
    points = points.cpu().numpy()
    result_boxes = []
    result_points = []
    for b in range(batch_size):
        idx = np.where(image_indexes == b)
        result_boxes.append(boxes[idx].copy())
        result_points.append(points[idx].copy())
    result_boxes, result_points = np.array(result_boxes), np.array(result_points)
    return result_boxes, result_points
