from torchvision import transforms
import matplotlib.pyplot as plt
# import face_recognition
import numpy as np
import shutil
import visdom
import torch
import math
import cv2
import os


# def detect_faces(image, detector=face_recognition.face_locations) -> list:
#     locations = detector(image)
#     faces = []
#     for location in locations:
#         top, right, bottom, left = location
#         faces.append(image[top:bottom, left:right])
#     return faces
#
#
# def image_resize_for_resnet(image):
#     return cv2.resize(image, (256, 256))
#
#
# def rotate_image(image, angle):
#     (h, w) = image.shape[:2]
#     (cx, cy) = (w // 2, h // 2)
#     m = cv2.getRotationMatrix2D((cx, cy), -angle, 1.0)
#     cos = np.abs(m[0, 0])
#     sin = np.abs(m[0, 1])
#     nw = int(h * sin + w * cos)
#     nh = int(h * cos + w * sin)
#     m[0, 2] += (nw / 2) - cx
#     m[1, 2] += (nh / 2) - cy
#     return cv2.warpAffine(image, m, (nw, nh))
#
#
# def image_rotate_and_get_face(image, angle = 270):
#     return detect_faces(rotate_image(image, angle))
#
#
# def image_manipulate_and_transport(src, dst, predicate, begin, postfix):
#     if not os.path.exists(dst):
#         os.mkdir(dst)
#     i = begin
#     for root, dirs, files in os.walk(src):
#         if ".DS_Store" in files:
#             files.remove(".DS_Store")
#         for name in files:
#             faces = predicate(cv2.imread(src + "//" + name))
#             for face in faces:
#                 try:
#                     cv2.imwrite(dst + "//" + str(i) + postfix, face)
#                     i += 1
#                 except Exception:
#                     continue
#     return i
#
#
# def add_label_to_image(path: str, label: str, begin: int = 1) -> None:
#     filelist = os.listdir(path)
#     i = begin
#     for item in filelist:
#         if item.endswith(".jpg"):
#             src = os.path.join(os.path.abspath(path), item)
#             dst = os.path.join(os.path.abspath(path), "train_" + str(i) + "_" + label + ".jpg")
#             try:
#                 os.rename(src, dst)
#                 i += 1
#             except:
#                 continue
#
#
# def dispatch_data(src: str, names: list) -> None:
#     for name in names:
#         if not os.path.exists(name):
#             os.mkdir(name)
#     for root, dirs, files in os.walk(src):
#         for name in files:
#             words = name.split('_')
#             if len(words) < 3:
#                 continue
#             word = words[2].split('.')
#             offset = ord(word[0]) - ord('0')
#             shutil.move(src + "//" + name, names[offset])


def transfer_state_dict_if(pretrained_dict, model_dict, predicate):
    state_dict = {}
    for k, v in pretrained_dict.items():
        if k in model_dict.keys() and predicate(k):
            state_dict[k] = v
    return state_dict


def transfer_model_if(pretrained_model, model, predicate):
    pretrained_dict = pretrained_model.state_dict()
    model_dict = model.state_dict()
    updated_dict = transfer_state_dict_if(pretrained_dict, model_dict, predicate)
    model_dict.update(updated_dict)
    model.load_state_dict(model_dict)
    return model


def isnt_extra(s: str):
    word = s.split(".")[0]
    extra = word.split("_")
    return not extra[0] == "extra"


def dummy(s):
    return True


def augment(image):
    transform = transforms.Compose(
        [   transforms.RandomResizedCrop(96, scale=(0.08, 0.1)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(0.4, 0.4, 0.2, 0.1),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([transforms.GaussianBlur(kernel_size=23)], p=0.5),
            transforms.RandomSolarize(threshold=128, p=0.1), #128 taken from BYOL
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ]
    )
    image_a = transform(image)
    image_b = transform(image)
    return image_a, image_b


def adjust_learning_rate(args, optimizer, loader, step):
    max_steps = args.epoch * len(loader)
    warmup_steps = 10 * len(loader)
    base_lr = args.batch_size / 256
    if step < warmup_steps:
        lr = base_lr * step / warmup_steps
    else:
        step -= warmup_steps
        max_steps -= warmup_steps
        q = 0.5 * (1 + math.cos(math.pi * step / max_steps))
        end_lr = base_lr * 0.001
        lr = base_lr * q + end_lr * (1 - q)
    optimizer.param_groups[0]['lr'] = lr * args.lr


def show_loss(path, name, step=1):
    with open(path, "r") as f:
        data = f.read()
    data = data.split(" ")[:-1]
    x = np.linspace(1, len(data) + 1, len(data)) * step
    y = []
    for i in range(len(data)):
        y.append(float(data[i]))
    vis = visdom.Visdom(env='loss')
    vis.line(X=x, Y=y, win=name, opts={'title': name, "xlabel": "epoch", "ylabel": name})


def select_and_transfer(src, dst, begin):
    if not os.path.exists(dst):
        os.mkdir(dst)
    i = begin
    for root, dirs, files in os.walk(src):
        if ".DS_Store" in files:
            files.remove(".DS_Store")
        if "frontal" in root:
            os.mkdir(os.path.join(dst, str(i)))
            for _root, _dirs, _files in os.walk(root):
                if ".DS_Store" in _files:
                    _files.remove(".DS_Store")
                for name in _files:
                    shutil.move(os.path.join(root, name), os.path.join(dst, str(i)))
            i += 1


def encoder_predict(encoder, db_path, target_path, transformer, distance):
    target = torch.stack([transformer(cv2.imread(target_path))], 0)
    maximum = 0
    tot = 0
    hot = "unknown"
    for root, dirs, files in os.walk(db_path):
        files.remove(".DS_Store")
        for name in files:
            candidate = torch.stack([transformer(cv2.imread(db_path + "//" + name))], 0)
            dis = distance(encoder(candidate), encoder(target))
            tot += math.exp(-dis)
        for name in files:
            candidate = torch.stack([transformer(cv2.imread(os.path.join(db_path, name)))], 0)
            dis = distance(encoder(candidate), encoder(target))
            probability = math.exp(-dis) / tot
            if maximum < probability:
                maximum = probability
                hot = name
    return hot.split(".")[0]


def read_file(src):
    file = open(src, "r")
    line = file.readline()
    batch = 1
    data = []
    while line:
        data.append(eval(line))
        line = file.readline()
        batch += 1
    file.close()
    return data, batch


def visualize_file(src, dst, title, xlable, ylable):
    if not os.path.exists(src):
        raise
    data, batch = read_file(src)
    plt.figure(figsize=(18, 12), dpi=(100))
    plt.xlabel(xlable)
    plt.ylabel(ylable)
    plt.plot(range(1, batch, 1), data)
    plt.savefig(os.path.join(dst, title + ".png"))
    plt.show()


def mean(vec):
    s = 0
    for item in vec:
        s += item
    return s / len(vec)


def get_mean(src, batch_size):
    if not os.path.exists(src):
        raise
    data, batch = read_file(src)
    ret = []
    for i in range(batch // batch_size):
        subvec = data[i*batch_size: (i+1)*batch_size]
        ret.append(mean(subvec))
    return ret


def visualize(data, length, xlable, ylable, title, dst):
    plt.figure(figsize=(18, 12), dpi=(100))
    plt.xlabel(xlable)
    plt.ylabel(ylable)
    plt.plot(range(1, length, 1), data)
    plt.savefig(os.path.join(dst, title + ".png"))
    plt.show()


def visualize_multi(data_list, title, xlabel, ylabel, dst):
    fig, plot = plt.subplots(figsize=(10, 8))
    plot.set_xlabel(xlabel)
    plot.set_ylabel(ylabel)
    plot.set_title(title)
    for item in data_list:
        data, label = item
        length = len(data)
        plot.plot(range(0, length, 1), data, label=label)
    plot.legend()
    plt.savefig(os.path.join(dst, title + ".png"))
    plt.show()


if __name__ == "__main__":
    mean_validate_losses = []
    mean_training_losses = []
    mean_validate_acc = []
    mean_training_acc = []
    src = os.path.join("..", "output")
    for root, dirs, files in os.walk(src):
        if len(root.split("/")) == 2:
            continue
        for _root, _dirs, _files in os.walk(root):
            network_name = _root.split("/")[2]
            for name in _files:
                data_src = os.path.join(_root, name)
                if name == "train_acc.txt":
                    mean_training_acc.append([get_mean(data_src, 100), network_name])
                elif name == "val_acc.txt":
                    mean_validate_acc.append([get_mean(data_src, 100), network_name])
                elif name == "train_loss.txt":
                    mean_training_losses.append([get_mean(data_src, 100), network_name])
                elif name == "val_loss.txt":
                    mean_validate_losses.append([get_mean(data_src, 100), network_name])
    dst = os.path.join("..", "comparasion")
    if not os.path.exists(dst):
        os.mkdir(dst)
    visualize_multi(mean_training_acc, "training accuracy", "epoch", "accuracy", dst)
    visualize_multi(mean_validate_acc, "validate accuracy", "epoch", "accuracy", dst)
    visualize_multi(mean_training_losses, "training losses", "epoch", "loss", dst)
    visualize_multi(mean_validate_losses, "validate losses", "epoch", "loss", dst)
