import os
import sys
root_path = os.getcwd()
print(root_path)
sys.path.append("/Users/robert/Desktop/Python语言程序设计/FSL for face recognition的副本/")
print(sys.path)
from prototypical_loss_fn import prototypical_loss as loss_update
from random_batch_sampler import RandomizedBatchSampler
from training_options import get_parser
import torchvision.transforms as T
from torchvision import datasets
import torch.utils.data as data
from matchers.networks import facenet
from tqdm import tqdm
import numpy as np
import torch
from matchers.networks.resnet import res_encoder


def init_seed(opt):
    torch.cuda.cudnn_enabled = False
    np.random.seed(opt.manual_seed)
    torch.manual_seed(opt.manual_seed)
    torch.cuda.manual_seed(opt.manual_seed)


def init_dataset(args):
    resize = 96
    input_transform = T.Compose([
        T.ToTensor(),
        T.Resize((resize, resize)),
        T.Grayscale(num_output_channels=1)
    ])
    train_dataset = datasets.ImageFolder(os.path.join(args.dataset_root, "train"), transform=input_transform)
    test_dataset = datasets.ImageFolder(os.path.join(args.dataset_root, "test"), transform=input_transform)
    n_classes = len(np.unique(train_dataset.targets))
    if n_classes < args.classes_per_it_tr or n_classes < args.classes_per_it_val:
        raise(Exception('There are not enough classes in the dataset in order ' +
                        'to satisfy the chosen classes_per_it. Decrease the ' +
                        'classes_per_it_{tr/val} option and try again.'))
    return train_dataset, test_dataset


def init_sampler(opt, labels, mode):
    if 'train' in mode:
        classes_per_it = opt.classes_per_it_tr
        num_samples = opt.num_support_tr + opt.num_query_tr
    else:
        classes_per_it = opt.classes_per_it_val
        num_samples = opt.num_support_val + opt.num_query_val
    return RandomizedBatchSampler(labels=labels, classes_per_it=classes_per_it, num_samples=num_samples, batch_size=opt.batch_size)


def init_dataloader(opt, mode):
    _tr_data, test_data = init_dataset(opt)
    size = len(_tr_data)
    tr_size = int(size * 0.75)
    val_size = size - tr_size
    tr_data, val_data = data.random_split(_tr_data, [tr_size, val_size])
    tr_sampler = init_sampler(opt, tr_data.dataset.targets, "train")
    val_sampler = init_sampler(opt, val_data.dataset.targets, "test")
    test_sampler = init_sampler(opt, test_data.targets, "test")
    tr_loader = data.DataLoader(tr_data.dataset, batch_sampler=tr_sampler)
    val_loader = data.DataLoader(val_data.dataset, batch_sampler=val_sampler)
    test_loader = data.DataLoader(test_data, batch_sampler=test_sampler)
    return tr_loader, val_loader, test_loader


def init_protonet(opt):
    model = res_encoder(1, 128)
    return model


def init_optim(opt, model):
    return torch.optim.Adam(params=model.parameters(),
                            lr=opt.learning_rate)


def init_lr_scheduler(opt, optim):
    return torch.optim.lr_scheduler.StepLR(optimizer=optim,
                                           gamma=opt.lr_scheduler_gamma,
                                           step_size=opt.lr_scheduler_step)


def save_list_to_file(path, thelist):
    with open(path, 'w') as f:
        for item in thelist:
            f.write("%s\n" % item)


def train(opt, tr_dataloader, model, optim, lr_scheduler, val_dataloader=None):
    device = 'cuda:0' if torch.cuda.is_available() and opt.cuda else 'cpu'
    if val_dataloader is None:
        best_state = None
    train_loss = []
    train_acc = []
    val_loss = []
    val_acc = []
    best_acc = 0
    best_model_path = os.path.join(opt.experiment_root, 'best_model.pth')
    last_model_path = os.path.join(opt.experiment_root, 'last_model.pth')
    for epoch in range(opt.epochs):
        print('=== Epoch: {} ==='.format(epoch))
        tr_iter = iter(tr_dataloader)
        model.train()
        for batch in tqdm(tr_iter):
            optim.zero_grad()
            x, y = batch
            x, y = x.to(device), y.to(device)
            model_output = model(x)
            loss, acc = loss_update(model_output, target=y,
                                n_support=opt.num_support_tr)
            loss.backward()
            optim.step()
            train_loss.append(loss.item())
            train_acc.append(acc.item())
        avg_loss = np.mean(train_loss[-opt.batch_size:])
        avg_acc = np.mean(train_acc[-opt.batch_size:])
        print('Avg Train Loss: {}, Avg Train Acc: {}'.format(avg_loss, avg_acc))
        lr_scheduler.step()
        if val_dataloader is None:
            continue
        val_iter = iter(val_dataloader)
        model.eval()
        for batch in tqdm(val_iter):
            x, y = batch
            x, y = x.to(device), y.to(device)
            model_output = model(x)
            loss, acc = loss_update(model_output, target=y,
                                n_support=opt.num_support_val)
            val_loss.append(loss.item())
            val_acc.append(acc.item())
        avg_loss = np.mean(val_loss[-opt.batch_size:])
        avg_acc = np.mean(val_acc[-opt.batch_size:])
        postfix = ' (Best)' if avg_acc >= best_acc else ' (Best: {})'.format(
            best_acc)
        print('Avg Val Loss: {}, Avg Val Acc: {}{}'.format(
            avg_loss, avg_acc, postfix))
        if avg_acc >= best_acc:
            torch.save(model.state_dict(), best_model_path)
            best_acc = avg_acc
            best_state = model.state_dict()
        for name in ['train_loss', 'train_acc', 'val_loss', 'val_acc']:
            save_list_to_file(os.path.join(opt.experiment_root,
                                           name + '.txt'), locals()[name])
    torch.save(model.state_dict(), last_model_path)
    for name in ['train_loss', 'train_acc', 'val_loss', 'val_acc']:
        save_list_to_file(os.path.join(opt.experiment_root,
                                       name + '.txt'), locals()[name])
    return best_state, best_acc, train_loss, train_acc, val_loss, val_acc


def generalization_test(opt, test_dataloader, model):
    device = 'cuda:0' if torch.cuda.is_available() and opt.cuda else 'cpu'
    avg_acc = list()
    test_iter = iter(test_dataloader)
    for batch in tqdm(test_iter):
        x, y = batch
        x, y = x.to(device), y.to(device)
        model_output = model(x)
        loss, acc = loss_update(model_output, target=y, n_support=opt.num_support_val)
        avg_acc.append(acc.item())
    avg_acc = np.mean(avg_acc)
    print('Test Acc: {}'.format(avg_acc))
    return avg_acc


def main():
    options = get_parser().parse_args()
    if not os.path.exists(options.experiment_root):
        os.makedirs(options.experiment_root)
    if torch.cuda.is_available() and not options.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")
    init_seed(options)
    tr_dataloader, val_dataloader, test_dataloader = init_dataloader(options, "")
    model = init_protonet(options)
    model_path = os.path.join("..", "output//lenet//best_model.pth")
    optim = init_optim(options, model)
    lr_scheduler = init_lr_scheduler(options, optim)
    res = train(opt=options, tr_dataloader=tr_dataloader, val_dataloader=val_dataloader, model=model, optim=optim,
                lr_scheduler=lr_scheduler)
    best_state, best_acc, train_loss, train_acc, val_loss, val_acc = res
    model.load_state_dict(best_state)
    print('Testing with best model..')
    model_path = os.path.join("..", "output//protonet//best_model.pth")
    model.load_state_dict(torch.load(model_path))


if __name__ == '__main__':
    main()
