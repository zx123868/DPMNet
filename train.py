from torchstat import stat
import argparse
import os
from torchsummary import summary
from collections import OrderedDict
from glob import glob
from thop import profile
import pandas as pd
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
import yaml
#from albumentations.augmentations import transforms
import albumentations as albu
#from albumentations.augmentations.transforms import Flip

from albumentations.core.composition import Compose, OneOf
from sklearn.model_selection import train_test_split
from torch.optim import lr_scheduler
from tqdm import tqdm
from albumentations import RandomRotate90,Resize
import archs
import losses
from dataset import Dataset
from metrics import iou_score
from utils import AverageMeter, str2bool
from archs import DPMNet


ARCH_NAMES = archs.__all__
LOSS_NAMES = losses.__all__
LOSS_NAMES.append('BCEWithLogitsLoss')



def parse_args():
    parser = argparse.ArgumentParser()#创建一个解析器，使用 argparse 的第一步是创建一个 ArgumentParser 对象：ArgumentParser 对象包含将命令行解析成 Python 数据类型所需的全部信息。
    #大多数对 ArgumentParser 构造方法的调用都会使用 description= 关键字参数。这个参数简要描述这个程度做什么以及怎么做。在帮助消息中，这个描述会显示在命令行用法字符串和各种参数的帮助消息之间。
    parser.add_argument('--name', default=None,
                        help='model name: (default: arch+timestamp)')
    parser.add_argument('--epochs', default=400, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('-b', '--batch_size', default=4, type=int,
                        metavar='N', help='mini-batch size (default: 16)')
    #给一个 ArgumentParser 添加程序参数信息是通过调用 add_argument() 方法完成的。通常，这些调用指定 ArgumentParser 如何获取命令行字符串并将其转换为对象。这些信息在 parse_args() 调用时被存储和使用。
    # model
    parser.add_argument('--arch', '-a', metavar='ARCH', default='DPMNet')
    parser.add_argument('--deep_supervision', default=False, type=str2bool)
    parser.add_argument('--input_channels', default=3, type=int,
                        help='input channels')
    parser.add_argument('--num_classes', default=1, type=int,
                        help='number of classes')
    parser.add_argument('--input_w', default=256, type=int,
                        help='image width')
    parser.add_argument('--input_h', default=256, type=int,
                        help='image height')
    
    # loss
    parser.add_argument('--loss', default='BCEDiceLoss',
                        choices=LOSS_NAMES,
                        help='loss: ' +
                        ' | '.join(LOSS_NAMES) +
                        ' (default: BCEDiceLoss)')
    
    # dataset

    #这里的isic是要改成自己的数据集名字
    #parser.add_argument('--dataset', default='isic',
     #                   help='dataset name')
    #parser.add_argument('--dataset', default='Kvasir-SEG',
     #                   help='dataset name')
    parser.add_argument('--dataset', default='colon',#CVC_ClinicDB,#Kvasir-SEG,msd
                        help='dataset name')
    parser.add_argument('--img_ext', default='.png',
                        help='image file extension')
    parser.add_argument('--mask_ext', default='.png',
                        help='mask file extension')

    # optimizer
    parser.add_argument('--optimizer', default='Adam',
                        choices=['Adam', 'SGD'],
                        help='loss: ' +
                        ' | '.join(['Adam', 'SGD']) +
                        ' (default: Adam)') #default=Adam
    parser.add_argument('--lr', '--learning_rate', default=1e-4, type=float,
                        metavar='LR', help='initial learning rate')#default=0.001
    parser.add_argument('--momentum', default=0.9, type=float,
                        help='momentum')
    parser.add_argument('--weight_decay', default=1e-4, type=float,
                        help='weight decay')
    #权重衰退： 一种有效的正则化方法。在每次参数更新时，引入一个衰减系数。
    parser.add_argument('--nesterov', default=False, type=str2bool,
                        help='nesterov')#自然梯度法，一种优化算法
    #训练时的学习率调整：optimizer和scheduler。可以理解为optimizer是指定使用哪个优化器，scheduler是对优化器的学习率进行调整
    # scheduler
    parser.add_argument('--scheduler', default='CosineAnnealingLR',
                        choices=['CosineAnnealingLR', 'ReduceLROnPlateau', 'MultiStepLR', 'ConstantLR'])
    #default = CosineAnnealingLR
    parser.add_argument('--min_lr', default=1e-5, type=float,
                        help='minimum learning rate')
    parser.add_argument('--factor', default=0.1, type=float)
    parser.add_argument('--patience', default=2, type=int)
    parser.add_argument('--milestones', default='1,2', type=str)
    parser.add_argument('--gamma', default=2/3, type=float)
    parser.add_argument('--early_stopping', default=-1, type=int,
                        metavar='N', help='early stopping (default: -1)')
    parser.add_argument('--cfg', type=str, metavar="FILE", help='path to config file', )

    parser.add_argument('--num_workers', default=4, type=int)#default = 4






    config = parser.parse_args()

    return config

# args = parser.parse_args()
#ArgumentParser 通过 parse_args() 方法解析参数。它将检查命令行，把每个参数转换为适当的类型然后调用相应的操作。
def train(config, train_loader, model, criterion, optimizer):
    avg_meters = {'loss': AverageMeter(),
                  'iou': AverageMeter()}
    model.train()

    pbar = tqdm(total=len(train_loader))
    #Tqdm 是 Python 进度条库，可以在 Python 长循环中添加一个进度提示信息
    for input, target, _ in train_loader:
        input = input.cuda()
        target = target.cuda()

        # compute output
        #深度监督表示什么意思？
        # parser.add_argument('--deep_supervision', default=False, type=str2bool)
        if config['deep_supervision']:
            outputs = model(input)
            loss = 0
            for output in outputs:
                loss += criterion(output, target)
            loss /= len(outputs)
            iou,dice = iou_score(outputs[-1], target)
        else:
            output = model(input)
            loss = criterion(output, target)
            iou,dice = iou_score(output, target)

        # compute gradient and do optimizing step
        #用pytorch训练模型时，通常会在遍历epochs的过程中依次用到optimizer.zero_grad(),loss.backward()和optimizer.step()三个函数
        #总得来说，这三个函数的作用是先将梯度归零（optimizer.zero_grad()），然后反向传播计算得到每个参数的梯度值（loss.backward()），最后通过梯度下降执行一步参数更新（optimizer.step()）
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        avg_meters['loss'].update(loss.item(), input.size(0))
        avg_meters['iou'].update(iou, input.size(0))

        postfix = OrderedDict([ #OrderedDict听名字就知道他是 按照有序插入顺序存储 的有序字典。 除此之外还可根据key， val进行排序。
            ('loss', avg_meters['loss'].avg),
            ('iou', avg_meters['iou'].avg),
        ])
        #进度条传入的是不可迭代对象，手动更新
        pbar.set_postfix(postfix)
        pbar.update(1) # # 手动更新，默认参数n=1，每update一次，进度+n
    pbar.close()

    return OrderedDict([('loss', avg_meters['loss'].avg),
                        ('iou', avg_meters['iou'].avg)])


def validate(config, val_loader, model, criterion):
    avg_meters = {'loss': AverageMeter(),
                  'iou': AverageMeter(),
                   'dice': AverageMeter()}

    # switch to evaluate mode
    model.eval()
    #主要是针对model 在训练时和评价时不同的 Batch Normalization 和 Dropout 方法模式。
    #with 语句适用于对资源进行访问的场合，确保不管使用过程中是否发生异常都会执行必要的“清理”操作，释放资源，比如文件使用后自动关闭／线程中锁的自动获取和释放等。使用with的话，能够减少冗长，还能自动处理上下文环境产生的异常。
    '''
    当requires_grad设置为False时,反向传播时就不会自动求导了，因此大大节约了显存或者说内存。
    with torch.no_grad的作用:在该模块下，所有计算得出的tensor的requires_grad都自动设置为False。
    即使一个tensor（命名为x）的requires_grad = True，在with torch.no_grad计算，由x得到的新tensor（命名为w-标量）requires_grad也为False，且grad_fn也为None,即不会对w求导。   
    被with torch.no_grad()包住的代码，不用跟踪反向梯度计算
    '''

    with torch.no_grad():
        pbar = tqdm(total=len(val_loader))
        for input, target, _ in val_loader:

            input = input.cuda()
            target = target.cuda()

            # compute output
            if config['deep_supervision']:
                outputs = model(input)

                loss = 0
                for output in outputs:
                    loss += criterion(output, target)
                loss /= len(outputs)
                iou,dice = iou_score(outputs[-1], target)
            else:
                output = model(input)
                loss = criterion(output, target)
                iou,dice = iou_score(output, target)

            avg_meters['loss'].update(loss.item(), input.size(0))
            avg_meters['iou'].update(iou, input.size(0))
            avg_meters['dice'].update(dice, input.size(0))

            postfix = OrderedDict([
                ('loss', avg_meters['loss'].avg),
                ('iou', avg_meters['iou'].avg),
                ('dice', avg_meters['dice'].avg)
            ])
            pbar.set_postfix(postfix)
            pbar.update(1)
        pbar.close()

    return OrderedDict([('loss', avg_meters['loss'].avg),
                        ('iou', avg_meters['iou'].avg),
                        ('dice', avg_meters['dice'].avg)])
    #这里的排序是把分别按loss，iou，dice按大小排序吗

def main():
    #vars () 函数返回对象object的属性和属性值的字典对象。
    '''
    print(vars(Runoob))
    {'a': 1, '__module__': '__main__', '__doc__': None}
    '''
    config = vars(parse_args())
    #不懂这一段name代表的是什么名字，arch是什么，好像是模型的名字
    if config['name'] is None:
        if config['deep_supervision']:
            config['name'] = '%s_%s_wDS' % (config['dataset'], config['arch'])
        else:
            config['name'] = '%s_%s_woDS' % (config['dataset'], config['arch'])
    #os.makedirs用来创建多层目录（单层请用os.mkdir)
    os.makedirs('models/%s' % config['name'], exist_ok=True)
    #分割线
    print('-' * 20)
    #把参数值打印出来
    for key in config:
        print('%s: %s' % (key, config[key]))
    print('-' * 20)
    #yml的适用场景：
    '''
    脚本语言：由于实现简单，解析成本很低，YAML 特别适合在脚本语言中使用
    序列化： YAML是由宿主语言数据类型直转，的比较适合做序列化。
    配置文件：写 YAML 要比写 XML 快得多(无需关注标签或引号)，并且比 INI 文档功能更强。由于兼容性问题，不同语言间的数据流转建议不要用 YAML。
    '''
    #   config.yaml文件读入后是一个字典，可用来配置程序中的相关参数；
    with open('models/%s/config.yml' % config['name'], 'w') as f:
        yaml.dump(config, f)
    #个人理解yaml.dump()函数，就是将yaml文件一次性全部写入你创建的文件:'models/%s/config.yml' % config['name'], 'w'。
    # define loss function (criterion)
    if config['loss'] == 'BCEWithLogitsLoss':
        criterion = nn.BCEWithLogitsLoss()
    else:
        criterion = losses.__dict__[config['loss']]()

    cudnn.benchmark = True
    #arch?linux?
    # create model
    model = archs.__dict__[config['arch']](config['num_classes'],
                                           config['input_channels'],
                                           config['deep_supervision']).cuda()
    '''
    x = torch.randn(4, 3, 256, 256)

    # Check if the model is on the GPU, if not, move it to the GPU
    # model = model.to('cuda:0')

    flops, params = profile(model, (x,))
    print('flops: ', flops, 'params: ', params)
    print('flops: %.2f G, params: %.2f M' % (flops / 1000000000.0, params / 1000000.0))
    '''
    #stat(model,(3,256,256))
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    #summary(model.to(device),input_size=(3,256,256))
    #输出是一个字典，这个字典包含了该对象里面所有的属性，可以通过这种方式来访问对象的属性

    #parameters是存储神经网络中间产数矩阵的变量
    params = filter(lambda p: p.requires_grad, model.parameters())
    '''
    lambda作为一个表达式，定义了一个匿名函数，上例的代码x为入口参数，x+1为函数体，
    >>> foo = [2, 18, 9, 22, 17, 24, 8, 12, 27]
    >>> print filter(lambda x: x % 3 == 0, foo)
    [18, 9, 24, 12, 27]
    '''
    if config['optimizer'] == 'Adam':
        optimizer = optim.Adam(
            params, lr=config['lr'], weight_decay=config['weight_decay'])
    elif config['optimizer'] == 'SGD':
        optimizer = optim.SGD(params, lr=config['lr'], momentum=config['momentum'],
                              nesterov=config['nesterov'], weight_decay=config['weight_decay'])
    else:
        raise NotImplementedError
    #这里的LR是learning rate学习率不是逻辑回归
    #==后面代表四种不同的学习率
    if config['scheduler'] == 'CosineAnnealingLR':
        scheduler = lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=config['epochs'], eta_min=config['min_lr'])
    elif config['scheduler'] == 'ReduceLROnPlateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, factor=config['factor'], patience=config['patience'],
                                                   verbose=1, min_lr=config['min_lr'])
    elif config['scheduler'] == 'MultiStepLR':
        scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[int(e) for e in config['milestones'].split(',')], gamma=config['gamma'])
    elif config['scheduler'] == 'ConstantLR':
        scheduler = None
    else:
        raise NotImplementedError

    # Data loading code
    #这两行怎么改
    #img_ids = glob(os.path.join('inputs', config['dataset'], 'images', '*' + config['img_ext']))
    #img_ids = [os.path.splitext(os.path.basename(p))[0] for p in img_ids]
    img_ids = glob(os.path.join('inputs', config['dataset'], 'images', '*' + config['img_ext']))
    print(img_ids)
    mask_ids = glob(os.path.join('inputs', config['dataset'], 'masks', '0','*' + config['mask_ext']))
    print(mask_ids)
    #inputs\\CVC-ClinicDB\\images\\1.png', 'inputs\\CVC-ClinicDB\\images\\10.png', 'inputs\\CVC-ClinicDB\\images\\100.png',
    img_ids = [os.path.splitext(os.path.basename(p))[0] for p in img_ids]
    print(img_ids)
    #'1', '10', '100', '101', '102',
    #下面这行代码相当于循环遍历上述文件夹下的图片
    train_img_ids, val_img_ids = train_test_split(img_ids, test_size=0.2, random_state=100)

    train_transform = Compose([
        RandomRotate90(),
        #transforms.Flip(),
        albu.Flip(),
        Resize(config['input_h'], config['input_w']),
        #transforms.Normalize(),
        albu.Normalize(),
    ])

    val_transform = Compose([
        Resize(config['input_h'], config['input_w']),
        #transforms.Normalize(),
        albu.Normalize(),
    ])

    train_dataset = Dataset(
        img_ids=train_img_ids,
        img_dir=os.path.join('inputs', config['dataset'], 'images'),
        mask_dir=os.path.join('inputs', config['dataset'], 'masks'),
        img_ext=config['img_ext'],
        mask_ext=config['mask_ext'],
        num_classes=config['num_classes'],
        transform=train_transform)
    val_dataset = Dataset(
        img_ids=val_img_ids,
        img_dir=os.path.join('inputs', config['dataset'], 'images'),
        mask_dir=os.path.join('inputs', config['dataset'], 'masks'),
        img_ext=config['img_ext'],
        mask_ext=config['mask_ext'],
        num_classes=config['num_classes'],
        transform=val_transform)
    # image, label, _ = val_dataset[1]

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=config['num_workers'],
        drop_last=True)
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config['num_workers'],
        drop_last=False)



    log = OrderedDict([
        ('epoch', []),
        ('lr', []),
        ('loss', []),
        ('iou', []),
        ('val_loss', []),
        ('val_iou', []),
        ('val_dice', []),
    ])

    best_iou = 0
    trigger = 0


    for epoch in range(config['epochs']):
        print('Epoch [%d/%d]' % (epoch, config['epochs']))

        # train for one epoch
        train_log = train(config, train_loader, model, criterion, optimizer)
        # evaluate on validation set
        val_log = validate(config, val_loader, model, criterion)

        if config['scheduler'] == 'CosineAnnealingLR':
            scheduler.step()
        elif config['scheduler'] == 'ReduceLROnPlateau':
            scheduler.step(val_log['loss'])

        print('loss %.4f - iou %.4f - val_loss %.4f - val_iou %.4f'
              % (train_log['loss'], train_log['iou'], val_log['loss'], val_log['iou']))

        log['epoch'].append(epoch)
        log['lr'].append(config['lr'])
        log['loss'].append(train_log['loss'])
        log['iou'].append(train_log['iou'])
        log['val_loss'].append(val_log['loss'])
        log['val_iou'].append(val_log['iou'])
        log['val_dice'].append(val_log['dice'])

        pd.DataFrame(log).to_csv('models/%s/log.csv' %
                                 config['name'], index=False)

        trigger += 1

        if val_log['iou'] > best_iou:
            torch.save(model.state_dict(), 'models/%s/model.pth' %
                       config['name'])
            best_iou = val_log['iou']
            print("=> saved best model")
            trigger = 0

        # early stopping
        if config['early_stopping'] >= 0 and trigger >= config['early_stopping']:
            print("=> early stopping")
            break

        torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
