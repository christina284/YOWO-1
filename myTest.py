# from utils import do_detect
# from opts import parse_opts
# from PIL import Image
# import matplotlib.pyplot as plt
# from model import YOWO
# from train import test
#
#
# import torch
# if __name__ == '__main__':
# image_path = '/mnt/terabyte/datasets/tiger'
# img = [cv2.imread(os.path.join(image_path, i)) for i in os.listdir(image_path)]
# img =[cv2.resize(i, (320, 240)) for i in img]
# opt = parse_opts()
# test(2)
from __future__ import print_function
import sys, os, time
import random
import math
import cv2 as cv
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torchvision import datasets, transforms

import dataset
from opts import parse_opts
from utils import *
from cfg import parse_cfg
from region_loss import RegionLoss

from model import YOWO


def bbox_iou(box1, box2, x1y1x2y2=True):
    if x1y1x2y2:
        mx = min(box1[0], box2[0])*1.0
        Mx = max(box1[2], box2[2])*1.0
        my = min(box1[1], box2[1])*1.0
        My = max(box1[3], box2[3])*1.0
        w1 = box1[2]*1.0 - box1[0]*1.0
        h1 = box1[3]*1.0 - box1[1]*1.0
        w2 = box2[2]*1.0 - box2[0]*1.0
        h2 = box2[3]*1.0 - box2[1]*1.0
    else:
        mx = min(float(box1[0] - box1[2] / 2.0), float(box2[0] - box2[2] / 2.0))
        Mx = max(float(box1[0] + box1[2] / 2.0), float(box2[0] + box2[2] / 2.0))
        my = min(float(box1[1] - box1[3] / 2.0), float(box2[1] - box2[3] / 2.0))
        My = max(float(box1[1] + box1[3] / 2.0), float(box2[1] + box2[3] / 2.0))
        w1 = box1[2]
        h1 = box1[3]
        w2 = box2[2]
        h2 = box2[3]
    uw = Mx - mx
    uh = My - my
    cw = w1 + w2 - uw
    ch = h1 + h2 - uh
    carea = 0
    if cw <= 0 or ch <= 0:
        return 0.0

    area1 = w1 * h1
    area2 = w2 * h2
    carea = cw * ch
    uarea = area1 + area2 - carea
    return carea / uarea


def nms(boxes, nms_thresh):
    if len(boxes) == 0:
        return boxes

    det_confs = torch.zeros(len(boxes))
    for i in range(len(boxes)):
        det_confs[i] = 1 - boxes[i][4]

    _, sortIds = torch.sort(det_confs)
    out_boxes = []
    for i in range(len(boxes)):
        box_i = boxes[sortIds[i]]
        if box_i[4] > 0:
            out_boxes.append(box_i)
            for j in range(i + 1, len(boxes)):
                box_j = boxes[sortIds[j]]
                if bbox_iou(box_i, box_j, x1y1x2y2=True) > nms_thresh:
                    box_j[4] = 0
    return out_boxes


def get_config():
    opt = parse_opts()  # Training settings
    dataset_use = opt.dataset  # which dataset to use
    datacfg = opt.data_cfg  # path for dataset of training and validation, e.g: cfg/ucf24.data
    cfgfile = opt.cfg_file  # path for cfg file, e.g: cfg/ucf24.cfg
    # assert dataset_use == 'ucf101-24' or dataset_use == 'jhmdb-21', 'invalid dataset'
    #
    # loss parameters
    loss_options = parse_cfg(cfgfile)[1]
    region_loss = RegionLoss()
    anchors = loss_options['anchors'].split(',')
    region_loss.anchors = [float(i) for i in anchors]
    region_loss.num_classes = int(loss_options['classes'])
    region_loss.num_anchors = int(loss_options['num'])

    return opt, region_loss


def load_model(opt, pretrained_path):
    seed = int(time.time())
    use_cuda = True
    gpus = '0'
    torch.manual_seed(seed)
    if use_cuda:
        os.environ['CUDA_VISIBLE_DEVICES'] = gpus
    torch.cuda.manual_seed(seed)

    # Create model
    model = YOWO(opt)
    model = model.cuda()
    model = nn.DataParallel(model, device_ids=None)  # in multi-gpu case
    model.seen = 0

    checkpoint = torch.load(pretrained_path)
    epoch = checkpoint['epoch']
    fscore = checkpoint['fscore']
    model.load_state_dict(checkpoint['state_dict'])

    return model, epoch, fscore


def infer(model, data, region_loss):
    num_classes = region_loss.num_classes
    anchors = region_loss.anchors
    num_anchors = region_loss.num_anchors
    conf_thresh_valid = 0.25
    nms_thresh = 0.7

    model.eval()

    data = data.cuda()
    res = []
    with torch.no_grad():
        output = model(data).data
        all_boxes = get_region_boxes(output, conf_thresh_valid, num_classes, anchors, num_anchors, 0, 1)

        for i in range(output.size(0)):
            boxes = all_boxes[i]
            boxes = nms(boxes, nms_thresh)

            for box in boxes:
                x1 = round(float(box[0] - box[2] / 2.0) * 320.0)
                y1 = round(float(box[1] - box[3] / 2.0) * 240.0)
                x2 = round(float(box[0] + box[2] / 2.0) * 320.0)
                y2 = round(float(box[1] + box[3] / 2.0) * 240.0)
                det_conf = float(box[4])

                for j in range((len(box) - 5) // 2):
                    cls_conf = float(box[5 + 2 * j].item())
                    if type(box[6 + 2 * j]) == torch.Tensor:
                        cls_id = int(box[6 + 2 * j].item())
                    else:
                        cls_id = int(box[6 + 2 * j])
                    prob = det_conf * cls_conf
                    res.append(str(int(box[6]) + 1) + ' ' + str(prob) + ' ' + str(x1) + ' ' + str(y1) + ' ' + str(
                        x2) + ' ' + str(y2))
    return res


def pre_process_image(images, clip_duration, input_shape=(224, 224)):
    # resize to (224,224)
    clip = [img.resize(input_shape) for img in images]
    # numpy to tensor
    op_transforms = transforms.Compose([transforms.ToTensor()])
    clip = [op_transforms(img) for img in clip]
    # change dimension
    clip = torch.cat(clip, 0).view((clip_duration, -1) + input_shape).permute(1, 0, 2, 3)
    # expand dimmension to (batch_size, channel, duration, w, h)
    clip = clip.unsqueeze(0)

    return clip


def post_process(images, bboxs):
    jhmdb_cls = ('', 'Basketball', 'BasketballDunk', 'Biking', 'CliffDiving', 'CricketBowling',
                   'Diving', 'Fencing', 'FloorGymnastics', 'GolfSwing', 'HorseRiding',
                   'IceDancing', 'LongJump', 'PoleVault', 'RopeClimbing', 'SalsaSpin',
                   'SkateBoarding', 'Skiing', 'Skijet', 'SoccerJuggling', 'Surfing',
                   'TennisSwing', 'TrampolineJumping', 'VolleyballSpiking', 'WalkingWithDog')
    conf_thresh = 0.01
    nms_thresh = 0.4

    proposals = []
    for i in range(len(bboxs)):
        line = bboxs[i]
        cls, score, x1, y1, x2, y2 = line.strip().split(' ')

        if float(score) < conf_thresh:
            continue

        a = 240/h
        b = 320/w
        proposals.append(
            [int(int(float(x1)) * a), int(int(float(y1)) * b), int(int(float(x2)) * a),
             int(int(float(y2)) * b), float(score),
             int(cls)])

    proposals = nms(proposals, nms_thresh)

    image = cv.cvtColor(np.asarray(images[-1], dtype=np.uint8), cv.COLOR_RGB2BGR)
    for proposal in proposals:
        x1, y1, x2, y2, score, cls = proposal
        cv.rectangle(image, (x1, y1), (x2, y2), (255, 255, 255), 2, cv.LINE_4)

        text = '[{:.2f}] {}'.format(score, jhmdb_cls[cls])
        font_type = 5
        font_size = 1
        line_szie = 1
        textsize = cv.getTextSize(text, font_type, font_size, line_szie)
        y1 = y1 - 10
        p1 = (x1, y1 - textsize[0][1])
        p2 = (x1 + textsize[0][0], y1 + textsize[1])
        cv.rectangle(image, p1, p2, (180, 238, 180), -1)
        cv.putText(image, text, (x1, y1), font_type, font_size, (255, 255, 255), line_szie, 1)


    return image


if __name__ == '__main__':
    duration = 16
    num_sample = 8
    pretrained_path = '/mnt/terabyte/chris_data/repos/YOWO/backup/yowo_ucf101-24_16f_best.pth'
    # video_path = '/mnt/terabyte/datasets/ucf24/videos/SkateBoarding/v_SkateBoarding_g25_c05.avi'

    # load parameters
    opt, region_loss = get_config()
    video_path = opt.video_path
    # load model
    model, epoch, fscore = load_model(opt, pretrained_path)
    # read video
    video = cv.VideoCapture(video_path)
    w = int(video.get(3))  # float
    h = int(video.get(4))  # float

    fourcc = cv.VideoWriter_fourcc(*'XVID')
    out = cv.VideoWriter('output.avi', fourcc, 30.0, (w, h))

    stack = []
    n = 0
    t0 = time.time()

    while (True):
        ret, frame = video.read()
        if not ret:
            break
        n += 1

        frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        frame = Image.fromarray(np.uint8(frame))
        stack.append(frame)

        if len(stack) == duration:
            # 1. preprocess images
            input_data = pre_process_image(stack, duration)
            # 2. YOWO detect action tube
            output_data = infer(model, input_data, region_loss)
            # 3. draw result to images
            result_img = post_process(stack, output_data)
            # 4. write to video
            out.write(result_img)

            for i in range(num_sample):
                stack.pop(0)

            t = time.time() - t0
            print('cost {:.2f}, {:.2f} FPS'.format(t, num_sample / t))
            t0 = time.time()

    out.release()
    video.release()