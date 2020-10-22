# Face alignment demo
# Uses MTCNN as face detector
# Cunjian Chen (ccunjian@gmail.com)
from __future__ import division
import argparse
import torch
import os
import cv2
import numpy as np
from PIL import Image
from common.utils import BBox, drawLandmark, drawLandmark_multiple
from models.basenet import MobileNet_GDConv
from src import face_detect
import glob
import time
parser = argparse.ArgumentParser(description='PyTorch face landmark')
parser.add_argument(
    '-c',
    '--checkpoint',
    default='checkpoint/mobilenet_224_model_best_gdconv_external.pth.tar',
    type=str,
    metavar='PATH',
    help='path to save checkpoint (default: checkpoint)')

args = parser.parse_args()
mean = np.asarray([0.485, 0.456, 0.406])
std = np.asarray([0.229, 0.224, 0.225])

if torch.cuda.is_available():
    def map_location(storage, loc): return storage.cuda()
else:
    map_location = 'cpu'


def load_model():
    model = MobileNet_GDConv(136)
    model = torch.nn.DataParallel(model)
    checkpoint = torch.load(args.checkpoint, map_location=map_location)
    model.load_state_dict(checkpoint['state_dict'])
    return model


def find_min_dist(results):
    return_id = 0
    min_dist = np.inf
    for idx, temp in enumerate(results):
        center_dist = temp[0]
        if center_dist < min_dist:
            min_dist = center_dist
            return_id = idx
    return return_id


def landmark_detect(faces, width, height, img, out_size, model):
    if len(faces) == 0:
        return []
    center = width / 2, height / 2
    return_list = []
    for k, face in enumerate(faces):
        if len(face) < 4:
            continue
        x1, y1, x2, y2 = face[0:4]
        face_center = (x1 + x2) / 2, (y1 + y2) / 2
        center_dist = np.sqrt((center[0] - face_center[0]) ** 2 + (center[1] - face_center[1]) ** 2)
        w = x2 - x1 + 1
        h = y2 - y1 + 1
        size = int(min([w, h]) * 1.2)
        cx = x1 + w // 2
        cy = y1 + h // 2
        x1 = cx - size // 2
        x2 = x1 + size
        y1 = cy - size // 2
        y2 = y1 + size

        dx = max(0, -x1)
        dy = max(0, -y1)
        x1 = max(0, x1)
        y1 = max(0, y1)

        edx = max(0, x2 - width)
        edy = max(0, y2 - height)
        x2 = min(width, x2)
        y2 = min(height, y2)
        new_bbox = list(map(int, [x1, x2, y1, y2]))
        new_bbox = BBox(new_bbox)
        cropped = img[new_bbox.top:new_bbox.bottom,
                  new_bbox.left:new_bbox.right]
        if dx > 0 or dy > 0 or edx > 0 or edy > 0:
            cropped = cv2.copyMakeBorder(cropped, int(dy), int(
                edy), int(dx), int(edx), cv2.BORDER_CONSTANT, 0)
        cropped_face = cv2.resize(cropped, (out_size, out_size))

        if cropped_face.shape[0] <= 0 or cropped_face.shape[1] <= 0:
            continue
        test_face = cropped_face.copy()
        test_face = test_face / 255.0
        test_face = (test_face - mean) / std
        test_face = test_face.transpose((2, 0, 1))
        test_face = test_face.reshape((1,) + test_face.shape)
        input = torch.from_numpy(test_face).float()
        input = torch.autograd.Variable(input)
        # start = time.time()
        landmark = model(input).cpu().data.numpy()
        # end = time.time()
        # print('Time: {:.6f}s.'.format(end - start))
        landmark = landmark.reshape(-1, 2)
        landmark = new_bbox.reprojectLandmark(landmark)
        return_list.append([center_dist, new_bbox, landmark])

    return return_list


def main():
    out_size = 224
    model = load_model()
    model = model.eval()
    filenames = glob.glob("samples/12--Group/*.jpg")
    for imgname in filenames:
        print(imgname)
        srcimg = cv2.imread(imgname)
        height, width, _ = srcimg.shape
        img = srcimg[0: int(height / 3), int(width * 10 / 110): int(width * 100 / 110)].copy()
        height, width, _ = img.shape

        # perform face detection using MTCNN
        image = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        faces = face_detect(image)
        results = landmark_detect(faces, width, height, img, out_size, model)
        if not results:
            print('NO face is detected!')
        else:
            idx = find_min_dist(results)
            new_bbox, landmark = results[idx][1:]
            img = drawLandmark_multiple(img, new_bbox, landmark)
        cv2.imwrite(os.path.join('results', os.path.basename(imgname)), img)


if __name__ == '__main__':
    main()
