from __future__ import division
import argparse
import torch
import os
import cv2
import json
import numpy as np
from PIL import Image
from common.utils import BBox, drawLandmark, drawLandmark_multiple
from models.basenet import MobileNet_GDConv
from src import face_detect

SCENE_NAMES = [
    # '1-HIT_Canteen_frames',
    '2-OCT_Bay_frames',
    '7-Shenzhennorth_Station_frames',
    '3-Xili_Crossroad_frames',
    '4-Nanshan_I_Park_frames',
    '8-Xili_Pedestrian_Street_frames',
    '5-Primary_School_frames',
    '9-Tsinghuasz_Basketball_Court_frames',
    '10-Xinzhongguan_frames',
    '12-Tsinghua_Zhulou_frames',
    '13-Tsinghua_Xicao_frames',
    '11-Tsinghua_ceremony_frames',
    '16-Xili_frames',
    # "14-Dongmen_frames",
    # "15-Huaqiangbei_frames"
]
ROOT_DIR = '/media/luvision/新加卷1/PANDA_each_person_data'
MIN_WIDTH_THRES = 200

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

    for scene in SCENE_NAMES:
        print('Processing scene: ', scene)
        scene_root = os.path.join(ROOT_DIR, scene)
        with open(os.path.join(scene_root, scene+'.json'), 'r') as load_f:
            batch_labels = json.load(load_f)
        save_dict = {}
        for pid in batch_labels.keys():
            if batch_labels[pid]:
                save_dict[pid] = []
                for bid, batch in enumerate(batch_labels[pid]):
                    buffer = []
                    images = batch['images']
                    face_oris = batch['face_oris']
                    # process each image
                    for iid, img_name in enumerate(images):
                        if face_oris[iid] not in ['right_back', 'back', 'left_back']:
                            buffer.append([])
                            continue
                        img_path = os.path.join(scene_root, pid, img_name)
                        srcimg = cv2.imread(img_path)
                        height, width, _ = srcimg.shape
                        if width < MIN_WIDTH_THRES:
                            buffer.append([])
                            continue
                        img = srcimg[0: int(height / 3), int(width * 10 / 110): int(width * 100 / 110)].copy()
                        height, width, _ = img.shape
                        # perform face detection using MTCNN
                        print('Generating face landmark on scene: {} person: {} batch: {} image: {}'.format(scene, pid, bid, iid))
                        image = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                        faces = face_detect(image)
                        results = landmark_detect(faces, width, height, img, out_size, model)
                        if not results:
                            buffer.append([])
                        else:
                            idx = find_min_dist(results)
                            new_bbox, landmark = results[idx][1:]
                            buffer.append(landmark.tolist())
                            # img = drawLandmark_multiple(img, new_bbox, landmark)
                        # cv2.imwrite(os.path.join('results', os.path.basename(pid + '-' + img_name)), img)
                    save_dict[pid].append(buffer)
        json_string = json.dumps(save_dict, indent=2)
        with open(os.path.join(scene_root, scene + '_face_landmarks.json'), "w") as f:
            f.write(json_string)


if __name__ == '__main__':
    main()
