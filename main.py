import websockets
from argparse import ArgumentParser
import os
import time

import torch
from PIL import Image
from tensorboardX import SummaryWriter

from wcyy.models import create_model
from wcyy.utils.device import to_device, get_default_device
from predict import predict_img
from weather_request import get_weather_info, default_cfg
from camera import get_img
import config

default_classes = ['cloudy', 'haze', 'rainy', 'snow', 'sunny', 'thunder']


def weatype2idx(wea_type: str, classes=default_classes):
    d = {
        '晴': 'sunny',
        '多云': 'cloudy',
        '少云': 'sunny',
        '晴间多云': 'cloudy',
        '阴': 'cloudy',
        '阵雨': 'rainy',
        '强阵雨': 'rainy',
        '雷阵雨': 'rainy',
        '强雷阵雨': 'rainy',
        '雷阵雨伴有冰雹': 'rainy',
        '小雨': 'rainy',
        '中雨': 'rainy',
        '大雨': 'rainy',
        '极端降雨': 'rainy',
        '毛毛雨/细雨': 'rainy',
        '暴雨': 'rainy',
        '大暴雨': 'rainy',
        '特大暴雨': 'rainy',
        '冻雨': 'rainy',
        '小到中雨': 'rainy',
        '中到大雨': 'rainy',
        '大到暴雨': 'rainy',
        '暴雨到大暴雨': 'rainy',
        '大暴雨到特大暴雨': 'rainy',
        '雨': 'rainy',
        '小雪': 'snow',
        '中雪': 'snow',
        '大雪': 'snow',
        '暴雪': 'snow',
        '雨夹雪': 'rainy',
        '雨雪天气': 'rainy',
        '阵雨夹雪': 'rainy',
        '阵雪': 'snow',
        '小到中雪': 'snow',
        '中到大雪': 'snow',
        '大到暴雪': 'snow',
        '雪': 'Snow',
        '阵雨夹雪': 'rainy',
        '阵雪': 'snow',
        '薄雾': 'haze',
        '雾': 'haze',
        '霾': 'haze',
        '扬沙': 'haze',
        '浮尘': 'haze',
        '沙尘暴': 'haze',
        '强沙尘暴': 'haze',
        '浓雾': 'haze',
        '强浓雾': 'haze',
        '中度霾': 'haze',
        '重度霾': 'haze',
        '严重霾': 'haze',
        '大雾': 'haze',
        '特强浓雾': 'haze',
        '热': None,
        '冷': None,
        '未知': None
    }
    class2idx = {class_name: i for i, class_name in enumerate(classes)}
    wea_type = d.get(wea_type, None)
    return wea_type, class2idx.get(wea_type, -1)


def pil_loader(path: str) -> Image.Image:
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


def solve_one_img(img, classes, model, transform, device):
    ans = dict()

    out = predict_img(model, img, transform, device)
    out = torch.nn.Softmax(dim=1)(out)
    print(f'out: {out}')
    weather_info = get_weather_info(default_cfg)
    wea_type = weather_info['type']
    wea_type, wea_idx = weatype2idx(wea_type, classes)
    print(wea_type)
    if wea_type is not None:
        if wea_type in ['cloudy', 'haze']:
            out[0][wea_idx] += 0.7
        else:
            out[0][wea_idx] += 0.3
        print(f'out: {out}')
    _, indices = torch.sort(out, descending=True)
    print(indices)
    pred_idx = indices[0][0]
    if classes[pred_idx] in ['cloudy', 'haze'] and wea_type in ['cloudy', 'haze']:
        class2idx = {class_name: i for i, class_name in enumerate(classes)}
        pred_idx = class2idx[wea_type]
    ans['pred'] = classes[pred_idx]
    ans['forecast'] = weather_info

    return ans


def main():
    parser = ArgumentParser()
    parser.add_argument('--intervel', type=int, default=1)
    parser.add_argument('--camera_img_url',
                        type=str,
                        default='http://127.0.0.1:8000/image')
    parser.add_argument('--web_server_url',
                        type=str,
                        default='http://localhost:8000')
    parser.add_argument(
        '--cfg',
        type=str,
        default='efficientnet_b3a_e81_b16_tt7_vt2_explr_WeatherModel5_freeze')
    parser.add_argument('--gpu', type=str, default=None)
    parser.add_argument('--ckpt', type=str, default='model.ckpt')

    args = parser.parse_args()
    pass
    if args.gpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    cfg = getattr(config, args.cfg)
    cfg['ckpt'] = args.ckpt
    cfg['device'] = get_default_device()
    # cfg['device'] = torch.device('cpu')
    if cfg.get('exp_id', None) is None:
        cfg['exp_id'] = f'exp-{args.cfg}'

    # data meta-info
    classes = getattr(config, cfg['classes'])
    print(f'classes: {classes}')
    num_classes = len(classes)
    class2idx = {class_name: i for i, class_name in enumerate(classes)}
    print(class2idx)
    if cfg.get('exp_id', None) is None:
        cfg['exp_id'] = f'exp-{args.cfg}'

    # load model
    exps_root = 'runs'
    exp_id = cfg['exp_id']
    writer = SummaryWriter(os.path.join(exps_root, exp_id))
    model = to_device(create_model(cfg, num_classes), cfg['device'])
    model_ckpt = torch.load(os.path.join(writer.logdir, cfg['ckpt']))
    model.load_state_dict(model_ckpt)

    # load transform
    valid_transform = getattr(config, cfg['valid_transform'])

    # main loop
    # while True:
    # time.sleep(1)
    # img = get_img(args.camera_img_url)
    # result = predict_img(model, img, valid_transform, cfg['device'])

    # websocket

    # test
    img_path = './data/cloudy/cloudy_00009.jpg'
    img = pil_loader(img_path)
    data = solve_one_img(img, classes, model, valid_transform, cfg['device'])
    print(data)


if __name__ == '__main__':
    main()
