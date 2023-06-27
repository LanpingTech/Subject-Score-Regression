import os
import time
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import logging
import colorlog
import logging.handlers
from tqdm import tqdm
from dataset import DynamicDataset
from model import DCN1
from torch.utils.data import DataLoader

os.environ['CUDA_VISIBLE_DEVICES'] = '2'

BATCH_SIZE = 64
LEARNING_RATE = 0.005
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# 日志
def getLogger():
    if not os.path.exists("logs"):
        os.mkdir("logs")
    logger = logging.getLogger()
    console = logging.StreamHandler()
    logger.setLevel(logging.INFO)
    console.setLevel(logging.DEBUG)
    handler = logging.handlers.TimedRotatingFileHandler('logs/log.log', when='midnight', interval=1, backupCount=7,
                                                        encoding='utf-8')
    handler.suffix = "_%Y%m%d.log"
    formatter_file = logging.Formatter(
        '%(asctime)s - %(thread)s - %(filename)s[%(lineno)d] - %(levelname)s: %(message)s')
    formatter_console = colorlog.ColoredFormatter(
        fmt='%(log_color)s%(asctime)s - %(thread)s - %(filename)s[%(lineno)d] - %(levelname)s: %(message)s',
        log_colors={
            'DEBUG': 'white',
            'INFO': 'cyan',
            'WARNING': 'yellow',
            'ERROR': 'red',
            'CRITICAL': 'bold_red',
        }
    )
    handler.setFormatter(formatter_file)
    console.setFormatter(formatter_console)
    logger.addHandler(handler)
    logger.addHandler(console)
    return logger


# 日志
logger = getLogger()

dataset_conf = {
        'features': {
            '年级': {
                'name': '年级',
                'index': 0,
                'code': '年级',
                'isnum': False,
                'dict': {
                    '高一': 0,
                    '高二': 1,
                    '高三': 2
                },
                'dict_rv': {
                    0: '高一',
                    1: '高二',
                    2: '高三'
                }
            },
            '语文': {
                'name': '语文',
                'index': 1,
                'code': '语文',
                'isnum': True
            },
            '数学': {
                'name': '数学',
                'index': 2,
                'code': '数学',
                'isnum': True
            },
            '英语': {
                'name': '英语',
                'index': 3,
                'code': '英语',
                'isnum': True
            },
            '物理': {
                'name': '物理',
                'index': 4,
                'code': '物理',
                'isnum': True
            },
            '化学': {
                'name': '化学',
                'index': 5,
                'code': '化学',
                'isnum': True
            },
            '生物': {
                'name': '生物',
                'index': 6,
                'code': '生物',
                'isnum': True
            },
            # '历史': {
            #     'name': '历史',
            #     'index': 7,
            #     'code': '历史',
            #     'isnum': True
            # }
        },
        'features_cate': ['年级'],
        'features_num': ['语文', '数学', '英语', '物理', '化学', '生物'
                         # , '历史'
                         ],
        'labels': {
            '综合': {
                'name': '综合',
                'index': 8,
                'code': '综合',
                'isnum': True
            },
            '等级': {
                'name': '等级',
                'index': 9,
                'code': '等级',
                'isnum': False,
                'dict': {
                    '甲': 0,
                    '乙': 1,
                    '丙': 2,
                    '丁': 3
                },
                'dict_rv': {
                    0: '甲',
                    1: '乙',
                    2: '丙',
                    3: '丁'
                }
            }},
        'labels_cate': ['等级'],
        'labels_num': ['综合']
    }


def train():

    logger.info(f"开始执行训练任务")
    # 获取模型配置
    EPOCHS = 0
    module_path = 'dynamic_model.pth'
    # 获取数据集配置
    logger.info(f"数据集配置加载完成")
    logger.info(f"模型输入特征：{[*dataset_conf['features']]}")
    logger.info(f"模型输出特征：{[*dataset_conf['labels']]}")
    # 获取数据
    dataset_data = pd.read_csv('data.csv')
    logger.info(f"训练数据集加载完成，数据集大小：{len(dataset_data)}")

    # 初始化数据集
    spends = time.time()
    dataset = DynamicDataset(dataset_conf, dataset_data)
    logger.info(f"训练数据集初始化完成，耗时{time.time() - spends:.3f}秒")

    # spends = time.time()
    # tmp = dataset[0]
    train_size = int(len(dataset) * 0.8)
    train_data, test_data = torch.utils.data.random_split(dataset, [train_size, len(dataset) - train_size])
    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False)

    model = DCN1(dataset.in_dim_cate(), dataset.in_dim_num(), dataset.out_dim_cate(), dataset.out_dim_num()).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    criterion_cate = torch.nn.CrossEntropyLoss()
    criterion_num = torch.nn.MSELoss()
    best_loss = np.inf
    # logger.info("训练准备耗时{}".format(time.time() - spends))

    spends = time.time()

    epoch = 0
    best_keep = 0
    best_max = 50
    if EPOCHS <= 0:
        logger.info(f"迭代次数上限[{EPOCHS}]设置无效，连续[{best_max}]次迭代无更低损失则结束训练")
    else:
        logger.info(f"迭代次数上限[{EPOCHS}]设置成功，迭代次数达到上限则结束训练")
    while (EPOCHS <= 0 and best_keep < best_max) or epoch <= EPOCHS:
        epoch += 1

        model.train()
        train_loss = 0
        # pbar = tqdm(train_loader)
        # for feature_cate, feature_num, label_cate, label_num in pbar:
        for feature_cate, feature_num, label_cate, label_num in train_loader:
            feature_cate = feature_cate.to(DEVICE).long() if len(feature_cate) > 0 else feature_cate
            # feature_num = nn.functional.normalize(feature_num.to(DEVICE).float())
            feature_num = feature_num.to(DEVICE).float()
            label_cate = label_cate.to(DEVICE).long().T if len(label_cate) > 0 else label_cate
            label_num = label_num.to(DEVICE).float().T if len(label_num) > 0 else label_num

            optimizer.zero_grad()
            tensor_input = torch.cat([feature_cate, feature_num], dim=1) if len(feature_cate) > 0 else feature_num
            tensor_output = model(tensor_input)

            loss_arr = []
            len_cate = len(label_cate)
            len_num = len(label_num)
            for i in range(0, len_cate + len_num):
                if i < len_cate:
                    loss_arr.append(criterion_cate(tensor_output[i], label_cate[i]))
                else:
                    loss_arr.append(criterion_num(tensor_output[i], label_num[i - len_cate].unsqueeze(1)))
            loss = sum(loss_arr)

            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            # pbar.set_postfix(loss=loss.item())
        train_loss /= len(train_loader)

        model.eval()
        test_loss = 0
        with torch.no_grad():
            for feature_cate, feature_num, label_cate, label_num in test_loader:
                feature_cate = feature_cate.to(DEVICE).long() if len(feature_cate) > 0 else feature_cate
                # feature_num = nn.functional.normalize(feature_num.to(DEVICE).float())
                feature_num = feature_num.to(DEVICE).float()
                label_cate = label_cate.to(DEVICE).long().T if len(label_cate) > 0 else label_cate
                label_num = label_num.to(DEVICE).float().T if len(label_num) > 0 else label_num

                tensor_input = torch.cat([feature_cate, feature_num], dim=1) if len(feature_cate) > 0 else feature_num
                tensor_output = model(tensor_input)

                loss_arr = []
                len_cate = len(label_cate)
                len_num = len(label_num)
                for i in range(0, len_cate + len_num):
                    if i < len_cate:
                        loss_arr.append(criterion_cate(tensor_output[i], label_cate[i]))
                    else:
                        loss_arr.append(criterion_num(tensor_output[i], label_num[i - len_cate].unsqueeze(1)))
                loss = sum(loss_arr)

                test_loss += loss.item()
        test_loss /= len(test_loader)

        save_best = test_loss < best_loss
        if save_best:
            best_loss = test_loss
            torch.save(model, module_path)
            best_keep = 0
        else:
            best_keep += 1

        logger.info(
            f"第{epoch}次迭代，训练损失: {train_loss:.3f}，测试损失: {test_loss:.3f}{'，最低损失模型已保存' if save_best else ''}")

    logger.info(f"训练任务完成，耗时{time.time() - spends:.3f}秒")

    pass


def predict():
    logger.info(f"开始执行预测任务")
    module_path = 'dynamic_model.pth'
    # 获取数据集配置
    logger.info(f"数据集配置加载完成")
    logger.info(f"模型输入特征：{[*dataset_conf['features']]}")
    logger.info(f"模型输出特征：{[*dataset_conf['labels']]}")
    dataset_data = pd.read_csv('data.csv')
    logger.info(f"预测数据集加载完成，数据集大小：{len(dataset_data)}")

    # 初始化数据集
    spends = time.time()
    dataset = DynamicDataset(dataset_conf, dataset_data, predict=True)
    logger.info(f"预测数据集初始化完成，耗时{time.time() - spends:.3f}秒")

    spends = time.time()

    result_predicts = []
    result_rows = []
    model = torch.load(module_path)
    logger.info(f"模型文件加载完成，开始预测......")
    model.eval()
    with torch.no_grad():
        predict_loader = DataLoader(dataset, batch_size=1, shuffle=True)
        for idx, (feature_cate, feature_num) in enumerate(predict_loader):
            feature_cate = feature_cate.to(DEVICE).long() if len(feature_cate) > 0 else feature_cate
            # feature_num = nn.functional.normalize(feature_num.to(DEVICE).float())
            feature_num = feature_num.to(DEVICE).float()

            tensor_input = torch.cat([feature_cate, feature_num], dim=1) if len(feature_cate) > 0 else feature_num
            tensor_output = model(tensor_input)

            predict_map = {}
            for cate in dataset_conf['labels_cate']:
                i = len(predict_map)
                label_dict_rv = dataset_conf['labels'][cate]['dict_rv']
                predict_map[cate] = label_dict_rv[tensor_output[i].argmax().item()]
            for num in dataset_conf['labels_num']:
                i = len(predict_map)
                predict_map[num] = tensor_output[i].item()
            predict_arr = []
            for label in dataset_conf['labels']:
                predict_arr.append(predict_map[label])

            predict_source = dataset.get_features_data(idx)
            logger.info(f"({idx + 1}/{len(dataset)}) 输入：{predict_source}，预测：{predict_arr}")

            result_predicts.append(predict_arr)
            result_rows.append(predict_source + predict_arr)

    result_predicts = {
        'titles': [*dataset_conf['labels']],
        'datas': result_predicts
    }

    result_rows = {
        'titles': [*dataset_conf['features']] + [*dataset_conf['labels']],
        'datas': result_rows
    }

    data_frame = pd.DataFrame(columns=result_rows['titles'])
    for i in range(len(result_rows['datas'])):
        data_frame.loc[i] = result_rows['datas'][i]
    file_path = 'data_predict.csv'
    if file_path.upper().endswith('.CSV'):
        data_frame.to_csv(file_path, index=False, encoding='utf-8')
    else:
        data_frame.to_excel(file_path, sheet_name='Sheet1', index=False)

    logger.info(f"预测任务完成，耗时{time.time() - spends:.3f}秒")
    pass


if __name__ == '__main__':
    train()
    predict()
