import time
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from tqdm import tqdm
from dataset import DynamicDataset
from model import DCN1
from torch.utils.data import DataLoader

if __name__ == '__main__':
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
                }
            }},
        'labels_cate': ['等级'],
        'labels_num': ['综合']
    }

    dataset_data = pd.read_csv('data.csv')

    spends = time.time()
    EPOCHS = 100
    BATCH_SIZE = 64
    LEARNING_RATE = 0.001
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    dataset = DynamicDataset(dataset_conf, dataset_data)
    print(f"初始化数据集耗时{time.time() - spends:.3f}秒")

    train_size = int(len(dataset) * 0.8)
    train_data, test_data = torch.utils.data.random_split(dataset, [train_size, len(dataset) - train_size])
    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False)

    model = DCN1(dataset.dim_input(), dataset.dim_cate(), dataset.dim_num()).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    criterion_cate = torch.nn.CrossEntropyLoss()
    criterion_num = torch.nn.MSELoss()

    spends = time.time()
    best_loss = np.inf
    for epoch in range(1, 1 + EPOCHS):
        model.train()
        train_loss = 0
        pbar = tqdm(train_loader)
        for feature_cate, feature_num, label_cate, label_num in pbar:
            # for feature_cate, feature_num, label_cate, label_num in train_loader:
            feature_cate = feature_cate.to(DEVICE).long()
            feature_num = nn.functional.normalize(feature_num.to(DEVICE).float())
            label_cate = label_cate.to(DEVICE).long().T
            label_num = label_num.to(DEVICE).float().T

            optimizer.zero_grad()
            tensor_input = torch.cat([feature_cate, feature_num], dim=1)
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
            pbar.set_postfix(loss=loss.item())
        train_loss /= len(train_loader)

        model.eval()
        test_loss = 0
        with torch.no_grad():
            for feature_cate, feature_num, label_cate, label_num in test_loader:
                feature_cate = feature_cate.to(DEVICE).long()
                feature_num = nn.functional.normalize(feature_num.to(DEVICE).float())
                label_cate = label_cate.to(DEVICE).long().T
                label_num = label_num.to(DEVICE).float().T

                tensor_input = torch.cat([feature_cate, feature_num], dim=1)
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

        print(f'Epoch {epoch} | Train Loss: {train_loss:.3f} | Test Loss: {test_loss:.3f}')

        if test_loss < best_loss:
            best_loss = test_loss
            torch.save(model.state_dict(), 'dynamic_model.pth')
            print('Saved best model!')
    print(f"训练耗时{time.time() - spends:.3f}秒")

    pass
