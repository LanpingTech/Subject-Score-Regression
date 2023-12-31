import torch
import pandas as pd
from torch.utils.data import Dataset

GRADE_DICT = {'高一': 0, '高二': 1, '高三': 2}
RANK_DICT = {'甲': 0, '乙': 1, '丙': 2, '丁': 3}


class ScoreDataset(Dataset):
    def __init__(self, data_path):
        self.data = pd.read_csv(data_path)
        self.data[['语文', '数学', '英语', '物理', '化学', '生物', '历史', '综合']] /= 100

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        grade = GRADE_DICT[row['年级']]
        scores = row[['语文', '数学', '英语', '物理', '化学', '生物', '历史']].values.astype(float)
        rank = RANK_DICT[row['等级']]
        score = row['综合']
        return grade, scores, rank, score
    
    def get_indim(self):
        return self.data.shape[1] - 2


class DynamicDataset(Dataset):

    def __init__(self, conf, data, predict=False):
        self.conf = conf
        self.data = data
        self.predict = predict

        self.item_list = []
        for idx in range(len(self.data)):
            # 行数据
            row_data = self.data.iloc[idx]

            feature_cate = []
            for cate in self.conf['features_cate']:
                feature = self.conf['features'][cate]
                feature_cate.append(feature['dict'][row_data[feature['code']]])
            feature_cate = torch.LongTensor(feature_cate) if len(feature_cate) > 0 else []
            feature_num = row_data[self.conf['features_num']].values.astype(float)
            if predict:
                self.item_list.append((feature_cate, feature_num))
            else:
                label_cate = []
                for cate in self.conf['labels_cate']:
                    label = self.conf['labels'][cate]
                    label_cate.append(label['dict'][row_data[label['code']]])
                label_cate = torch.LongTensor(label_cate) if len(label_cate) > 0 else []
                label_num = row_data[self.conf['labels_num']].values.astype(float)
                self.item_list.append((feature_cate, feature_num, label_cate, label_num))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.item_list[idx]

    def dim_input(self):
        return len(self.conf['features'])

    def in_dim_cate(self):
        arr = []
        for cate in self.conf['features_cate']:
            label = self.conf['features'][cate]
            arr.append(len(label['dict']))
        return arr

    def in_dim_num(self):
        return len(self.conf['features_num'])

    def out_dim_cate(self):
        arr = []
        for cate in self.conf['labels_cate']:
            label = self.conf['labels'][cate]
            arr.append(len(label['dict']))
        return arr

    def out_dim_num(self):
        return len(self.conf['labels_num'])

    def get_features_titles(self):
        return [*self.conf['features']]

    def get_features_data(self, idx):
        row_data = self.data.iloc[idx]
        return row_data[self.get_features_titles()].tolist()
