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
        grade = GRADE_DICT[row['年纪']]
        scores = row[['语文', '数学', '英语', '物理', '化学', '生物', '历史']].values.astype(float)
        rank = RANK_DICT[row['等级']]
        score = row['综合']
        return grade, scores, rank, score
    
    def get_indim(self):
        return self.data.shape[1] - 2
