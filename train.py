import os
os.environ['CUDA_VISIBLE_DEVICES'] = '2'
import numpy as np
import pandas as pd
import torch

from dataset import ScoreDataset
from torch.utils.data import DataLoader
from model import DCN

from tqdm import tqdm

# hyperparameters
EPOCHS = 100
BATCH_SIZE = 64
LEARNING_RATE = 0.001
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# read data
all_data = ScoreDataset('data.csv')
train_size = int(len(all_data) * 0.8)
train_data, test_data = torch.utils.data.random_split(all_data, [train_size, len(all_data) - train_size])
train_loader = DataLoader(train_data, batch_size = BATCH_SIZE, shuffle = True)
test_loader = DataLoader(test_data, batch_size = BATCH_SIZE, shuffle = False)

# initialize model
model = DCN(all_data.get_indim()).to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr = LEARNING_RATE)
criterion1 = torch.nn.MSELoss()
criterion2 = torch.nn.CrossEntropyLoss()

# train
best_loss = np.inf
for epoch in range(EPOCHS):
    model.train()
    train_loss = 0
    pbar = tqdm(train_loader)
    for grade, scores, rank, score in pbar:
        grade = grade.to(DEVICE).long()
        scores = scores.to(DEVICE).float()
        rank = rank.to(DEVICE).long()
        score = score.to(DEVICE).float().unsqueeze(1)
        optimizer.zero_grad()
        input_tensor = torch.cat([grade.unsqueeze(1), scores], dim = 1)
        pred_score, pred_rank = model(input_tensor)
        loss1 = criterion1(pred_score, score)
        loss2 = criterion2(pred_rank, rank)
        loss = loss1 + loss2
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        pbar.set_postfix(loss=loss.item(), rank_loss=loss2.item(), score_loss=loss1.item())
    train_loss /= len(train_loader)
    print(f'Epoch {epoch + 1} | Train Loss: {train_loss:.3f}')

    model.eval()
    test_loss = 0
    with torch.no_grad():
        for grade, scores, rank, score in test_loader:
            grade = grade.to(DEVICE).long()
            scores = scores.to(DEVICE).float()
            rank = rank.to(DEVICE).long()
            score = score.to(DEVICE).float().unsqueeze(1)
            input_tensor = torch.cat([grade.unsqueeze(1), scores], dim = 1)
            pred_score, pred_rank = model(input_tensor)
            loss1 = criterion1(pred_score, score)
            loss2 = criterion2(pred_rank, rank)
            loss = loss1 + loss2
            test_loss += loss.item()
    test_loss /= len(test_loader)
    print(f'Epoch {epoch + 1} | Test Loss: {test_loss:.3f}')

    if test_loss < best_loss:
        best_loss = test_loss
        torch.save(model.state_dict(), 'best_model.pth')
        print('Saved best model!')

# save model
torch.save(model.state_dict(), 'model.pth')




