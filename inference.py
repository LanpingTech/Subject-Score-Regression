import torch
from model import DCN

# load model
model = DCN()
model.load_state_dict(torch.load('best_model.pth'))
model.eval()

# inference
def inference(grade, scores):
    grade = torch.tensor([grade]).long()
    scores = torch.tensor([scores]).float()
    input_tensor = torch.cat([grade.unsqueeze(1), scores], dim = 1)
    pred_score, pred_rank = model(input_tensor)
    return pred_score.item(), pred_rank.argmax().item()