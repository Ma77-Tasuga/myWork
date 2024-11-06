import torch



dataset2parse:str = ''

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')