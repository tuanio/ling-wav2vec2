from utils import greedy_decode
import torch

a = torch.LongTensor([[0, 1, 2, 3], [5, 6, 7, 8]])

b = torch.Tensor([0, 1, 2, 3, 4])

c = torch.rand(4, 10, 100)

print(greedy_decode(a))
print(greedy_decode(b))
print(greedy_decode(c))

from dataset import SupervisedDataset

train_set = SupervisedDataset(
    "data/splitted_data/train.json"
)

print(train_set[0])
print(len(train_set))
