import torch
from torch_geometric.data import Data, Batch
#from torch_geometric.data import Data, DataLoader
from torch.utils.data import DataLoader

# 示例数据集
data_list = [
    Data(x=torch.tensor([[1], [2], [3]]), y=torch.tensor([1]), z=torch.tensor([[1, 2, 3], [1, 2, 3]])),
    Data(x=torch.tensor([[4], [5], [6]]), y=torch.tensor([0]), z=torch.tensor([[4, 5], [4, 5]])),
]

# 自定义 collate 函数
def custom_collate(batch):
    batch_data = {}
    
    # 获取所有属性的 keys
    keys = batch[0].keys
    
    for key in keys:
        values = [getattr(data, key) for data in batch]
        
        # 如果属性是张量并且大小一致，进行拼接
        if isinstance(values[0], torch.Tensor):
            try:
                batch_data[key] = torch.cat(values, dim=0)
            except RuntimeError:
                # 如果大小不一致，返回为列表
                batch_data[key] = values
        else:
            batch_data[key] = values

    return Data(**batch_data)

# 使用 torch_geometric 的 DataLoader，并传入自定义的 collate 函数
loader = DataLoader(data_list, batch_size=2, collate_fn=custom_collate)

# 迭代 DataLoader
for batch_data in loader:
    print(batch_data)
