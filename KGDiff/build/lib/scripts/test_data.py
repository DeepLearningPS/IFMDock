from torch_geometric.data import Data, Batch
#from torch.utils.data import DataLoader
from torch_geometric.loader import DataLoader
import torch

# 示例数据集
data_list = [
    Data(x=torch.tensor([[1], [2], [3]]), y=torch.tensor([1]), z=set(torch.tensor([[1, 2, 3], [1, 2, 3]]))),
    Data(x=torch.tensor([[4], [5], [6]]), y=torch.tensor([0]), z=set(torch.tensor([[4, 5], [4, 5], [4, 5]]))),
    Data(x=torch.tensor([[4], [5], [6]]), y=torch.tensor([0]), z=set(torch.tensor([[4, 5], [4, 5], [4, 5]]))),
    Data(x=torch.tensor([[1], [2], [3]]), y=torch.tensor([1]), z=set(torch.tensor([[1, 2, 3], [1, 2, 3]]))),
]


'''
data_list = [
    Data(x=torch.tensor([[1], [2], [3]]), y=torch.tensor([1]), z='a'),
    Data(x=torch.tensor([[4], [5], [6]]), y=torch.tensor([0]), z='b'),
    Data(x=torch.tensor([[4], [5], [6]]), y=torch.tensor([0]), z='c'),
    Data(x=torch.tensor([[1], [2], [3]]), y=torch.tensor([1]), z='d'),
]
'''

# 自定义collate函数
def custom_collate(batch):
    # 在这里指定不想连接的分量（比如 'z'）
    exclude_keys = ['z']

    # 初始化用于存储连接数据的字典
    batch_data = {}
    
    keys = batch[0].keys #使用pyg2.1.0，更新的版本，则不行
    # 处理每个属性
    for key in keys:
        if key in exclude_keys:
            # 对于需要排除的分量，收集成列表
            batch_data[key] = [getattr(data, key) for data in batch]
        else:
            # 对于需要连接的分量，使用默认的方式进行连接
            batch_data[key] = torch.cat([getattr(data, key) for data in batch], dim=0)

    return batch_data



# 自定义collate函数
def custom_collate2(batch):
    # 在这里指定不想连接的分量（比如 'z'）
    exclude_keys = ['z']
    

    # 初始化用于存储连接数据的字典
    batch_data = {}
    
    # 处理每个属性
    for key in batch[0].keys:
        if key in exclude_keys:
            # 对于需要排除的分量，收集成列表
            batch_data[key] = [getattr(data, key) for data in batch]
        else:
            # 对于需要连接的分量，使用默认的方式进行连接
            batch_data[key] = torch.cat([getattr(data, key) for data in batch], dim=0)

    return batch_data

exclude_keys = []

# 使用DataLoader并传入自定义的collate函数
loader = DataLoader(data_list, batch_size=2, collate_fn=custom_collate2, exclude_keys = exclude_keys) 
#在PYG dataloader中collate_fn参数是被删除的，所以不起作用，而exclude_keys成了关键参数，因此如果想不连接某些数据对象，只需要提供exclude_keys即可

# 迭代DataLoader
for batch_data_ in loader:
    batch_data = batch_data_.cuda()
    print(batch_data)
    zz = batch_data.z
    print('batch_data.x:', batch_data.x)
    print('zz:', zz)
    zzz = []
    #[{tensor([1, 2, 3]), tensor([1, 2, 3])}, {tensor([4, 5]), tensor([4, 5]), tensor([4, 5])}]
    for i in zz:
        print('i:', i)
        ii = torch.stack(list(i), dim = 0) #集合转list
        zzz.append(ii.cuda())
    
    print('zzz:', zzz)

    '''
    zzz: [tensor([[1, 2, 3],
        [1, 2, 3]]), tensor([[4, 5],
        [4, 5],
        [4, 5]])]
    '''
