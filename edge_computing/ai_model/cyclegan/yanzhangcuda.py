# import torch
# print(torch.cuda.is_available())
#
#
# # import torch
# print(torch.version.cuda)

import torch
print(torch.cuda.device_count())  # 查看可用的 GPU 数量
for i in range(torch.cuda.device_count()):
    print(torch.cuda.get_device_name(i))  # 查看每个 GPU 的名称


