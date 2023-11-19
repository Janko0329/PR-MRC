import os
import torch

def save(model,optimizer,output_model):
    # torch.save({
    #     'model_state_dict': model.state_dict(),
    #     'optimizer_state_dict': optimizer.state_dict()
    # }, output_model)
    torch.save(model.state_dict(), output_model)
    print('The best model has been saved')  # 保存最好的结果的模型

def mkdir(path):
    # 判断路径是否存在
    isExists = os.path.exists(path)
    # 判断结果
    if not isExists:
        os.makedirs(path)
        print(path + ' 创建成功')
        return True
    else:
        print(path + ' 目录已存在')
        return False