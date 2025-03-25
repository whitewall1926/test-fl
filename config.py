class Config:
    # 基本训练参数
    local_epoch = 2         # 每个客户端的本地训练轮数
    global_rounds = 200     # 全局训练轮数
    batch_size = 64         # 本地训练批次大小
    
    # 模型参数
    model_name = 'mnist_cnn'  # 使用的模型名称
    
    # 优化器参数
    learning_rate = 0.001   # 学习率
    
    # 联邦学习参数
    num_clients = 10        # 客户端总数
    clients_ratio = 0.5   # 每轮参与训练的客户端比例
    
    # 数据集参数
    dataset = 'mnist'       # 使用的数据集名称
    alpha = 0.9             # 长尾划分的比例    
    
    # Fed+策略参数+iwds
    beta = 0.99             # 初始beta值
    beta_zero = 0.9999      # 最终beta值
    rou = 0.992             # beta衰减因子
    
    # 设备设置
    device = 'cuda'         # 训练设备 ('cuda' 或 'cpu')
    
    # 随机种子
    seed = 42               # 随机数种子

