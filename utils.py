import json
import matplotlib.pyplot as plt
import torch
import numpy as np

def load_config(config_path='config.json'):
    """
    加载JSON格式的配置文件
    
    Args:
        config_path: JSON配置文件路径
        
    Returns:
        包含配置参数的字典
    """
    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)
    return config

def plot(accuracy_history, save_path=None):
    """
    绘制准确率历史曲线
    
    Args:
        accuracy_history: 准确率历史数据列表
        save_path: 图像保存路径，如果为None则显示图像
    """
    plt.figure(figsize=(10, 6))
    plt.plot(accuracy_history, 'b-', linewidth=2)
    plt.title('模型准确率随训练轮数的变化')
    plt.xlabel('训练轮数')
    plt.ylabel('准确率')
    plt.grid(True)
    
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()
