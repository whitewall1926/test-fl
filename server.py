class Server:
    def __init__(self, rounds, clients, model):
        # 初始化服务器
        # ...existing code...
        self.rounds =  rounds
        self.clients = clients
        self.model = model

    def train(self):
        parameters = self.model.state_dict()
        for t in range(self.rounds):
            for _, client in enumerate(self.clients):
                client.train(parameters)
        
        pass

    def aggregate_model_parameters(self):
        # 聚合模型参数的方法
        # ...existing code...
        global_parameters
        for _, client in enumerate(self.clients):
            
        pass
