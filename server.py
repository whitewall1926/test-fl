class Server:
    def __init__(self, rounds, clients, model):
        # ��ʼ��������
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
        # �ۺ�ģ�Ͳ����ķ���
        # ...existing code...
        global_parameters
        for _, client in enumerate(self.clients):
            
        pass
