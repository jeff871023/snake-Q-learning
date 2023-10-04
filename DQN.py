import random  # 匯入隨機模組
import numpy as np  # 匯入 NumPy 模組，用於數值運算
import pandas as pd  # 匯入 Pandas 模組，用於資料處理
from operator import add  # 匯入 add 函數，用於列表元素相加
import collections  # 匯入 collections 模組，用於建立雙向佇列
import torch  # 匯入 PyTorch 模組
import torch.nn as nn  # 匯入神經網絡模組
import torch.nn.functional as F  # 匯入神經網絡相關函數
import torch.optim as optim  # 匯入優化器模組
import copy  # 匯入 copy 模組，用於物件的複製

# 指定設備為 CPU，如果有 GPU 則使用 'cuda'
DEVICE = 'cpu'  # 'cuda' if torch.cuda.is_available() else 'cpu'

class DQNAgent(torch.nn.Module):
    def __init__(self, params):
        super().__init__()
        self.reward = 0  # 初始獎勵為 0
        self.gamma = 0.9  # 衰減因子
        self.dataframe = pd.DataFrame()  # 創建一個 Pandas DataFrame
        self.short_memory = np.array([])  # 創建一個空的 NumPy 陣列
        self.agent_target = 1  # 代理目標
        self.agent_predict = 0  # 代理預測
        self.learning_rate = params['learning_rate']  # 學習率        
        self.epsilon = 1  # 探索率初始值
        self.actual = []  # 儲存實際動作
        self.first_layer = params['first_layer_size']  # 第一層神經元數量
        self.second_layer = params['second_layer_size']  # 第二層神經元數量
        self.third_layer = params['third_layer_size']  # 第三層神經元數量
        self.memory = collections.deque(maxlen=params['memory_size'])  # 創建一個雙向佇列作為記憶體
        self.weights = params['weights_path']  # 權重文件的路徑
        self.load_weights = params['load_weights']  # 是否載入現有權重
        self.optimizer = None  # 優化器初始化為 None
        self.network()  # 初始化神經網絡

    def network(self):
        # 創建神經網絡的各層
        self.f1 = nn.Linear(11, self.first_layer)  # 第一層全連接層
        self.f2 = nn.Linear(self.first_layer, self.second_layer)  # 第二層全連接層
        self.f3 = nn.Linear(self.second_layer, self.third_layer)  # 第三層全連接層
        self.f4 = nn.Linear(self.third_layer, 3)  # 第四層全連接層

        # 如果設置了載入權重，則載入權重
        if self.load_weights:
            self.model = self.load_state_dict(torch.load(self.weights))
            print("已載入權重")

    def forward(self, x):
        # 定義前向傳播
        x = F.relu(self.f1(x))
        x = F.relu(self.f2(x))
        x = F.relu(self.f3(x))
        x = F.softmax(self.f4(x), dim=-1)
        return x
    
    def get_state(self, game, player, food):
        """
        返回當前狀態
        狀態是一個包含 11 個值的 NumPy 陣列，代表：
            - 1 或 2 步前的危險情況
            - 1 或 2 步右側的危險情況
            - 1 或 2 步左側的危險情況
            - 蛇向左移動
            - 蛇向右移動
            - 蛇向上移動
            - 蛇向下移動
            - 食物在左側
            - 食物在右側
            - 食物在上方
            - 食物在下方      
        """
        state = [
            (player.x_change == 20 and player.y_change == 0 and ((list(map(add, player.position[-1], [20, 0])) in player.position) or
            player.position[-1][0] + 20 >= (game.game_width - 20))) or (player.x_change == -20 and player.y_change == 0 and ((list(map(add, player.position[-1], [-20, 0])) in player.position) or
            player.position[-1][0] - 20 < 20)) or (player.x_change == 0 and player.y_change == -20 and ((list(map(add, player.position[-1], [0, -20])) in player.position) or
            player.position[-1][-1] - 20 < 20)) or (player.x_change == 0 and player.y_change == 20 and ((list(map(add, player.position[-1], [0, 20])) in player.position) or
            player.position[-1][-1] + 20 >= (game.game_height-20))),  # 危險直行

            (player.x_change == 0 and player.y_change == -20 and ((list(map(add,player.position[-1],[20, 0])) in player.position) or
            player.position[ -1][0] + 20 > (game.game_width-20))) or (player.x_change == 0 and player.y_change == 20 and ((list(map(add,player.position[-1],
            [-20,0])) in player.position) or player.position[-1][0] - 20 < 20)) or (player.x_change == -20 and player.y_change == 0 and ((list(map(
            add,player.position[-1],[0,-20])) in player.position) or player.position[-1][-1] - 20 < 20)) or (player.x_change == 20 and player.y_change == 0 and (
            (list(map(add,player.position[-1],[0,20])) in player.position) or player.position[-1][
             -1] + 20 >= (game.game_height-20))),  # 危險向右

             (player.x_change == 0 and player.y_change == 20 and ((list(map(add,player.position[-1],[20,0])) in player.position) or
             player.position[-1][0] + 20 > (game.game_width-20))) or (player.x_change == 0 and player.y_change == -20 and ((list(map(
             add, player.position[-1],[-20,0])) in player.position) or player.position[-1][0] - 20 < 20)) or (player.x_change == 20 and player.y_change == 0 and (
            (list(map(add,player.position[-1],[0,-20])) in player.position) or player.position[-1][-1] - 20 < 20)) or (
            player.x_change == -20 and player.y_change == 0 and ((list(map(add,player.position[-1],[0,20])) in player.position) or
            player.position[-1][-1] + 20 >= (game.game_height-20))), #危險向左

            player.x_change == -20,  # 向左移動
            player.x_change == 20,  # 向右移動
            player.y_change == -20,  # 向上移動
            player.y_change == 20,  # 向下移動
            food.x_food < player.x,  # 食物在左側
            food.x_food > player.x,  # 食物在右側
            food.y_food < player.y,  # 食物在上方
            food.y_food > player.y  # 食物在下方
        ]

        for i in range(len(state)):
            if state[i]:
                state[i]=1
            else:
                state[i]=0

        return np.asarray(state)

    def set_reward(self, player, crash):
        """
        返回獎勵值
        獎勵值為：
            - 當蛇撞到障礙物時為 -10
            - 當蛇吃到食物時為 +10
            - 否則為 0
        """
        self.reward = 0
        if crash:
            self.reward = -10
            return self.reward
        if player.eaten:
            self.reward = 10
        return self.reward

    def remember(self, state, action, reward, next_state, done):
        """
        將 <狀態, 動作, 獎勵, 下一個狀態, 是否結束> 元組儲存在記憶體緩衝區中。
        """
        self.memory.append((state, action, reward, next_state, done))

    def replay_new(self, memory, batch_size):
        """
        回放記憶體。
        """
        if len(memory) > batch_size:
            minibatch = random.sample(memory, batch_size)
        else:
            minibatch = memory
        for state, action, reward, next_state, done in minibatch:
            self.train()
            torch.set_grad_enabled(True)
            target = reward
            next_state_tensor = torch.tensor(np.expand_dims(next_state, 0), dtype=torch.float32).to(DEVICE)
            state_tensor = torch.tensor(np.expand_dims(state, 0), dtype=torch.float32, requires_grad=True).to(DEVICE)
            if not done:
                target = reward + self.gamma * torch.max(self.forward(next_state_tensor)[0])
            output = self.forward(state_tensor)
            target_f = output.clone()
            target_f[0][np.argmax(action)] = target
            target_f.detach()
            self.optimizer.zero_grad()
            loss = F.mse_loss(output, target_f)
            loss.backward()
            self.optimizer.step()            

    def train_short_memory(self, state, action, reward, next_state, done):
        """
        在當前時間步驟上訓練 DQN 代理器，使用 <狀態, 動作, 獎勵, 下一個狀態, 是否結束> 元組。
        """
        self.train()
        torch.set_grad_enabled(True)
        target = reward
        next_state_tensor = torch.tensor(next_state.reshape((1, 11)), dtype=torch.float32).to(DEVICE)
        state_tensor = torch.tensor(state.reshape((1, 11)), dtype=torch.float32, requires_grad=True).to(DEVICE)
        if not done:
            target = reward + self.gamma * torch.max(self.forward(next_state_tensor[0]))
        output = self.forward(state_tensor)
        target_f = output.clone()
        target_f[0][np.argmax(action)] = target
        target_f.detach()
        self.optimizer.zero_grad()
        loss = F.mse_loss(output, target_f)
        loss.backward()
        self.optimizer.step()
