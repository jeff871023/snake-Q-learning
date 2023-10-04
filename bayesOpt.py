# -*- coding: utf-8 -*-
"""
這個程式是用來進行貝葉斯優化的設計，以優化神經網絡模型的參數，主要用於一個名為 'snake' 的應用場景。

"""
# 匯入所需的模組
from snakeClass import run  # 匯入名為 'run' 的函式
from GPyOpt.methods import BayesianOptimization  # 匯入貝葉斯優化相關模組
import datetime  # 匯入用於處理日期和時間的模組

################################################
#       設定貝葉斯優化的相關參數       #
################################################

class BayesianOptimizer():  # 定義一個貝葉斯優化器的類別
    def __init__(self, params):  # 初始化函式，接受一個名為 'params' 的參數
        self.params = params  # 將傳入的 'params' 參數儲存在物件的 'params' 屬性中

    def optimize_RL(self):  # 定義一個用於進行強化學習優化的方法
        def optimize(inputs):  # 內部函式，接受一個名為 'inputs' 的參數
            print("INPUT", inputs)  # 印出輸入的 'inputs'
            inputs = inputs[0]  # 將 'inputs' 解包

            # 需要進行優化的變數
            self.params["learning_rate"] = inputs[0]  # 將 'learning_rate' 設為 'inputs' 中的第一個元素
            lr_string = '{:.8f}'.format(self.params["learning_rate"])[2:]  # 將 'learning_rate' 格式化為字串
            self.params["first_layer_size"] = int(inputs[1])  # 將 'first_layer_size' 設為 'inputs' 中的第二個元素
            self.params["second_layer_size"] = int(inputs[2])  # 將 'second_layer_size' 設為 'inputs' 中的第三個元素
            self.params["third_layer_size"] = int(inputs[3])  # 將 'third_layer_size' 設為 'inputs' 中的第四個元素
            self.params["epsilon_decay_linear"] = int(inputs[4])  # 將 'epsilon_decay_linear' 設為 'inputs' 中的第五個元素

            self.params['name_scenario'] = 'snake_lr{}_struct{}_{}_{}_eps{}'.format(lr_string,
                                                                               self.params['first_layer_size'],
                                                                               self.params['second_layer_size'],
                                                                               self.params['third_layer_size'],
                                                                               self.params['epsilon_decay_linear'])

            self.params['weights_path'] = 'weights/weights_' + self.params['name_scenario'] + '.h5'  # 設定權重文件的路徑
            self.params['load_weights'] = False  # 不讀取現有的權重
            self.params['train'] = True  # 設定模型為訓練模式
            print(self.params)  # 印出目前的參數設定
            score, mean, stdev = run(self.params)  # 使用 'run' 函式進行模型訓練
            print('總分數: {}   平均值: {}   標準差:   {}'.format(score, mean, stdev))  # 印出訓練結果
            with open(self.params['log_path'], 'a') as f:  # 以追加模式打開日誌文件
                f.write(str(self.params['name_scenario']) + '\n')  # 寫入場景名稱
                f.write('參數: ' + str(self.params) + '\n')  # 寫入參數設定
            return score  # 返回得分

        # 定義優化的參數範圍
        optim_params = [
            {"name": "learning_rate", "type": "continuous", "domain": (0.00005, 0.001)},  # 連續型變數 'learning_rate'
            {"name": "first_layer_size", "type": "discrete", "domain": (20,50,100,200)},  # 確定型變數 'first_layer_size'
            {"name": "second_layer_size", "type": "discrete", "domain": (20,50,100,200)},  # 確定型變數 'second_layer_size'
            {"name": "third_layer_size", "type": "discrete", "domain": (20,50,100,200)},  # 確定型變數 'third_layer_size'
            {"name":'epsilon_decay_linear', "type": "discrete", "domain": (self.params['episodes']*0.2,
                                                                           self.params['episodes']*0.4,
                                                                           self.params['episodes']*0.6,
                                                                           self.params['episodes']*0.8,
                                                                           self.params['episodes']*1)}  # 確定型變數 'epsilon_decay_linear'
        ]

        # 初始化貝葉斯優化器
        bayes_optimizer = BayesianOptimization(f=optimize,  # 使用上面定義的 'optimize' 方法作為目標函數
                                               domain=optim_params,  # 優化的參數範圍
                                               initial_design_numdata=6,  # 初始設計的數據點數量
                                               acquisition_type="EI",  # 使用概率改進策略
                                               exact_feval=True,  # 精確的函數評估
                                               maximize=True)  # 最大化目標函數

        bayes_optimizer.run_optimization(max_iter=20)  # 在優化器上執行優化，最多執行 20 次優化迭代
        print('最佳學習率: ', bayes_optimizer.x_opt[0])  # 印出最佳的學習率
        print('最佳第一層神經元數: ', bayes_optimizer.x_opt[1])  # 印出最佳的第一層神經元數
        print('最佳第二層神經元數: ', bayes_optimizer.x_opt[2])  # 印出最佳的第二層神經元數
        print('最佳第三層神經元數: ', bayes_optimizer.x_opt[3])  # 印出最佳的第三層神經元數
        print('最佳 epsilon 線性衰減值: ', bayes_optimizer.x_opt[4])  # 印出最佳的 epsilon 線性衰減值
        return self.params  # 返回參數設定

##################
#      主程式      #
##################
if __name__ == '__main__':  # 如果這個程式是直接執行的主程式
    # 定義一個名為 'bayesOpt' 的貝葉斯優化器實例，並傳入參數 'params'
    bayesOpt = BayesianOptimizer(params)
    bayesOpt.optimize_RL()  # 呼叫 'optimize_RL' 方法進行強化學習優化
