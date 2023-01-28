# 다양한 강화학습 방식을 수행하기 위한 학습기 클래스

import os
import logging
import abc
#abc는 추상베이스 클래스,추상메서드를 선언할 수 있다.
import collections
import threading
import time
import numpy as np
from utils import sigmoid
from environment import Environment
from agent import Agent
from networks import Network, DNN, LSTMNetwork, CNN
from visualizer import Visualizer

class ReinforcementLearner:
    __metaclass__=abc.ABCMeta
    lock = threading.Lock()

    def __init__(self,rl_method='rl', stock_code=None,
                 chart_data=None, training_data=None,
                 min_trading_unit=1, max_trading_unit=2,
                 delayed_reward_threshold=.05,
                 net='dnn', num_steps=1, lr=0.001,
                 value_network=None,policy_network=None,
                 output_path='', reuse_models=True):
        # 인자확인
        assert min_trading_unit >0
        assert max_trading_unit >0
        assert max_trading_unit >=min_trading_unit
        assert num_steps >0
        assert lr >0

        #강화학습 기법 설정
        self.rl_method = rl_method
        #환경설정
        self.stock_code = stock_code
        self.chart_data = chart_data #일봉차트 데이터
        self.environment = Environment(chart_data)

        #에이전트 설정
        self.agent = Agent(self.environment,min_trading_unit=min_trading_unit,
                           max_trading_unit=max_trading_unit,
                           delayed_reward_threshold=delayed_reward_threshold)

        #학습데이터
        self.training_data = training_data
        self.sample = None
        self.training_data_idx = -1

        #벡터 크기 = 학습데이터 벡터 크기 + 에이전트 상태크기
        self.num_features=self.agent.STATE_DIM
        if self.training_data is not None:
            self.num_features +=self.training_data.shape[1]

        #신경망설정
        self.net = net
        self.num_steps = num_steps
        self.lr =lr
        self.value_network = value_network
        self.policy_network = policy_network
        self.reuse_models = reuse_models
        self.visualizer = Visualizer()

        #메모리
        self.memory_sample =[]
        self.memory_action =[]
        self.memory_reward =[]
        self.memory_value =[]
        self.memory_policy= []
        self.memory_pv = []
        self.memory_num_stocks =[]
        self.memory_exp_idx =[]
        self.memory_learning_idx = []

        #에포크 관련정보
        self.loss =0
        self.itr_cnt =0
        self.exploration_cnt =0
        self.batch_size =0
        self.learning_cnt =0
        #로그 등 출력 경로
        self.output_path = output_path

    def init_value_network(self, shared_network=None, activation='linear', loss='mse'):
        #손익률을 희귀분석하는 모델
        if self.net =='dnn':
            self.value_network=DNN(input_dim = self.num_features,
                                   output_dim = self.agent.NUM_ACTIONS,
                                   lr=self.lr,
                                   shared_network=shared_network,
                                   activation=activation,loss=loss)
        elif self.net =='lstm':
            self.value_network=LSTMNetwork(input_dim = self.num_features,
                                   output_dim = self.agent.NUM_ACTIONS,
                                   lr=self.lr,num_steps=self.num_steps,
                                   shared_network=shared_network,
                                   activation=activation,loss=loss)
        elif self.net =='cnn':
            self.value_network=CNN(input_dim = self.num_features,
                                   output_dim = self.agent.NUM_ACTIONS,
                                   lr=self.lr,num_steps=self.num_steps,
                                   shared_network=shared_network,
                                   activation=activation,loss=loss)

        if self.reuse_models and os.path.exists(self.value_network_path):
            self.value_network.load_model(model_path=self.value_network_path)

    def init_policy_network(self, shared_network=None, activation='sigmoid', loss='mse'):
        #샘플에 대해서 PV를 높이기 위해 취하기 좋은행동에 대한 분류 모델
        if self.net == 'dnn':
            self.policy_network = DNN(input_dim=self.num_features,
                                     output_dim=self.agent.NUM_ACTIONS,
                                     lr=self.lr,
                                     shared_network=shared_network,
                                     activation=activation, loss=loss)
        elif self.net == 'lstm':
            self.policy_network = LSTMNetwork(input_dim=self.num_features,
                                             output_dim=self.agent.NUM_ACTIONS,
                                             lr=self.lr, num_steps=self.num_steps,
                                             shared_network=shared_network,
                                             activation=activation, loss=loss)
        elif self.net == 'cnn':
            self.policy_network = CNN(input_dim=self.num_features,
                                     output_dim=self.agent.NUM_ACTIONS,
                                     lr=self.lr, num_steps=self.num_steps,
                                     shared_network=shared_network,
                                     activation=activation, loss=loss)

        if self.reuse_models and os.path.exists(self.policy_network_path):
            self.policy_network.load_model(model_path=self.policy_network_path)

    def reset(self):
        #에포크마다 새로 데이터가 쌓이는 변수들을 초기화하는 함수
        self.sample = None
        self.training_data_idx = -1#학습데이터를 처음부터 읽기위해서
        self.environment.reset()
        self.agent.reset()
        self.visualizer.clear([0,len(self.chart_data)])
        self.memory_sample = []
        self.memory_action = []
        self.memory_reward = []
        self.memory_value = []
        self.memory_policy =[]
        self.memory_pv =[]
        self.memory_num_stocks =[]
        self.memory_exp_idx =[]
        self.memory_learning_idx = []

        #에포크 관련 정보 초기화
        self.loss = 0.
        self.itr_cnt = 0 #수행한 에포크 수를 저장
        self.exploration_cnt =0 #무작위 투자항한 횟수
        self.batch_size =0
        self.learning_cnt = 0

    def build_sample(self):
        #학습데이터를 구성하는 샘플 하나를 생성하는 함수
        self.environment.observe()#다음 데이터를 얻어온다.
        if len(self.training_data) > self.training_data_idx +1:
            self.training_data_idx +=1
            self.sample = self.training_data.iloc[self.training_data_idx].tolist()
            self.sample.extend(self.agent.get_states())#26개+2개
            return self.sample
        return None

    @abc.abstractmethod
    def get_batch(self,batch_size,delayed_reward, discount_factor):
        pass##배치학습데이터를 생성합니다.

    def update_networks(self,batch_size, delayed_reward, discount_factor):
        #신경망을 학습합니다.
        x, y_value, y_policy = self.get_batch(batch_size,delayed_reward,discount_factor)
        if len(x) > 0:
            loss=0
            if y_value is not None:
                #가치 신경망을 갱신
                loss += self.value_network.train_on_batch(x,y_value)
            if y_policy is not None:
                #정책 신경망을 갱신(학습후 손실을 반환)
                loss += self.policy_network.train_on_batch(x,y_policy)
            return loss
        return None

    def fit(self, delayed_reward,discount_factor):
        #배치 학습데이터의 크기를 정하고, update_networks함수를 호출,
        if self.batch_size >0:
            _loss = self.update_networks(self.batch_size,delayed_reward,discount_factor)
            if _loss is not None:
                self.loss += abs(_loss)#에포크동안의 전체 손실에 합산
                self.learning_cnt +=1#학습횟수 저장
                self.memory_learning_idx.append(self.training_data_idx)
            self.batch_size = 0