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