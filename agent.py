import numpy as np
import utils

class Agent:
    #에이전트가 가진 상태의 수
    STATE_DIM=2 # 주식 보유 비율, 포트폴리오 가치 비율

    #매매 수수료 및 세금
    TRAIDING_CHARGE = 0.00015 # 거래 수수료 (0.015%)
    TRAIDING_TAX = 0.0025 # 거래세 (0.25%)

    #행동
    ACTION_BUY =0 # 매수
    ACTION_SELL = 1 # 매도
    ACTION_HOLD =2 # 홀딩, 매수와 매도중에서 행동을 고를 수없으면 이 행동을 선택한다.

    #인공신경망에서 확률을 구할 행동들
    ACTIONS = [ACTION_BUY, ACTION_SELL]# 정책신경망의 확률이 들어간다.
    NUM_ACTIONS = len(ACTIONS) # 인공 신경망에서 고려할 출력 값의 갯수