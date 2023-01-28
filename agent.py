import numpy as np
import utils

class Agent:
    #에이전트가 가진 상태의 수
    STATE_DIM=2 # 주식 보유 비율, 포트폴리오 가치 비율

    #매매 수수료 및 세금
    TRAIDING_CHARGE = 0.00015 # 거래 수수료 (0.015%) 1이면 100%
    TRAIDING_TAX = 0.0025 # 거래세 (0.25%)

    #행동
    ACTION_BUY =0 # 매수
    ACTION_SELL = 1 # 매도
    ACTION_HOLD =2 # 홀딩, 매수와 매도중에서 행동을 고를 수없으면 이 행동을 선택한다.

    #인공신경망에서 확률을 구할 행동들
    ACTIONS = [ACTION_BUY, ACTION_SELL]# 정책신경망의 확률이 들어간다.
    NUM_ACTIONS = len(ACTIONS) # 인공 신경망에서 고려할 출력 값의 갯수

    def __init__(self, environment, min_trading_unit=1,max_trading_unit=2, delayed_reward_threshold=.05):
        # Environment
        self.environment =environment

        # 최소 매매 단위, 최대 매매 단위, 지연보상 임계치
        # 믿음이 높으면 더 많이 거래한다.
        self.min_trading_unit = min_trading_unit
        self.max_trading_unit = max_trading_unit

        # 지연 보상 임계치
        self.delayed_reward_threshold = delayed_reward_threshold

        # Agent 클래스의 속성
        self.initial_balance =0 # 초기 자본금
        self.balance = 0 #현재 현금 잔고
        self.num_stocks = 0 #보유 주식 수
        self.portfolio_value =0 # 보유현금 + 주식수 *주가
        self.base_portfolio_value =0 # 직전 합습 시점의 PV
        self.num_buy = 0 # 매수횟수
        self.num_sell =0 # 매도횟수
        self.num_hold = 0 # 홀딩횟수
        self.immediate_reward = 0 # 즉시보상(가장최근행동)
        self.profitloss = 0 # 현재 손익
        self.base_profitloss = 0
        self.exploration_base = 0 #x탐험행동 결정기준, 매수에서 탐험할지, 매도에서 탐험할지

        # Agent 클래스의  상태
        self.ratio_hold = 0 # 주식보유 비율
        self.ratio_portfolio_value =0 # 포트폴리오 가치비율

    def reset(self):
        # 각 에포크마다 초기화 필요
        self.balance = self.initial_balance
        self.num_stocks = 0
        self.portfolio_value = self.initial_balance
        self.base_portfolio_value = self.initial_balance
        self.num_buy=0
        self.num_sell =0
        self.num_hold =0
        self.immediate_reward =0
        self.ratio_hold = 0
        self.ratio_portfolio_value = 0

    def reset_exploration(self):
        #매수를 선호하기때문 0.5(50%) 매수 탐험확률을 부여함
        self.exploration_base = 0.5 +np.random.rand() / 2

    def set_balance(self, balance):
        # 초기자본금 설정
        self.initial_balance = balance

    def get_states(self):
        # 에이전트의 상태를 반환
        self.ratio_hold = self.num_hold/int(self.portfolio_value / self.environment.get_price())#주식 보유비율 비교(실제 주식수/ 살수 있는 주식수)
        self.ratio_portfolio_value = (self.portfolio_value/ self.base_portfolio_value)
        # 과거 포트폴리오가치(목표 수익 또는 손익률에 달성할때)와 현재 것을 비교

        return (self.ratio_hold, self.ratio_portfolio_value)#메모리를 적게 사용함

    def decide_action(self, pred_value, pred_policy, epsilon):
        confidence = 0

        pred = pred_policy#정책 신경망의 출력이 있는 경우
        if pred is None:
            pred = pred_value# 가치신경망으로 한다. ex) DQN
        if pred is None:
            # 예측 값이 없는 경우 탐험
            epsilon =1
        else :
            # 값이 모두 같은 경우 탐험
            maxpred = np.max(pred)
            if (pred ==maxpred).all():
                epsilon =1
        # 탐험결정
        if np.random.rand() < epsilon:
            exploration = True
            if np.random.rand() < self.exploration_base: # 초기에는 매수 위주
                action = self.ACTION_BUY
            else :
                action = np.random.randint(self.NUM_ACTIONS-1)+1# 매수 or 매도
        else :
            exploration = False
            action = np.argmax(pred)
        confidence = 0.5
        if pred_policy is not None:# 정책신경망을 사용했을 경우, 그대로 확률을 사용한다.
            confidence = pred[action]
        elif pred_value is not None:# 시그모이드를 사용했을 경우
            confidence = utils.sigmoid(pred[action])

    def validate_action(self, action):
        # 신용매수나 공매수가 없음
        if action == Agent.ACTION_BUY:
            # 적어도 한주를 살 수 있는지 확인(거래 수수료 고려)
            if self.balance < self.environment.get_price()*(1+self.TRAIDING_CHARGE)*self.min_trading_unit:
                return False
        elif action == Agent.ACTION_SELL:
            if self.num_stocks <=0:
                return False
        return True

    def decide_trading_unit(self,confidence):
        if np.isnan(confidence):
            return self.min_trading_unit
        adding_trading = max(min(int(confidence*(self.max_trading_unit-
                                                 self.min_trading_unit)),self.max_trading_unit-self.min_trading_unit),0)

        return self.min_trading_unit + adding_trading

    def act(self,action, confidence):
        #행동과 결과를 반환
        if not self.validate_action(action):# 해당행동 못한다면
            action = self.ACTION_HOLD
        # 환경에서 현재 가격 얻기
        curr_price = self.environment.get_price()

        # 즉시 보상 초기화
        self.immediate_reward = 0

        if action==Agent.ACTION_BUY:
            trading_unit = self.decide_trading_unit(confidence)
            balance = (self.balance - curr_price*(1+self.TRAIDING_CHARGE)*trading_unit)
            # 보유 현금이 모자랄 경우 보유 현금으로 가능한 만큼 최대한 매수
            if balance <0:
                trading_unit = max(min(int(self.balance/(curr_price*(1+self.TRAIDING_CHARGE))),self.max_trading_unit), self.min_trading_unit)

            invest_amount = curr_price*(1+self.TRAIDING_CHARGE)*trading_unit
            if invest_amount >0:
                self.balance -= invest_amount #보유 현금 갱신
                self.num_stocks += trading_unit # 보유 주식 수를 갱신
                self.num_buy +=1 # 매수 횟수 증가
        elif action == Agent.ACTION_SELL:
            trading_unit = self.decide_trading_unit(confidence)
            trading_unit = min(trading_unit,self.num_stocks)
            #매도
            invest_amount = curr_price*(1-(self.TRAIDING_TAX+self.TRAIDING_CHARGE))*trading_unit

            if invest_amount >0:
                self.num_stocks -= trading_unit # 보유 주식 수를 갱신
                self.balance += invest_amount # 보유현금 갱신
                self.num_sell += 1# 매도 횟수 증가

        elif action == Agent.ACTION_HOLD:
            self.num_hold += 1

        # 포트폴리오 가치 갱신
        self.portfolio_value = self.balance +curr_price*self.num_stocks
        self.profitloss = ((self.portfolio_value - self.initial_balance)/ self.initial_balance)

        # 즉시보상 - 수익률
        self.immediate_reward = self.profitloss

        # 지역보상 - 익절, 손절 기준
        delayed_reward = 0

        self.base_profitloss = ((self.portfolio_value - self.base_portfolio_value)/ self.base_portfolio_value)

        if self.base_profitloss > self.delayed_reward_threshold or \
            self.base_profitloss < -self.delayed_reward_threshold:
            # 목표 수익률 달성하여 기준 포트폴리오 가치 갱신
            # 또는 손실 기준치를 초과하여 기준 포트폴리오 가치 갱신
            self.base_portfolio_value = self.portfolio_value
            delayed_reward = self.immediate_reward
        else:
            delayed_reward =0

        return self.immediate_reward, delayed_reward
