import threading
import numpy as np
import matplotlib.pylab as plt
plt.switch_backend('agg')

from mpl_finance import candlestick_ohlc
from agent import Agent

lock = threading.Lock()

class Visualizer:
    COLORS = ['r', 'g', 'b']

    def __init__(self, vnet=False):
        # 캔버스 같은 역할을 하는 Matplotlib의 Figure 클래스
        self.canvas = None
        # 차트를 그리기 위한 Axes 클래스
        self.axes = None
        self.title =''#그림제목

    def prepare(self,chart_data, title):
        self.title = title
        with lock:
            #캔버스를 초기화하고, 5개의 차트를 그릴준비
            self.fig,self.axes = plt.subplots(nrows=5,ncols=1,facecolor='w',sharex=True)
            for ax in self.axes:
                #보기 어려운 과학적 표기 비활성화
                ax.get_xaxis().get_major_formatter().set_scientific(False)
                ax.get_yaxis().get_major_formatter().set_scientific(False)
                #y axis의 위치를 오른쪽으로 변경
                ax.yaxis.tick_right()

            # 차트 1봉표시
            self.axes[0].set_ylabel('Env.')
            x = np.arange(len(chart_data))#N개 1xN, 인덱스
            # open, high, low, close 순서로 된 2차원 배열
            ohlc = np.hstack((x.reshape(-1,1),np.array(chart_data)[:,1:-1]))#차트 데이터중에서 일부를 붙인다. Nx5, 맨앞은 인덱스

            #양봉은 빨간색으로 음봉은 파란색으로 표시
            candlestick_ohlc(self.axes[0],ohlc,colorup='r',colordown='b')# Nx5로 넣어줘야한다.

            #거래량 시각화
            ax=self.axes[0].twinx()
            volume = np.array(chart_data)[:,-1].tolist()
            ax.bar(x,volume,color='b',alpha=0.3)

    def plot(self,epoch_str=None, num_epochs=None, epsilon=None,
             action_list=None,actions=None,num_stocks=None,
             outvals_value=[], outvals_policy=[],exps=None,
             learing_idxes=None,initial_balance=None,pvs=None):
        with lock:
            x= np.arange(len(actions))
            actions = np.array(actions)# 에이전트의 행동 배열
            #가치 신경망의 출력 배열
            outvals_value = np.array(outvals_value)
            #정책 신경망의 출력 배열
            outvals_policy = np.array(outvals_policy)
            #초기 자본금 배열
            pvs_base = np.zeros(len(actions))+initial_balance

            for action, color in zip(action_list, self.COLORS):
                for i in x[actions==action]:
                    # 배경색으로 행동을 표시
                    self.axes[1].axvline(i, color=color, alpha=0.1)
            self.axes[1].plot(x, num_stocks, '-k')#보유 주식수를 line차트로 그린다.

            # 차트 3 가치신경망
            if len(outvals_value) >0:
                max_actions = np.argmax(outvals_value,axis=1)
                for action, color in zip(action_list, self.COLORS):
                    for idx in x:
                        if max_actions[idx] == actions:
                            self.axes[2].axvline(idx,
                                                 color=color, alpha=0.1)
                    # 가치 신경 출력의 tanh 그리기
                    self.axes[2].plot(x,outvals_value[:,action],color=color,linestyle="-")
                    #매수는 빨간색, 매도는 파란색, 관망은 초록색으로 표시한다.

            # 차트 4 정책신경망
            # 탐험은 노란색으로 표시
            for exp_idx in exps:
                self.axes[3].axvline(exp_idx,color='y')
            # 행동을 배경으로 그리기
            _outvals= outvals_policy if len(outvals_policy) >0 else outvals_value
            for idx, outval in zip(x,_outvals):
                color = 'white'
                if np.isnan(outval.max()):
                    continue
                if outval.argmax() == Agent.ACTION_BUY:
                    color= 'r'
                elif outval.argmax() == Agent.ACTION_SELL:
                    color='b'
                self.axes[3].axvline(idx,color=color, alpha=0.1)
            # 정책신경망의 출력 그리기
            if len(outvals_policy)>0:
                for action, color in zip(action_list,self.COLORS):
                    self.axes[3].plot(x,outvals_policy[:,action],color=color,linestyle='-')

            #차트 5 포트폴리오 가치
            self.axes[4].axhline(initial_balance, linestyle="-", color="gray")
            self.axes[4].fill_between(x,pvs,pvs_base,where=pvs > pvs_base, facecolor='r', alpha=0.1)
            self.axes[4].fill_between(x, pvs, pvs_base, where=pvs < pvs_base, facecolor='b', alpha=0.1)
            self.axes[4].plot(x,pvs,'-k')
            #학습 위치 표시
            for learing_idx in learing_idxes:
                self.axes[4].axvline(learing_idx, color='y')

            self.fig.suptitle('{} \nEpoch:{}\{} e={:.2f}'.format(self.title,epoch_str,num_epochs, epsilon))
            self.fig.tight_layout()
            self.fig.subplots_adjust(top=0.85)

    def clear(self, xlim):
        with lock:
            _axes = self.axes.tolist()
            for ax in _axes[1:]:
                ax.cla() #그린 차트를 지운다.
                ax.relim() # limit를 초기화
                ax.autoscale() # 스케일 재설정

            self.axes[1].set_ylabel('Agent')
            self.axes[2].set_ylabel('V')
            self.axes[3].set_ylabel('P')
            self.axes[4].set_ylabel('PV')
            for ax in _axes:
                ax.set_xlim(xlim)
                ax.get_xaxis().get_major_formatter().set_scientific(False)
                ax.get_yaxis().get_major_formatter().set_scientific(False)
                # x축 간격을 일정하게 설정, 토요일이나 일요일과 같이 휴장하는 날에는 그부분 차트를 표현하지 않음
                ax.ticklabel_format(useOffset=False)

    def save(self, path):
        with lock:
            self.fig.savefig(path)

