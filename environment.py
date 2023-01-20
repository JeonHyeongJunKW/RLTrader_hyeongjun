'''
Environment 클래스가 투자할 종목의 차트 데이터를 관리합니다.
- 전체 차트 데이터가 있다.
- 과거 시점부터 가장 최근 시점까지 순차적으로 데이터를 제공한다.
'''

class Environment:
    PRICE_IDX =4 #종가의 위치

    def __init__(self, chart_data=None):
        self.chart_data= chart_data
        self.observation=None# 현재 관측치
        self.idx = -1# 차트 데이터에서 현재 위치
    def reset(self):
        self.observation = None
        self.idx=-1

    def observe(self):
        if len(self.chart_data) > self.idx +1:
            self.idx +=1
            self.observation = self.chart_data.iloc[self.idx]
            return self.observation
        return None

    def get_price(self):#현재 observation에서 종가를 획득
        if self.observation is not None:
            return self.observation[self.PRICE_IDX]
        return None

    def set_chart_data(self, chart_data):
        self.chart_data = chart_data
