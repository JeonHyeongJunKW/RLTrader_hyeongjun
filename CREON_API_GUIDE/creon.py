import time
import win32com.client
import pandas as pd

class Creon:
    def __init__(self):
        self.obj_CpCodeMgr = win32com.client.Dispatch('CpUtil.CpCodeMgr')
        self.obj_CpCybos = win32com.client.Dispatch('CpUtil.CpCybos')
        self.obj_StockChart = win32com.client.Dispatch('CpSysDib.StockChart')

    def creon_7400_주식차트조회(self,code, date_from, date_to):
        b_connected = self.obj_CpCybos.IsConnect
        if b_connected ==0:
            print("연결실패")
            return None

        list_field_key = [0,1,2,3,4,5,8]
        list_field_name = ['date', 'time', 'open', 'high','low', 'close', 'volume']
        dict_chart = {name : [] for name in list_field_name}

        self.obj_StockChart.SetInputValue(0,'A'+code)
        self.obj_StockChart.SetInputValue(1, ord('1'))
        self.obj_StockChart.SetInputValue(2, date_to)
        self.obj_StockChart.SetInputValue(3, date_from)
        self.obj_StockChart.SetInputValue(5, list_field_key)
        self.obj_StockChart.SetInputValue(6, ord('D')) #차트 단위를 일단위로 넣습니다.
        self.obj_StockChart.BlockRequest()

        status = self.obj_StockChart.GetDibStatus()# 요청결과 상태를 받아온다. 0이아니면 이상이 있음.

        msg = self.obj_StockChart.GetDibMsg1()
        print("통신상태: {} {}".format(status,msg))
        if status !=0:
            return None
        cnt = self.obj_StockChart.GetHeaderValue(3) # 출력 결과물의 개수
        for i in range(cnt):
            dict_item =({name: self.obj_StockChart.GetDataValue(pos,i) for pos, name in zip(range(len(list_field_name)), list_field_name)})
            for k,v in dict_item.items():
                dict_chart[k].append(v)

        print("차트:{} {}".format(cnt,dict_chart))
        return pd.DataFrame(dict_chart, columns=list_field_name)

if __name__=='__main__':
    creon = Creon()
    print(creon.creon_7400_주식차트조회('035420',20150101,20171231))
