import sys
import os
from PyQt5.QAxContainer import *
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
import pandas as pd
import numpy as np

class kiwoomAPI(QAxWidget) :
    def __init__(self):
        super().__init__()
        self.setControl("KHOPENAPI.KHOpenAPICtrl.1")

        self._set_signal_slots()

    def _set_signal_slots(self):
        self.OnEventConnect.connect(self._event_connect)
        self.OnReceiveConditionVer.connect(self._receive_condition_ver)
        self.OnReceiveTrCondition.connect(self._receive_tr_condition)
        self.OnReceiveTrData.connect(self._receive_tr_data)
        # self.OnreceiveRealData.connect(self.receive_realData)
        # self.OnReceiveMsg.connect(self.receive_Msg)
        # self.OnReceiveChejanData.connect(self.receive_chejanData)

    def _event_connect(self, err_code):
        if err_code == 0:
            print("connected")
        else:
            print("disconnected")

        self.conditionLoop.exit()

    def _receive_condition_ver(self, ret, msg):
        self.conditionLoop.exit()

    def _receive_tr_condition(self, sNo, codes, cName, cIndex, next):
        codeList = codes.split(';')[:-1]

        print(len(codeList))
        self.tr_condition_data = codeList
        self.conditionLoop.exit()

    def _receive_tr_data(self, sNo, rqName, trCode, recordName, preNext, dataLength, errorCode, msg, splmMsg):
        if preNext == '2' :
            self.tr_remained = True
        else :
            self.tr_remained = False

        data = self._opt10081(trCode, recordName)

        fname = '{}.csv'.format(self.stock_code)

        data.to_csv(fname)

        self.conditionLoop.exit()

    """
    def getRepeatCnt(self, trCode, rqName):
        count = self.dynamicCall("GetRepeatCnt(QString, QString)", trCode, rqName)
        return count
    """

    def commConnect(self):
        self.dynamicCall("CommConnect()")
        self.conditionLoop = QEventLoop()
        self.conditionLoop.exec_()

    def getConditionLoad(self):
        ret = self.dynamicCall("GetConditionLoad()")

        if not ret :
            print("fail to condition load")

        else :
            print("success to condition load")

        self.conditionLoop = QEventLoop()
        self.conditionLoop.exec_()

    def getConditionNameList(self):
        data = self.dynamicCall("GetConditionNameList()")

        if data == "":
            print("Have No List")

        else :
            conditions = data.split(";")[:-1]

            conds = []
            for condition in conditions:
                index, name = condition.split('^')
                conds.append((index, name))

            print(conds)
            return conds

    def sendCondition(self, sNo, cName, cIndex, isRealtime):
        ret = self.dynamicCall("SendCondition(QString, QString, int, int)", sNo, cName, cIndex, isRealtime)

        if not ret :
            print("fail to send condition")
        else :
            print("success to send condition")

        self.conditionLoop = QEventLoop()
        self.conditionLoop.exec_()

    def sendConditionstop(self, sNo, cName, cIndex):
        print("finish to send condition")

        self.dynamicCall("SendConditionStop(QString, QString, int)", sNo, cName, cIndex)

    def setInputValue(self, id, value):
        print(id, value)
        if id == '종목코드' :
            self.stock_code = value
        self.dynamicCall("SetInputValue(QString, QString)", id, value)

    def commRqData(self, rqName, trCode, prevNext, sNo):
        print(rqName, trCode, prevNext, sNo)
        self.dynamicCall("CommRqData(QString, QString, int, QString)", rqName, trCode, prevNext, sNo)
        self.conditionLoop = QEventLoop()
        self.conditionLoop.exec_()

    def getCommdataEx(self,trCode, recordName):
        data = self.dynamicCall("GetCommDataEx(QString, QString)", trCode, recordName)
        return data

    def _opt10081(self, trCode, rqName):
        data = np.array(self.getCommdataEx(trCode, rqName))
        df = pd.DataFrame(data[:, 1:8], columns=['Close', 'Volume', 'Volume2', 'Date', 'Open', 'High', 'Low'])
        df.set_index('Date', inplace= True)
        df.astype(float)
        return df

    def get_codes(self):
        self.getConditionLoad()
        conditions = self.getConditionNameList()
        self.sendCondition(1010,conditions[0][1],conditions[0][0],0)
        return self.tr_condition_data

if __name__ == "__main__" :
    app = QApplication(sys.argv)
    kiwoom = kiwoomAPI()
    kiwoom.commConnect()
    codes = kiwoom.get_codes()
    print(codes[0])

    kiwoom.setInputValue("종목코드", codes[0])
    kiwoom.setInputValue("기준일자", "20210307")
    kiwoom.setInputValue("수정가구분", 1)
    kiwoom.commRqData("opt10081", "opt10081", 0, "0101")

