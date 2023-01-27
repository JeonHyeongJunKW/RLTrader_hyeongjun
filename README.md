# RLTrader_hyeongjun
강화학습기반의 주식 투자 프로그램

### 목표
강화학습과 tensorflow라이브러리를 사용하여, 하나의 종목에 대해서 

모델 사용 10일간 포트폴리오 수익률 10프로를 목표로 합니다.

### 프로그래밍 기술
- reinforce learning
- tensorflow
- numpy
- cuda
- python 

### 기능
- 코드 모듈화
- 결과 시각화
- 주식 정보 가공
- 딥러닝 모델학습 및 저장
- (핵심기능) 투자 행동 결정 

### 일정
- 1주차 : 기본 예제코드를 구현합니다.
- 2주차 : 기본 모델 학습 및 성능을 분석합니다.
- 3주차 : 기존 예제코드를 개선합니다.
- 4주차 : 개선 모델 학습 및 성능 분석
- 5주차 : 마무리 및 실제 테스트 준비

## 코드설명

### networks.py
- DNN, LSTM, CNN 구조로 강화학습모델을 생성합니다.
- tensorflow가 필요합니다.

### environment.py
- 주가 정보등을 얻어오고, 전처리 합니다.
- observation을 얻어오는 내용을 포함합니다.

### agent.py
- 모델의 출력과 observation으로부터 주식 코드를 제어합니다.


### 참고문헌
- [rltrader 예제 코드](https://github.com/quantylab/rltrader)