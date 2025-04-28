"""
Asset Dynamics - 자산 가격 모델링 및 시뮬레이션

이 스크립트는 다양한 자산 가격 모델을 사용하여 
금융 자산의 가격 움직임을 시뮬레이션하고 분석합니다.

주요 모델:
1. 가산적 모델 (Addictive Model)
2. 승법적 모델 (Multiplicative Model)
3. 랜덤 워크 (Random Walk)
4. 위너 과정 (Wiener Process)
5. 기하 위너 과정 (Geometric Wiener Process)
6. 기하 브라운 운동 (Geometric Brownian Motion, GBM)
7. 옵션 시장 적용 (Option Market Application)
"""

import pandas as pd
import pandas_datareader as web
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
import seaborn as sns
import scipy.stats as stats
from datetime import datetime, timedelta

# 모듈 임포트
from models.addictive_model import simulate_addictive_model
from models.multiplicative_model import simulate_multiplicative_model
from models.random_walk import simulate_random_walk
from models.wiener_process import simulate_wiener_process
from models.geometric_wiener_process import simulate_geometric_wiener_process
from models.gbm import simulate_gbm
from models.option_market import simulate_option_market

def fetch_stock_data(ticker='META', start_date='2020-01-01', end_date='2020-12-31'):
    """
    주식 데이터를 가져오는 함수
    
    Parameters:
    -----------
    ticker : str
        주식 티커 심볼
    start_date : str
        시작 날짜 (YYYY-MM-DD)
    end_date : str
        종료 날짜 (YYYY-MM-DD)
    
    Returns:
    --------
    pd.Series
        주식의 조정 종가 시계열
    """
    print(f"{ticker} 주식 데이터를 {start_date}부터 {end_date}까지 가져오는 중...")
    df = web.get_data_yahoo(ticker, start=start_date, end=end_date)['Adj Close']
    print(f"데이터 가져오기 완료: {len(df)} 데이터 포인트")
    return df

def calculate_returns(prices):
    """
    가격 시계열에서 수익률을 계산하는 함수
    
    Parameters:
    -----------
    prices : pd.Series
        자산 가격 시계열
    
    Returns:
    --------
    tuple
        (산술 수익률, 로그 수익률)
    """
    # 산술 수익률
    arithmetic_returns = prices.pct_change().dropna()
    
    # 로그 가격 및 로그 수익률
    log_prices = np.log(prices)
    log_returns = (log_prices.shift(-1) - log_prices).dropna()
    
    return arithmetic_returns, log_returns

def main():
    """
    메인 실행 함수
    """
    # 설정
    ticker = 'META'
    start_date = '2020-01-01'
    end_date = '2020-12-31'
    prediction_days = 100
    num_scenarios = 10000
    seed = 42
    
    # 랜덤 시드 설정
    np.random.seed(seed)
    
    # 데이터 가져오기
    prices = fetch_stock_data(ticker, start_date, end_date)
    
    # 수익률 계산
    returns, log_returns = calculate_returns(prices)
    
    # 모델 파라미터 계산
    drift = log_returns.mean()
    volatility = np.sqrt(log_returns.var())
    
    print(f"\n모델 파라미터:")
    print(f"드리프트(μ): {drift:.6f}")
    print(f"변동성(σ): {volatility:.6f}")
    
    # 초기 가격 설정 (마지막 가격)
    initial_price = prices.iloc[-1]
    
    print(f"\n각 모델의 시뮬레이션을 수행합니다...")
    
    # 1. 가산적 모델
    addictive_paths = simulate_addictive_model(
        initial_price=initial_price,
        drift=drift,
        volatility=volatility,
        days=prediction_days,
        num_paths=5
    )
    
    # 2. 승법적 모델
    multiplicative_paths = simulate_multiplicative_model(
        initial_price=initial_price,
        drift=drift,
        volatility=volatility,
        days=prediction_days,
        num_paths=5
    )
    
    # 3. 랜덤 워크
    random_walk_paths = simulate_random_walk(
        initial_price=initial_price,
        volatility=volatility,
        days=prediction_days,
        num_paths=5
    )
    
    # 4. 위너 과정
    wiener_paths = simulate_wiener_process(
        initial_value=0,
        days=prediction_days,
        num_paths=5,
        dt=1/252
    )
    
    # 5. 기하 위너 과정
    geo_wiener_paths = simulate_geometric_wiener_process(
        initial_price=initial_price,
        drift=drift,
        volatility=volatility,
        days=prediction_days,
        num_paths=5,
        dt=1/252
    )
    
    # 6. 기하 브라운 운동 (GBM)
    gbm_paths = simulate_gbm(
        initial_price=initial_price,
        drift=drift,
        volatility=volatility,
        days=prediction_days,
        num_paths=num_scenarios,
        dt=1/252
    )
    
    # 7. 옵션 시장 적용
    strike_price = initial_price * 1.2  # 행사가격: 현재 가격의 120%
    option_returns = simulate_option_market(
        price_paths=gbm_paths,
        strike_price=strike_price,
        option_type='call'
    )
    
    print("\n시뮬레이션 완료!")
    print(f"가산적 모델: {len(addictive_paths)} 경로")
    print(f"승법적 모델: {len(multiplicative_paths)} 경로")
    print(f"랜덤 워크: {len(random_walk_paths)} 경로")
    print(f"위너 과정: {len(wiener_paths)} 경로")
    print(f"기하 위너 과정: {len(geo_wiener_paths)} 경로")
    print(f"기하 브라운 운동: {len(gbm_paths)} 경로")
    print(f"옵션 수익률: {len(option_returns)} 시나리오")
    
    # 옵션 행사 통계
    exercised_options = np.sum(np.array(option_returns) > 0)
    print(f"\n옵션 행사 결과:")
    print(f"행사된 옵션 수: {exercised_options}")
    print(f"행사되지 않은 옵션 수: {len(option_returns) - exercised_options}")
    print(f"행사 비율: {exercised_options / len(option_returns):.2%}")
    
    # 행사된 옵션의 수익률 통계
    exercised_returns = np.array(option_returns)[np.array(option_returns) > 0]
    if len(exercised_returns) > 0:
        print(f"\n행사된 옵션 수익률 통계:")
        print(f"평균 수익률: {np.mean(exercised_returns):.4f}")
        print(f"표준편차: {np.std(exercised_returns):.4f}")
        print(f"최소 수익률: {np.min(exercised_returns):.4f}")
        print(f"최대 수익률: {np.max(exercised_returns):.4f}")

if __name__ == "__main__":
    main() 