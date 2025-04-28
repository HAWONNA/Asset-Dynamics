"""
옵션 시장 적용 (Option Market Application)

기하 브라운 운동(GBM)으로 시뮬레이션된 주가 경로를 옵션 평가에 적용하는 모듈입니다.
몬테카를로 시뮬레이션을 통해 옵션의 가치와 수익률을 계산합니다.

이 모듈은 콜옵션과 풋옵션 모두 지원하며, 다양한 전략 분석 및 페이오프 다이어그램을 시각화합니다.
Black-Scholes 모델의 가정을 따르는 시뮬레이션을 통해 실제 옵션 가격을 추정합니다.
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import pandas as pd
import seaborn as sns

def simulate_option_market(price_paths, strike_price, option_type='call', risk_free_rate=0.03, plot=False):
    """
    시뮬레이션된 가격 경로를 사용하여, 옵션의 행사 여부와 수익률을 계산합니다.
    
    Parameters:
    -----------
    price_paths : list
        시뮬레이션된 자산 가격 경로 리스트
    strike_price : float
        옵션 행사가격 (K)
    option_type : str, optional
        옵션 유형, 'call' 또는 'put' (기본값: 'call')
    risk_free_rate : float, optional
        무위험 이자율 (연간화된 값, 기본값: 0.03)
    plot : bool, optional
        결과를 시각화할지 여부 (기본값: False)
    
    Returns:
    --------
    list
        옵션 수익률 리스트
    """
    # 각 경로의 마지막 가격 추출
    final_prices = [path[-1] for path in price_paths]
    
    # 옵션 수익 계산
    if option_type.lower() == 'call':
        # 콜 옵션: max(0, S_T - K)
        payoffs = [max(0, price - strike_price) for price in final_prices]
    else:
        # 풋 옵션: max(0, K - S_T)
        payoffs = [max(0, strike_price - price) for price in final_prices]
    
    # 옵션 행사 여부
    exercised = [payoff > 0 for payoff in payoffs]
    num_exercised = sum(exercised)
    exercise_ratio = num_exercised / len(price_paths)
    
    # 수익률 계산 (행사가격 대비)
    returns = [payoff / strike_price for payoff in payoffs]
    
    # 결과 요약
    if plot:
        # 행사 여부 파이 차트
        plt.figure(figsize=(10, 6))
        plt.subplot(1, 2, 1)
        labels = ['행사함', '행사하지 않음']
        sizes = [num_exercised, len(price_paths) - num_exercised]
        plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90)
        plt.title(f'{option_type.capitalize()} 옵션 행사 비율')
        
        # 수익률 분포
        plt.subplot(1, 2, 2)
        positive_returns = [r for r in returns if r > 0]
        if positive_returns:
            plt.hist(positive_returns, bins=30, alpha=0.7, density=True)
            plt.title(f'행사된 {option_type.capitalize()} 옵션의 수익률 분포')
            plt.xlabel('수익률 (행사가격 대비)')
            plt.ylabel('확률 밀도')
            plt.grid(True)
        else:
            plt.text(0.5, 0.5, '행사된 옵션이 없습니다', ha='center', va='center')
        
        plt.tight_layout()
        plt.show()
    
    return returns

def calculate_option_price(price_paths, strike_price, option_type='call', risk_free_rate=0.03, T=1.0):
    """
    몬테카를로 시뮬레이션을 통해 옵션 가격을 계산합니다.
    
    Parameters:
    -----------
    price_paths : list
        시뮬레이션된 자산 가격 경로 리스트
    strike_price : float
        옵션 행사가격 (K)
    option_type : str, optional
        옵션 유형, 'call' 또는 'put' (기본값: 'call')
    risk_free_rate : float, optional
        무위험 이자율 (연간화된 값, 기본값: 0.03)
    T : float, optional
        만기까지의 시간 (년, 기본값: 1.0)
    
    Returns:
    --------
    float
        몬테카를로 시뮬레이션으로 계산된 옵션 가격
    """
    # 각 경로의 마지막 가격 추출
    final_prices = [path[-1] for path in price_paths]
    
    # 옵션 수익 계산
    if option_type.lower() == 'call':
        # 콜 옵션: max(0, S_T - K)
        payoffs = [max(0, price - strike_price) for price in final_prices]
    else:
        # 풋 옵션: max(0, K - S_T)
        payoffs = [max(0, strike_price - price) for price in final_prices]
    
    # 할인 요인
    discount_factor = np.exp(-risk_free_rate * T)
    
    # 옵션 가격 = 기대 수익의 현재 가치
    option_price = discount_factor * np.mean(payoffs)
    
    return option_price

def analyze_option_strategies(price_paths, initial_price, strategies, risk_free_rate=0.03, T=1.0):
    """
    다양한 옵션 전략의 수익 분포를 분석합니다.
    
    Parameters:
    -----------
    price_paths : list
        시뮬레이션된 자산 가격 경로 리스트
    initial_price : float
        초기 자산 가격
    strategies : list of dict
        각 전략을 기술하는 딕셔너리 리스트
        예: [{'name': 'Long Call', 'type': 'call', 'strike': 110, 'position': 'long'}]
    risk_free_rate : float, optional
        무위험 이자율 (연간화된 값, 기본값: 0.03)
    T : float, optional
        만기까지의 시간 (년, 기본값: 1.0)
    
    Returns:
    --------
    dict
        각 전략의 분석 결과
    """
    # 각 경로의 마지막 가격 추출
    final_prices = [path[-1] for path in price_paths]
    
    results = {}
    
    for strategy in strategies:
        strategy_name = strategy['name']
        option_type = strategy['type']
        strike_price = strategy['strike']
        position = strategy['position']  # 'long' 또는 'short'
        
        # 옵션 프리미엄 계산
        premium = calculate_option_price(
            price_paths=price_paths,
            strike_price=strike_price,
            option_type=option_type,
            risk_free_rate=risk_free_rate,
            T=T
        )
        
        # 만기 시점의 수익 계산
        if option_type.lower() == 'call':
            # 콜 옵션: max(0, S_T - K)
            payoffs = [max(0, price - strike_price) for price in final_prices]
        else:
            # 풋 옵션: max(0, K - S_T)
            payoffs = [max(0, strike_price - price) for price in final_prices]
        
        # 포지션에 따른 부호 조정
        multiplier = 1 if position.lower() == 'long' else -1
        
        # 순이익 = 수익 - 프리미엄 (long) 또는 프리미엄 - 수익 (short)
        net_profits = [(payoff - premium) * multiplier for payoff in payoffs]
        
        # 투자 수익률 (ROI) = 순이익 / 프리미엄
        rois = [profit / premium for profit in net_profits]
        
        # 결과 저장
        results[strategy_name] = {
            'premium': premium,
            'mean_payoff': np.mean(payoffs),
            'mean_net_profit': np.mean(net_profits),
            'mean_roi': np.mean(rois),
            'std_roi': np.std(rois),
            'positive_roi_ratio': np.mean([roi > 0 for roi in rois]),
            'payoffs': payoffs,
            'net_profits': net_profits,
            'rois': rois
        }
    
    return results

def plot_option_payoff_diagram(strikes, initial_price, volatility, risk_free_rate, T, option_types):
    """
    옵션의 수익 구조(payoff diagram)를 시각화합니다.
    
    Parameters:
    -----------
    strikes : list
        분석할 행사가격 목록
    initial_price : float
        초기 자산 가격
    volatility : float
        변동성 (연간화된 값)
    risk_free_rate : float
        무위험 이자율 (연간화된 값)
    T : float
        만기까지의 시간 (년)
    option_types : list
        분석할 옵션 유형 목록 ('call', 'put')
    """
    # 주가 범위 설정
    price_range = np.linspace(initial_price * 0.5, initial_price * 1.5, 100)
    
    plt.figure(figsize=(12, 6))
    
    for option_type in option_types:
        for strike in strikes:
            # 옵션 프리미엄 계산 (Black-Scholes 공식)
            d1 = (np.log(initial_price / strike) + (risk_free_rate + 0.5 * volatility**2) * T) / (volatility * np.sqrt(T))
            d2 = d1 - volatility * np.sqrt(T)
            
            if option_type == 'call':
                premium = initial_price * stats.norm.cdf(d1) - strike * np.exp(-risk_free_rate * T) * stats.norm.cdf(d2)
                payoffs = [max(0, price - strike) for price in price_range]
                label = f'콜 옵션 (K={strike})'
            else:
                premium = strike * np.exp(-risk_free_rate * T) * stats.norm.cdf(-d2) - initial_price * stats.norm.cdf(-d1)
                payoffs = [max(0, strike - price) for price in price_range]
                label = f'풋 옵션 (K={strike})'
            
            # 순이익 계산
            net_profits = [payoff - premium for payoff in payoffs]
            
            # 수익 곡선 그리기
            plt.plot(price_range, net_profits, label=label)
    
    # 기준선 추가
    plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    plt.axvline(x=initial_price, color='k', linestyle='--', alpha=0.3, label='현재 가격')
    
    plt.title('옵션 수익 구조 (Payoff Diagram)')
    plt.xlabel('만기 시점의 자산 가격')
    plt.ylabel('순이익')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

def main():
    """
    테스트 실행 함수
    """
    # 파라미터 설정
    initial_price = 100.0
    annual_drift = 0.08  # 8% 연간 기대 수익률
    annual_volatility = 0.2  # 20% 연간 변동성
    simulation_days = 252  # 1년 (거래일)
    num_paths = 10000
    risk_free_rate = 0.03  # 3% 무위험 이자율
    
    # GBM 시뮬레이션
    from gbm import simulate_gbm
    
    print("기하 브라운 운동 (GBM) 시뮬레이션 중...")
    price_paths = simulate_gbm(
        initial_price=initial_price,
        drift=annual_drift,
        volatility=annual_volatility,
        days=simulation_days,
        num_paths=num_paths
    )
    
    # 다양한 행사가격 설정
    atm_strike = initial_price  # At-the-money
    itm_call_strike = initial_price * 0.9  # In-the-money (for call)
    otm_call_strike = initial_price * 1.1  # Out-of-the-money (for call)
    
    # 옵션 시장 적용
    print("\n콜 옵션 수익률 계산 중...")
    
    # 1. ATM 콜 옵션
    print("\nATM 콜 옵션 (K = 현재 가격):")
    atm_call_returns = simulate_option_market(
        price_paths=price_paths,
        strike_price=atm_strike,
        option_type='call',
        risk_free_rate=risk_free_rate,
        plot=True
    )
    
    # 2. ITM 콜 옵션
    print("\nITM 콜 옵션 (K = 현재 가격의 90%):")
    itm_call_returns = simulate_option_market(
        price_paths=price_paths,
        strike_price=itm_call_strike,
        option_type='call',
        risk_free_rate=risk_free_rate,
        plot=True
    )
    
    # 3. OTM 콜 옵션
    print("\nOTM 콜 옵션 (K = 현재 가격의 110%):")
    otm_call_returns = simulate_option_market(
        price_paths=price_paths,
        strike_price=otm_call_strike,
        option_type='call',
        risk_free_rate=risk_free_rate,
        plot=True
    )
    
    # 옵션 가격 계산
    atm_call_price = calculate_option_price(
        price_paths=price_paths,
        strike_price=atm_strike,
        option_type='call',
        risk_free_rate=risk_free_rate
    )
    
    itm_call_price = calculate_option_price(
        price_paths=price_paths,
        strike_price=itm_call_strike,
        option_type='call',
        risk_free_rate=risk_free_rate
    )
    
    otm_call_price = calculate_option_price(
        price_paths=price_paths,
        strike_price=otm_call_strike,
        option_type='call',
        risk_free_rate=risk_free_rate
    )
    
    print("\n몬테카를로 시뮬레이션을 통한 옵션 가격:")
    print(f"ATM 콜 옵션 가격: {atm_call_price:.4f}")
    print(f"ITM 콜 옵션 가격: {itm_call_price:.4f}")
    print(f"OTM 콜 옵션 가격: {otm_call_price:.4f}")
    
    # 옵션 전략 분석
    strategies = [
        {'name': 'Long ATM Call', 'type': 'call', 'strike': atm_strike, 'position': 'long'},
        {'name': 'Long ITM Call', 'type': 'call', 'strike': itm_call_strike, 'position': 'long'},
        {'name': 'Long OTM Call', 'type': 'call', 'strike': otm_call_strike, 'position': 'long'},
        {'name': 'Short ATM Call', 'type': 'call', 'strike': atm_strike, 'position': 'short'},
        {'name': 'Long ATM Put', 'type': 'put', 'strike': atm_strike, 'position': 'long'},
        {'name': 'Short ATM Put', 'type': 'put', 'strike': atm_strike, 'position': 'short'}
    ]
    
    strategy_results = analyze_option_strategies(
        price_paths=price_paths,
        initial_price=initial_price,
        strategies=strategies,
        risk_free_rate=risk_free_rate
    )
    
    # 전략별 결과 출력
    print("\n옵션 전략 분석 결과:")
    for name, result in strategy_results.items():
        print(f"\n{name}:")
        print(f"  프리미엄: {result['premium']:.4f}")
        print(f"  평균 수익: {result['mean_payoff']:.4f}")
        print(f"  평균 순이익: {result['mean_net_profit']:.4f}")
        print(f"  평균 ROI: {result['mean_roi']:.4f}")
        print(f"  ROI 표준편차: {result['std_roi']:.4f}")
        print(f"  양의 ROI 비율: {result['positive_roi_ratio']:.4f}")
    
    # 수익 구조 시각화
    plot_option_payoff_diagram(
        strikes=[atm_strike, itm_call_strike, otm_call_strike],
        initial_price=initial_price,
        volatility=annual_volatility,
        risk_free_rate=risk_free_rate,
        T=1.0,
        option_types=['call', 'put']
    )

if __name__ == "__main__":
    main() 