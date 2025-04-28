"""
기하 위너 과정 (Geometric Wiener Process)

기하 위너 과정은 위너 과정을 기반으로 하지만, 자산 가격이 로그 정규 분포를 따른다고 가정하는 모델입니다.
수학적 표현: dS(t) = μS(t)dt + σS(t)dW(t)

여기서:
- S(t)는 시간 t에서의 자산 가격
- μ는 기대 수익률(드리프트)
- σ는 변동성
- dW(t)는 위너 과정의 증분
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats

def simulate_geometric_wiener_process(initial_price, drift, volatility, days, dt=1/252, num_paths=1, plot=False):
    """
    기하 위너 과정을 사용하여 자산 가격 경로를 시뮬레이션합니다.
    
    Parameters:
    -----------
    initial_price : float
        초기 자산 가격
    drift : float
        기대 수익률 (μ, 연간화된 값)
    volatility : float
        변동성 (σ, 연간화된 값)
    days : int
        시뮬레이션 일수
    dt : float, optional
        시간 증분 (기본값: 1/252, 1년 중 하루)
    num_paths : int, optional
        생성할 경로 수 (기본값: 1)
    plot : bool, optional
        결과를 시각화할지 여부 (기본값: False)
    
    Returns:
    --------
    list
        시뮬레이션된 가격 경로 리스트
    """
    # 시뮬레이션 스텝 수
    steps = int(days / dt)
    
    # 시간 배열
    time = np.linspace(0, days*dt, steps + 1)
    
    # 결과 저장 리스트
    paths = []
    
    for _ in range(num_paths):
        # 가격 경로 초기화
        price_path = np.zeros(steps + 1)
        price_path[0] = initial_price
        
        # 랜덤 샘플 생성 (표준 정규 분포)
        random_samples = np.random.standard_normal(steps)
        
        # 기하 위너 과정 적용: dS(t) = μS(t)dt + σS(t)dW(t)
        for t in range(steps):
            dW = random_samples[t] * np.sqrt(dt)
            # 오일러-마루야마 방법으로 SDE 근사
            price_path[t+1] = price_path[t] * (1 + drift * dt + volatility * dW)
        
        paths.append(price_path)
    
    # 결과 시각화
    if plot:
        plt.figure(figsize=(12, 6))
        for i, path in enumerate(paths):
            plt.plot(time, path, label=f'경로 {i+1}')
        
        plt.title('기하 위너 과정 (Geometric Wiener Process) 시뮬레이션')
        plt.xlabel('시간 (년)')
        plt.ylabel('자산 가격')
        plt.grid(True)
        
        if num_paths <= 10:
            plt.legend()
            
        plt.tight_layout()
        plt.show()
    
    return paths

def analyze_geometric_wiener_process(paths, initial_price, drift, volatility, dt=1/252):
    """
    기하 위너 과정 시뮬레이션 결과를 분석합니다.
    
    Parameters:
    -----------
    paths : list
        시뮬레이션된 가격 경로 리스트
    initial_price : float
        초기 자산 가격
    drift : float
        기대 수익률 (μ, 연간화된 값)
    volatility : float
        변동성 (σ, 연간화된 값)
    dt : float, optional
        시간 증분 (기본값: 1/252, 1년 중 하루)
    
    Returns:
    --------
    dict
        분석 결과를 담은 딕셔너리
    """
    # 시뮬레이션 기간
    T = len(paths[0]) * dt
    
    # 마지막 가격 추출
    final_prices = [path[-1] for path in paths]
    
    # 수익률 계산
    returns = [(price - initial_price) / initial_price for price in final_prices]
    log_returns = [np.log(price / initial_price) for price in final_prices]
    
    # 이론적 로그 정규 분포 매개변수
    theoretical_mean = (drift - 0.5 * volatility**2) * T
    theoretical_std = volatility * np.sqrt(T)
    
    # 로그 수익률 정규성 테스트
    _, p_value = stats.normaltest(log_returns)
    
    # 결과 저장
    results = {
        'mean_return': np.mean(returns),
        'std_return': np.std(returns),
        'mean_log_return': np.mean(log_returns),
        'std_log_return': np.std(log_returns),
        'theoretical_mean': theoretical_mean,
        'theoretical_std': theoretical_std,
        'normality_p_value': p_value
    }
    
    return results

def main():
    """
    테스트 실행 함수
    """
    # 파라미터 설정
    initial_price = 100.0
    annual_drift = 0.1  # 10% 연간 기대 수익률
    annual_volatility = 0.2  # 20% 연간 변동성
    simulation_days = 252  # 1년 (거래일)
    dt = 1/252  # 일별 (1년 중 하루)
    num_paths = 5
    
    # 기하 위너 과정 시뮬레이션
    print("기하 위너 과정 (Geometric Wiener Process) 시뮬레이션 중...")
    paths = simulate_geometric_wiener_process(
        initial_price=initial_price,
        drift=annual_drift,
        volatility=annual_volatility,
        days=simulation_days,
        dt=dt,
        num_paths=num_paths,
        plot=True
    )
    
    # 결과 분석
    analysis = analyze_geometric_wiener_process(
        paths=paths, 
        initial_price=initial_price,
        drift=annual_drift,
        volatility=annual_volatility,
        dt=dt
    )
    
    # 분석 결과 출력
    print("\n기하 위너 과정 분석 결과:")
    print(f"  평균 수익률: {analysis['mean_return']:.4f}")
    print(f"  수익률 표준편차: {analysis['std_return']:.4f}")
    print(f"  평균 로그 수익률: {analysis['mean_log_return']:.4f} (이론값: {analysis['theoretical_mean']:.4f})")
    print(f"  로그 수익률 표준편차: {analysis['std_log_return']:.4f} (이론값: {analysis['theoretical_std']:.4f})")
    print(f"  로그 수익률 정규성 테스트 p-value: {analysis['normality_p_value']:.6f}")
    
    # 로그 수익률 분포 시각화
    final_prices = [path[-1] for path in paths]
    log_returns = [np.log(price / initial_price) for price in final_prices]
    
    plt.figure(figsize=(10, 6))
    plt.hist(log_returns, bins=30, alpha=0.7, density=True, label='시뮬레이션 로그 수익률')
    
    # 이론적 정규 분포
    x = np.linspace(min(log_returns), max(log_returns), 100)
    plt.plot(x, stats.norm.pdf(x, analysis['theoretical_mean'], analysis['theoretical_std']), 
             'r-', lw=2, label='이론적 정규 분포')
    
    plt.title('기하 위너 과정의 로그 수익률 분포')
    plt.xlabel('로그 수익률')
    plt.ylabel('확률 밀도')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    
    # 기하 위너 과정과 GBM 비교
    print("\n기하 위너 과정과 기하 브라운 운동 (GBM):")
    print("  기하 위너 과정은 기하 브라운 운동의 기초가 됩니다.")
    print("  두 모델 모두 로그 수익률이 정규 분포를 따른다고 가정합니다.")
    print("  GBM은 연속 시간에서의 Itô 확률적 미분방정식으로 더 정확하게 정의됩니다.")
    print("  실제로 많은 금융 모델링에서 두 용어는 상호 교환적으로 사용되기도 합니다.")

if __name__ == "__main__":
    main() 