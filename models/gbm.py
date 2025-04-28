"""
기하 브라운 운동 (Geometric Brownian Motion, GBM)

기하 브라운 운동은 금융에서 가장 널리 사용되는 연속 시간 모델 중 하나입니다.
Black-Scholes 옵션 가격 모델의 기초가 되는 이 모델은 주식 가격이 로그 정규 분포를 따른다고 가정합니다.

수학적 표현:
dS(t) = μS(t)dt + σS(t)dW(t)

이를 풀면:
S(t) = S(0)exp((μ - σ²/2)t + σW(t))

여기서:
- S(t)는 시간 t에서의 자산 가격
- S(0)는 초기 자산 가격
- μ는 기대 수익률(드리프트)
- σ는 변동성
- W(t)는 위너 과정
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats

def simulate_gbm(initial_price, drift, volatility, days, num_paths=1, dt=1/252, plot=False):
    """
    기하 브라운 운동(GBM)을 사용하여 자산 가격 경로를 시뮬레이션합니다.
    
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
    num_paths : int, optional
        생성할 경로 수 (기본값: 1)
    dt : float, optional
        시간 증분 (기본값: 1/252, 1년 중 하루)
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
    
    # GBM 모델 매개변수 계산
    drift_adj = drift - 0.5 * volatility**2  # 조정된 드리프트
    
    for _ in range(num_paths):
        # 위너 과정 시뮬레이션
        dW = np.random.standard_normal(steps) * np.sqrt(dt)
        W = np.cumsum(dW)
        
        # 완전한 경로를 한 번에 계산
        # S(t) = S(0)exp((μ - σ²/2)t + σW(t))
        t_values = np.arange(1, steps + 1) * dt
        exponent = drift_adj * t_values + volatility * W
        price_path = np.zeros(steps + 1)
        price_path[0] = initial_price
        price_path[1:] = initial_price * np.exp(exponent)
        
        paths.append(price_path)
    
    # 결과 시각화
    if plot and num_paths <= 10:
        plt.figure(figsize=(12, 6))
        for i, path in enumerate(paths):
            plt.plot(time, path, label=f'경로 {i+1}')
        
        plt.title('기하 브라운 운동 (GBM) 시뮬레이션')
        plt.xlabel('시간 (년)')
        plt.ylabel('자산 가격')
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show()
    elif plot:
        # 경로가 너무 많을 경우 일부만 시각화
        sample_size = min(10, num_paths)
        sampled_indices = np.random.choice(num_paths, sample_size, replace=False)
        
        plt.figure(figsize=(12, 6))
        for i, idx in enumerate(sampled_indices):
            plt.plot(time, paths[idx], label=f'경로 {idx+1}')
        
        plt.title(f'기하 브라운 운동 (GBM) 시뮬레이션 (총 {num_paths}개 중 {sample_size}개 표시)')
        plt.xlabel('시간 (년)')
        plt.ylabel('자산 가격')
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show()
    
    return paths

def analyze_gbm_paths(paths, initial_price, drift, volatility, dt=1/252):
    """
    기하 브라운 운동 시뮬레이션 결과를 분석합니다.
    
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
    # 최종 가격 추출
    final_prices = [path[-1] for path in paths]
    
    # 전체 기간
    T = len(paths[0]) * dt - dt
    
    # 수익률 계산
    simple_returns = [(price / initial_price) - 1 for price in final_prices]
    log_returns = [np.log(price / initial_price) for price in final_prices]
    
    # 이론적 로그 정규 분포 매개변수
    theoretical_log_mean = (drift - 0.5 * volatility**2) * T
    theoretical_log_std = volatility * np.sqrt(T)
    
    # 이론적 로그 정규 분포에서 기대되는 평균 및 분산
    theoretical_mean = np.exp(theoretical_log_mean + 0.5 * theoretical_log_std**2) - 1
    theoretical_var = (np.exp(theoretical_log_std**2) - 1) * np.exp(2*theoretical_log_mean + theoretical_log_std**2)
    theoretical_std = np.sqrt(theoretical_var)
    
    # 로그 정규 분포 적합
    shape, loc, scale = stats.lognorm.fit(final_prices, floc=0)
    sigma = shape
    mu = np.log(scale)
    
    # 로그 수익률 정규성 테스트
    shapiro_test = stats.shapiro(log_returns)
    
    # 결과 저장
    results = {
        'mean_return': np.mean(simple_returns),
        'std_return': np.std(simple_returns),
        'mean_log_return': np.mean(log_returns),
        'std_log_return': np.std(log_returns),
        'theoretical_mean': theoretical_mean,
        'theoretical_std': theoretical_std,
        'theoretical_log_mean': theoretical_log_mean,
        'theoretical_log_std': theoretical_log_std,
        'lognorm_mu': mu,
        'lognorm_sigma': sigma,
        'shapiro_test': shapiro_test
    }
    
    return results

def simulate_and_analyze_multiple_horizons(initial_price, drift, volatility, horizons, num_paths=10000):
    """
    여러 투자 기간에 대해 GBM 시뮬레이션을 수행하고 분석합니다.
    
    Parameters:
    -----------
    initial_price : float
        초기 자산 가격
    drift : float
        기대 수익률 (μ, 연간화된 값)
    volatility : float
        변동성 (σ, 연간화된 값)
    horizons : list
        분석할 투자 기간 목록 (년 단위)
    num_paths : int, optional
        각 기간별 생성할 경로 수 (기본값: 10000)
    
    Returns:
    --------
    dict
        각 기간별 분석 결과
    """
    results = {}
    
    for horizon in horizons:
        print(f"기간 {horizon}년에 대한 GBM 시뮬레이션 수행 중...")
        days = int(horizon * 252)  # 거래일 수
        
        # GBM 시뮬레이션
        paths = simulate_gbm(
            initial_price=initial_price,
            drift=drift,
            volatility=volatility,
            days=days,
            num_paths=num_paths,
            plot=False
        )
        
        # 결과 분석
        analysis = analyze_gbm_paths(
            paths=paths,
            initial_price=initial_price,
            drift=drift,
            volatility=volatility
        )
        
        results[horizon] = analysis
    
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
    num_paths = 1000
    
    # GBM 시뮬레이션
    print("기하 브라운 운동 (GBM) 시뮬레이션 중...")
    paths = simulate_gbm(
        initial_price=initial_price,
        drift=annual_drift,
        volatility=annual_volatility,
        days=simulation_days,
        num_paths=num_paths,
        plot=True
    )
    
    # 결과 분석
    analysis = analyze_gbm_paths(
        paths=paths,
        initial_price=initial_price,
        drift=annual_drift,
        volatility=annual_volatility
    )
    
    # 분석 결과 출력
    print("\nGBM 분석 결과:")
    print(f"  평균 수익률: {analysis['mean_return']:.4f} (이론값: {analysis['theoretical_mean']:.4f})")
    print(f"  수익률 표준편차: {analysis['std_return']:.4f} (이론값: {analysis['theoretical_std']:.4f})")
    print(f"  평균 로그 수익률: {analysis['mean_log_return']:.4f} (이론값: {analysis['theoretical_log_mean']:.4f})")
    print(f"  로그 수익률 표준편차: {analysis['std_log_return']:.4f} (이론값: {analysis['theoretical_log_std']:.4f})")
    print(f"  로그 수익률 정규성 테스트 (Shapiro): p-value = {analysis['shapiro_test'][1]:.6f}")
    
    # 최종 가격 분포 시각화
    final_prices = [path[-1] for path in paths]
    
    plt.figure(figsize=(12, 6))
    
    # 히스토그램
    plt.hist(final_prices, bins=50, alpha=0.6, density=True, label='시뮬레이션 최종 가격')
    
    # 이론적 로그 정규 분포
    x = np.linspace(min(final_prices), max(final_prices), 1000)
    mu = analysis['lognorm_mu']
    sigma = analysis['lognorm_sigma']
    pdf = stats.lognorm.pdf(x, sigma, scale=np.exp(mu))
    plt.plot(x, pdf, 'r-', lw=2, label='적합된 로그 정규 분포')
    
    plt.title('GBM 시뮬레이션의 최종 가격 분포')
    plt.xlabel('최종 가격')
    plt.ylabel('확률 밀도')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    
    # 다양한 투자 기간에 대한 분석
    horizons = [1, 2, 5, 10]  # 1년, 2년, 5년, 10년
    multi_horizon_results = simulate_and_analyze_multiple_horizons(
        initial_price=initial_price,
        drift=annual_drift,
        volatility=annual_volatility,
        horizons=horizons,
        num_paths=5000
    )
    
    # 기간별 결과 출력
    print("\n다양한 투자 기간에 대한 GBM 분석:")
    for horizon, result in multi_horizon_results.items():
        print(f"\n기간: {horizon}년")
        print(f"  평균 수익률: {result['mean_return']:.4f} (이론값: {result['theoretical_mean']:.4f})")
        print(f"  수익률 표준편차: {result['std_return']:.4f} (이론값: {result['theoretical_std']:.4f})")
        print(f"  샤피로 테스트 p-value: {result['shapiro_test'][1]:.6f}")
    
    # 기간별 수익률 분포 시각화
    plt.figure(figsize=(12, 8))
    
    for i, horizon in enumerate(horizons):
        result = multi_horizon_results[horizon]
        final_prices = [path[-1] for path in paths]
        simple_returns = [(price / initial_price) - 1 for price in final_prices]
        
        plt.subplot(2, 2, i+1)
        plt.hist(simple_returns, bins=30, alpha=0.7, density=True)
        plt.title(f'{horizon}년 투자 기간의 수익률 분포')
        plt.xlabel('수익률')
        plt.ylabel('확률 밀도')
        plt.grid(True)
    
    plt.tight_layout()
    plt.show()
    
    # GBM의 핵심 특성 출력
    print("\nGBM의 핵심 특성:")
    print("  1. 자산 가격은 항상 양수를 유지합니다.")
    print("  2. 로그 수익률은 정규 분포를 따릅니다.")
    print("  3. 자산 가격은 로그 정규 분포를 따릅니다.")
    print("  4. 시간에 따른 수익률의 분산은 선형적으로 증가합니다.")
    print("  5. 이 모델은 Black-Scholes 옵션 가격 모델의 기초입니다.")

if __name__ == "__main__":
    main() 