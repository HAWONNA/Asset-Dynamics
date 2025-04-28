"""
가산적 모델 (Addictive Model)

자산 가격의 변화가 이전 가격에 고정된 증분을 더하는 방식으로 진행된다고 가정하는 모델입니다.
수학적 표현: S(t+1) = S(t) + μ + σε

여기서:
- S(t)는 시간 t에서의 자산 가격
- μ는 평균 변화량(드리프트)
- σ는 변동성(표준편차)
- ε는 표준 정규 분포를 따르는 랜덤 변수
"""

import numpy as np
import matplotlib.pyplot as plt

def simulate_addictive_model(initial_price, drift, volatility, days, num_paths=1, plot=False):
    """
    가산적 모델을 사용하여 자산 가격 경로를 시뮬레이션합니다.
    
    Parameters:
    -----------
    initial_price : float
        초기 자산 가격
    drift : float
        평균 드리프트 (μ)
    volatility : float
        변동성 (σ)
    days : int
        시뮬레이션 일수
    num_paths : int, optional
        생성할 경로 수 (기본값: 1)
    plot : bool, optional
        결과를 시각화할지 여부 (기본값: False)
    
    Returns:
    --------
    list
        시뮬레이션된 가격 경로 리스트
    """
    # 일별 시간 배열
    time = np.arange(0, days + 1)
    
    # 결과 저장 리스트
    paths = []
    
    for _ in range(num_paths):
        # 랜덤 샘플 생성 (표준 정규 분포)
        random_samples = np.random.standard_normal(days)
        
        # 가격 경로 초기화
        price_path = np.zeros(days + 1)
        price_path[0] = initial_price
        
        # 가산적 모델 적용: S(t+1) = S(t) + μ + σε
        for t in range(days):
            price_path[t+1] = price_path[t] + drift + volatility * random_samples[t]
        
        paths.append(price_path)
    
    # 결과 시각화
    if plot:
        plt.figure(figsize=(12, 6))
        for i, path in enumerate(paths):
            plt.plot(time, path, label=f'경로 {i+1}')
        
        plt.title('가산적 모델 (Addictive Model) 시뮬레이션')
        plt.xlabel('시간 (일)')
        plt.ylabel('자산 가격')
        plt.grid(True)
        
        if num_paths <= 10:
            plt.legend()
            
        plt.tight_layout()
        plt.show()
    
    return paths

def main():
    """
    테스트 실행 함수
    """
    # 파라미터 설정
    initial_price = 100
    daily_drift = 0.001  # 일별 평균 증분
    daily_volatility = 0.02  # 일별 변동성
    simulation_days = 252  # 1년 (거래일)
    num_simulations = 5
    
    # 가산적 모델 시뮬레이션
    print("가산적 모델 (Addictive Model) 시뮬레이션 중...")
    paths = simulate_addictive_model(
        initial_price=initial_price,
        drift=daily_drift,
        volatility=daily_volatility,
        days=simulation_days,
        num_paths=num_simulations,
        plot=True
    )
    
    # 결과 분석
    final_prices = [path[-1] for path in paths]
    print(f"최종 가격 통계:")
    print(f"  평균: {np.mean(final_prices):.2f}")
    print(f"  최소: {np.min(final_prices):.2f}")
    print(f"  최대: {np.max(final_prices):.2f}")
    print(f"  표준편차: {np.std(final_prices):.2f}")
    
    # 단점 강조: 음수 가격 발생 가능
    negative_days = [np.sum(path < 0) for path in paths]
    total_negative_days = sum(negative_days)
    
    if total_negative_days > 0:
        print(f"주의: 총 {total_negative_days}일 동안 음수 가격이 발생했습니다.")
        print("이는 가산적 모델의 주요 단점입니다.")

if __name__ == "__main__":
    main() 