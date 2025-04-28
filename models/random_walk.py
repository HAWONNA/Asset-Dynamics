"""
랜덤 워크 (Random Walk)

자산 가격의 다음 단계 변화가 현재 가격과 독립적인 랜덤 값에 의해 결정된다고 가정하는 모델입니다.
수학적 표현: S(t+1) = S(t) + ε

여기서:
- S(t)는 시간 t에서의 자산 가격
- ε는 독립적이고 동일하게 분포된(i.i.d.) 랜덤 변수

효율적 시장 가설(EMH)의 기초가 되는 모델로, 약형 효율성 시장에서 나타납니다.
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats

def simulate_random_walk(initial_price, volatility, days, num_paths=1, plot=False):
    """
    랜덤 워크 모델을 사용하여 자산 가격 경로를 시뮬레이션합니다.
    
    Parameters:
    -----------
    initial_price : float
        초기 자산 가격
    volatility : float
        랜덤 충격의 표준편차
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
        random_samples = np.random.standard_normal(days) * volatility
        
        # 가격 경로 초기화
        price_path = np.zeros(days + 1)
        price_path[0] = initial_price
        
        # 랜덤 워크 적용: S(t+1) = S(t) + ε
        for t in range(days):
            price_path[t+1] = price_path[t] + random_samples[t]
        
        paths.append(price_path)
    
    # 결과 시각화
    if plot:
        plt.figure(figsize=(12, 6))
        for i, path in enumerate(paths):
            plt.plot(time, path, label=f'경로 {i+1}')
        
        plt.title('랜덤 워크 (Random Walk) 시뮬레이션')
        plt.xlabel('시간 (일)')
        plt.ylabel('자산 가격')
        plt.grid(True)
        
        if num_paths <= 10:
            plt.legend()
            
        plt.tight_layout()
        plt.show()
    
    return paths

def analyze_random_walk(price_paths):
    """
    랜덤 워크 모델에서 생성된 가격 경로를 분석합니다.
    
    Parameters:
    -----------
    price_paths : list
        시뮬레이션된 가격 경로 리스트
    
    Returns:
    --------
    dict
        분석 결과를 담은 딕셔너리
    """
    # 가격 변화 계산
    price_changes = []
    for path in price_paths:
        changes = np.diff(path)
        price_changes.extend(changes)
    
    # 정규성 테스트
    shapiro_test = stats.shapiro(price_changes)
    
    # 자기상관 계산
    autocorr = np.correlate(price_changes, price_changes, mode='full')
    autocorr = autocorr[len(autocorr)//2:]
    autocorr = autocorr / autocorr[0]  # 정규화
    
    # 결과 저장
    results = {
        'mean_change': np.mean(price_changes),
        'std_change': np.std(price_changes),
        'shapiro_test': shapiro_test,
        'autocorrelation': autocorr[:10]  # 처음 10개의 시차에 대한 자기상관
    }
    
    return results

def main():
    """
    테스트 실행 함수
    """
    # 파라미터 설정
    initial_price = 100
    volatility = 1.0  # 변동성 (랜덤 충격의 표준편차)
    simulation_days = 252  # 1년 (거래일)
    num_simulations = 5
    
    # 랜덤 워크 시뮬레이션
    print("랜덤 워크 (Random Walk) 시뮬레이션 중...")
    paths = simulate_random_walk(
        initial_price=initial_price,
        volatility=volatility,
        days=simulation_days,
        num_paths=num_simulations,
        plot=True
    )
    
    # 결과 분석
    analysis = analyze_random_walk(paths)
    
    print("\n랜덤 워크 분석 결과:")
    print(f"  평균 가격 변화: {analysis['mean_change']:.6f}")
    print(f"  가격 변화의 표준편차: {analysis['std_change']:.6f}")
    print(f"  정규성 테스트 (Shapiro-Wilk): p-value = {analysis['shapiro_test'][1]:.6f}")
    
    # 자기상관 시각화
    plt.figure(figsize=(10, 6))
    plt.stem(range(10), analysis['autocorrelation'])
    plt.title('랜덤 워크 모델의 가격 변화 자기상관')
    plt.xlabel('시차 (Lag)')
    plt.ylabel('자기상관')
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    
    # 효율적 시장 가설 관련 설명
    print("\n랜덤 워크 모델과 효율적 시장 가설 (EMH):")
    print("  랜덤 워크 모델은 효율적 시장 가설의 기초가 됩니다.")
    print("  약형 효율적 시장에서는 과거 가격 정보만으로 미래 수익을 예측할 수 없습니다.")
    print("  따라서 가격 변화는 예측 불가능하며, 자기상관이 0에 가까워야 합니다.")
    
    # 가격 변화의 분포 시각화
    plt.figure(figsize=(10, 6))
    all_changes = np.concatenate([np.diff(path) for path in paths])
    plt.hist(all_changes, bins=30, alpha=0.7, density=True)
    
    # 적합한 정규 분포 곡선 추가
    x = np.linspace(min(all_changes), max(all_changes), 100)
    plt.plot(x, stats.norm.pdf(x, np.mean(all_changes), np.std(all_changes)), 'r-', lw=2)
    
    plt.title('랜덤 워크 모델의 가격 변화 분포')
    plt.xlabel('가격 변화')
    plt.ylabel('확률 밀도')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main() 