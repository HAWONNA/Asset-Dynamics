"""
위너 과정 (Wiener Process)

위너 과정(또는 브라운 운동)은 연속 시간에서의 랜덤 워크를 일반화한 확률 과정입니다.
주요 특성:
1. W(0) = 0
2. W(t)는 독립 증분을 가짐
3. W(t+s) - W(t)는 평균 0, 분산 s의 정규 분포를 따름
4. W(t)는 연속 경로를 가짐

수학적 표현: dW(t) = ε√dt

여기서:
- dW(t)는 위너 과정의 증분
- ε는 표준 정규 분포를 따르는 랜덤 변수
- dt는 시간의 미소 증분
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats

def simulate_wiener_process(initial_value=0, days=252, num_paths=1, dt=1/252, plot=False):
    """
    위너 과정(브라운 운동)을 시뮬레이션합니다.
    
    Parameters:
    -----------
    initial_value : float, optional
        초기값 (기본값: 0)
    days : int, optional
        시뮬레이션 일수 (기본값: 252, 1년 거래일)
    num_paths : int, optional
        생성할 경로 수 (기본값: 1)
    dt : float, optional
        시간 증분 (기본값: 1/252, 1년 중 하루)
    plot : bool, optional
        결과를 시각화할지 여부 (기본값: False)
    
    Returns:
    --------
    list
        시뮬레이션된 위너 과정 경로 리스트
    """
    # 시뮬레이션 스텝 수
    steps = int(days / dt)
    
    # 시간 배열
    time = np.linspace(0, days, steps + 1)
    
    # 결과 저장 리스트
    paths = []
    
    for _ in range(num_paths):
        # 랜덤 샘플 생성 (표준 정규 분포)
        random_increments = np.random.standard_normal(steps) * np.sqrt(dt)
        
        # 위너 과정 경로 초기화
        process_path = np.zeros(steps + 1)
        process_path[0] = initial_value
        
        # 위너 과정 증분 누적: W(t+dt) = W(t) + dW(t)
        for t in range(steps):
            process_path[t+1] = process_path[t] + random_increments[t]
        
        paths.append(process_path)
    
    # 결과 시각화
    if plot:
        plt.figure(figsize=(12, 6))
        for i, path in enumerate(paths):
            plt.plot(time, path, label=f'경로 {i+1}')
        
        plt.title('위너 과정 (Wiener Process) 시뮬레이션')
        plt.xlabel('시간 (년)')
        plt.ylabel('값')
        plt.grid(True)
        
        if num_paths <= 10:
            plt.legend()
            
        plt.tight_layout()
        plt.show()
    
    return paths

def analyze_wiener_process(paths, dt=1/252):
    """
    위너 과정의 특성을 분석합니다.
    
    Parameters:
    -----------
    paths : list
        시뮬레이션된 위너 과정 경로 리스트
    dt : float, optional
        시간 증분 (기본값: 1/252, 1년 중 하루)
    
    Returns:
    --------
    dict
        분석 결과를 담은 딕셔너리
    """
    # 증분 계산
    increments = []
    for path in paths:
        path_increments = np.diff(path)
        increments.extend(path_increments)
    
    # 통계량 계산
    mean_increment = np.mean(increments)
    var_increment = np.var(increments)
    expected_var = dt  # 위너 과정의 증분 분산은 dt와 같아야 함
    
    # 정규성 테스트
    _, p_value = stats.normaltest(increments)
    
    # 결과 저장
    results = {
        'mean_increment': mean_increment,
        'var_increment': var_increment,
        'expected_var': expected_var,
        'normality_p_value': p_value
    }
    
    return results

def main():
    """
    테스트 실행 함수
    """
    # 파라미터 설정
    days = 1.0  # 1년
    dt = 1/252  # 일별 (1년 중 하루)
    num_paths = 5
    
    # 위너 과정 시뮬레이션
    print("위너 과정 (Wiener Process) 시뮬레이션 중...")
    paths = simulate_wiener_process(
        days=days,
        num_paths=num_paths,
        dt=dt,
        plot=True
    )
    
    # 결과 분석
    analysis = analyze_wiener_process(paths, dt)
    
    print("\n위너 과정 분석 결과:")
    print(f"  평균 증분: {analysis['mean_increment']:.6f} (기대값: 0)")
    print(f"  증분 분산: {analysis['var_increment']:.6f} (기대값: {analysis['expected_var']:.6f})")
    print(f"  정규성 테스트 p-value: {analysis['normality_p_value']:.6f}")
    
    # 시간에 따른 분산 변화 시각화
    time_points = np.linspace(0.1, days, 10)
    variances = []
    
    for t in time_points:
        # 가장 가까운 시간 인덱스 찾기
        idx = int(t / dt)
        values_at_t = [path[idx] for path in paths]
        variances.append(np.var(values_at_t))
    
    plt.figure(figsize=(10, 6))
    plt.plot(time_points, variances, 'bo-', label='시뮬레이션 분산')
    plt.plot(time_points, time_points, 'r--', label='이론적 분산 (t)')
    plt.title('위너 과정의 시간에 따른 분산 변화')
    plt.xlabel('시간 (t)')
    plt.ylabel('분산')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    
    # 증분 분포 시각화
    increments = np.concatenate([np.diff(path) for path in paths])
    
    plt.figure(figsize=(10, 6))
    plt.hist(increments, bins=30, alpha=0.7, density=True, label='증분 히스토그램')
    
    # 이론적 분포 (정규 분포)
    x = np.linspace(min(increments), max(increments), 100)
    plt.plot(x, stats.norm.pdf(x, 0, np.sqrt(dt)), 'r-', lw=2, label=f'정규 분포 N(0, {np.sqrt(dt):.6f})')
    
    plt.title('위너 과정 증분의 분포')
    plt.xlabel('증분')
    plt.ylabel('확률 밀도')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    
    # 위너 과정의 금융적 의미 설명
    print("\n위너 과정의 금융적 의미:")
    print("  위너 과정은 금융 모델링에서 불확실성을 나타내는 기본 요소입니다.")
    print("  이토 적분과 함께 사용되어 확률적 미분방정식의 기초가 됩니다.")
    print("  Black-Scholes 모델과 같은 많은 파생상품 가격 모델의 기반입니다.")

if __name__ == "__main__":
    main() 