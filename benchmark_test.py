# 데이터 분석 시간 측정 스크립트
# Python이 설치되어 있어야 합니다!

import time
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans

def measure_analysis_time(rows, columns, analysis_type):
    """
    실제 데이터 분석 시간을 측정하는 함수
    
    Parameters:
    - rows: 데이터 행 수
    - columns: 데이터 열 수
    - analysis_type: 세부 분석 방법 (예: 'reg_linear_simple', 'clf_forest')
    """
    print(f"\n{'='*60}")
    print(f"테스트 조건: {rows:,}행 × {columns}열, 분석: {analysis_type}")
    print(f"{'='*60}")
    
    # 랜덤 데이터 생성
    print("1️⃣ 데이터 생성 중...")
    X = np.random.randn(rows, columns)
    y = np.random.randn(rows)
    
    # 전체 시작 시간
    total_start = time.time()
    
    # 1단계: 데이터 로딩 (Pandas DataFrame 변환)
    print("2️⃣ 데이터 로딩 중...")
    loading_start = time.time()
    df = pd.DataFrame(X)
    loading_time = time.time() - loading_start
    
    # 2단계: 전처리
    print("3️⃣ 데이터 전처리 중...")
    preprocessing_start = time.time()
    # 간단한 전처리 시뮬레이션
    df_processed = df.copy()
    preprocessing_time = time.time() - preprocessing_start
    
    # 3단계: 분석 실행
    print(f"4️⃣ {analysis_type} 분석 실행 중...")
    analysis_start = time.time()
    
    try:
        # 단순 집계
        if analysis_type == 'agg_basic':
            result = df.mean()
            result = df.sum()
            result = df.count()
        
        elif analysis_type == 'agg_groupby':
            # 첫 번째 열로 그룹화
            result = df.groupby(df.columns[0] % 10).mean()
        
        elif analysis_type == 'agg_pivot':
            df_small = df.head(min(10000, rows))  # 피벗은 작은 데이터로
            result = df_small.pivot_table(values=df_small.columns[0], 
                                         index=df_small.columns[1] % 5,
                                         aggfunc='mean')
        
        # 회귀분석
        elif analysis_type == 'reg_linear_simple':
            model = LinearRegression()
            model.fit(X[:, 0:1], y)  # 단순 회귀 (1개 변수)
        
        elif analysis_type == 'reg_linear_multiple':
            model = LinearRegression()
            model.fit(X, y)  # 다중 회귀
        
        elif analysis_type == 'reg_ridge':
            from sklearn.linear_model import Ridge
            model = Ridge(alpha=1.0, max_iter=1000)
            model.fit(X, y)
        
        elif analysis_type == 'reg_polynomial':
            from sklearn.preprocessing import PolynomialFeatures
            poly = PolynomialFeatures(degree=2)
            X_poly = poly.fit_transform(X[:, :min(5, columns)])  # 5개 컬럼만
            model = LinearRegression()
            model.fit(X_poly, y)
        
        # 분류
        elif analysis_type == 'clf_logistic':
            from sklearn.linear_model import LogisticRegression
            y_class = (y > 0).astype(int)
            model = LogisticRegression(max_iter=100)
            model.fit(X, y_class)
        
        elif analysis_type == 'clf_tree':
            from sklearn.tree import DecisionTreeClassifier
            y_class = (y > 0).astype(int)
            model = DecisionTreeClassifier(max_depth=10)
            model.fit(X, y_class)
        
        elif analysis_type == 'clf_forest':
            y_class = (y > 0).astype(int)
            model = RandomForestClassifier(n_estimators=100, max_depth=10)
            model.fit(X, y_class)
        
        elif analysis_type == 'clf_svm':
            from sklearn.svm import SVC
            y_class = (y > 0).astype(int)
            # SVM은 데이터가 많으면 너무 느려서 샘플링
            sample_size = min(10000, rows)
            X_sample = X[:sample_size]
            y_sample = y_class[:sample_size]
            model = SVC(kernel='rbf')
            model.fit(X_sample, y_sample)
        
        # 클러스터링
        elif analysis_type == 'clu_kmeans_small':
            model = KMeans(n_clusters=5, max_iter=300)
            model.fit(X)
        
        elif analysis_type == 'clu_kmeans_large':
            model = KMeans(n_clusters=20, max_iter=300)
            model.fit(X)
        
        elif analysis_type == 'clu_dbscan':
            from sklearn.cluster import DBSCAN
            # DBSCAN은 데이터가 많으면 샘플링
            sample_size = min(50000, rows)
            model = DBSCAN(eps=0.5, min_samples=5)
            model.fit(X[:sample_size])
        
        elif analysis_type == 'clu_hierarchical':
            from sklearn.cluster import AgglomerativeClustering
            # 계층적은 매우 느려서 작은 샘플만
            sample_size = min(5000, rows)
            model = AgglomerativeClustering(n_clusters=5)
            model.fit(X[:sample_size])
        
        # 딥러닝 (간단한 시뮬레이션)
        elif analysis_type == 'dl_simple':
            from sklearn.neural_network import MLPClassifier
            y_class = (y > 0).astype(int)
            model = MLPClassifier(hidden_layer_sizes=(50, 25), max_iter=10)
            model.fit(X, y_class)
        
        elif analysis_type == 'dl_deep':
            from sklearn.neural_network import MLPClassifier
            y_class = (y > 0).astype(int)
            model = MLPClassifier(hidden_layer_sizes=(100, 50, 25, 10), max_iter=20)
            model.fit(X, y_class)
        
        else:
            print(f"❌ 지원하지 않는 분석 방법: {analysis_type}")
            return None
        
        analysis_time = time.time() - analysis_start
        
    except Exception as e:
        print(f"❌ 오류 발생: {e}")
        return None
    
    # 전체 시간
    total_time = time.time() - total_start
    
    # 결과 출력
    print(f"\n✅ 측정 완료!")
    print(f"\n📊 측정 결과:")
    print(f"  • 데이터 로딩:   {loading_time:.2f}초 ({loading_time/total_time*100:.1f}%)")
    print(f"  • 데이터 전처리: {preprocessing_time:.2f}초 ({preprocessing_time/total_time*100:.1f}%)")
    print(f"  • 분석 실행:     {analysis_time:.2f}초 ({analysis_time/total_time*100:.1f}%)")
    print(f"  • 전체 시간:     {total_time:.2f}초")
    
    return {
        'rows': rows,
        'columns': columns,
        'analysis_type': analysis_type,
        'loading_time': loading_time,
        'preprocessing_time': preprocessing_time,
        'analysis_time': analysis_time,
        'total_time': total_time
    }

def compare_with_prediction(actual_time, predicted_time):
    """
    실제 시간과 예측 시간을 비교
    """
    error = abs(actual_time - predicted_time)
    error_percent = (error / actual_time) * 100
    
    print(f"\n🎯 정확도 분석:")
    print(f"  • 실제 시간:   {actual_time:.2f}초")
    print(f"  • 예측 시간:   {predicted_time:.2f}초")
    print(f"  • 오차:        {error:.2f}초")
    print(f"  • 오차율:      {error_percent:.1f}%")
    
    if error_percent <= 30:
        print(f"  • 평가:        ✅ 매우 정확! (목표: ±30% 이내)")
    elif error_percent <= 50:
        print(f"  • 평가:        ⚠️ 양호 (±50% 이내)")
    else:
        print(f"  • 평가:        ❌ 개선 필요 (±50% 초과)")
    
    return error_percent

# 실행 예제
if __name__ == "__main__":
    print("=" * 60)
    print("데이터 분석 시간 측정 도구")
    print("=" * 60)
    
    # 테스트 케이스들 (세부 방법)
    test_cases = [
        {'rows': 10000, 'columns': 10, 'analysis': 'agg_basic', 'name': '기본 집계'},
        {'rows': 100000, 'columns': 20, 'analysis': 'reg_linear_multiple', 'name': '다중 선형 회귀'},
        {'rows': 50000, 'columns': 15, 'analysis': 'clf_tree', 'name': '의사결정나무'},
        {'rows': 100000, 'columns': 20, 'analysis': 'clf_forest', 'name': '랜덤 포레스트'},
        {'rows': 50000, 'columns': 10, 'analysis': 'clu_kmeans_small', 'name': 'K-means (k=5)'},
    ]
    
    print("\n💡 테스트할 조건을 선택하세요:")
    for i, case in enumerate(test_cases, 1):
        print(f"{i}. {case['name']} ({case['rows']:,}행 × {case['columns']}열)")
    print(f"{len(test_cases)+1}. 사용자 정의")
    
    choice = input(f"\n선택 (1-{len(test_cases)+1}): ").strip()
    
    if choice == str(len(test_cases)+1):
        # 사용자 정의 입력
        rows = int(input("데이터 행 수: "))
        columns = int(input("데이터 열 수: "))
        print("\n분석 방법 예시:")
        print("  집계: agg_basic, agg_groupby, agg_pivot")
        print("  회귀: reg_linear_simple, reg_linear_multiple, reg_ridge, reg_polynomial")
        print("  분류: clf_logistic, clf_tree, clf_forest, clf_svm")
        print("  클러스터링: clu_kmeans_small, clu_kmeans_large, clu_dbscan")
        print("  딥러닝: dl_simple, dl_deep")
        analysis = input("\n분석 방법 선택: ").strip()
    elif choice.isdigit() and 1 <= int(choice) <= len(test_cases):
        test_case = test_cases[int(choice) - 1]
        rows = test_case['rows']
        columns = test_case['columns']
        analysis = test_case['analysis']
    else:
        print("❌ 잘못된 선택입니다. 기본값으로 실행합니다.")
        rows, columns, analysis = 10000, 10, 'agg_basic'
    
    # 실제 시간 측정
    result = measure_analysis_time(rows, columns, analysis)
    
    if result:
        print(f"\n💾 결과가 저장되었습니다!")
        print(f"\n📝 이제 웹페이지(index.html)에서 같은 조건으로 예측해보세요:")
        print(f"   - 데이터 행 수: {rows:,}")
        print(f"   - 데이터 열 수: {columns}")
        print(f"   - 분석 방법: {analysis}")
        print(f"   - 사용 툴: Python")
        
        print(f"\n그런 다음 예측된 시간을 입력하세요:")
        predicted = float(input("예측 시간 (초): "))
        
        compare_with_prediction(result['total_time'], predicted)
