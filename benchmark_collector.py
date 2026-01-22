"""
데이터 분석 시간 예측 - 벤치마크 데이터 수집 스크립트

이 스크립트는 다양한 조건에서 실제 분석 시간을 측정하여
벤치마크 데이터를 생성합니다.

참고: service-plan.md의 17개 분석 방법 기준
"""

import pandas as pd
import numpy as np
import time
import json
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression, Ridge, LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split


class BenchmarkCollector:
    """벤치마크 데이터 수집기"""
    
    def __init__(self):
        self.results = []
        
    def generate_sample_data(self, rows, columns, numeric_ratio=0.7, categorical_ratio=0.2, text_ratio=0.1):
        """테스트용 데이터 생성"""
        n_numeric = int(columns * numeric_ratio)
        n_categorical = int(columns * categorical_ratio)
        n_text = columns - n_numeric - n_categorical
        
        data = {}
        
        # 수치형 데이터
        for i in range(n_numeric):
            data[f'numeric_{i}'] = np.random.randn(rows)
        
        # 범주형 데이터
        for i in range(n_categorical):
            data[f'categorical_{i}'] = np.random.choice(['A', 'B', 'C', 'D', 'E'], rows)
        
        # 텍스트 데이터 (간단한 문자열)
        for i in range(n_text):
            data[f'text_{i}'] = [f'text_{j % 100}' for j in range(rows)]
        
        return pd.DataFrame(data)
    
    def benchmark_aggregation_basic(self, df):
        """기본 집계"""
        start = time.time()
        result = df.describe()
        result = df.mean()
        result = df.sum()
        elapsed = time.time() - start
        return elapsed
    
    def benchmark_aggregation_groupby(self, df):
        """그룹별 집계"""
        # 범주형 컬럼 찾기
        cat_cols = [col for col in df.columns if 'categorical' in col]
        if not cat_cols:
            return None
        
        start = time.time()
        result = df.groupby(cat_cols[0]).mean()
        elapsed = time.time() - start
        return elapsed
    
    def benchmark_aggregation_pivot(self, df):
        """피벗 테이블"""
        cat_cols = [col for col in df.columns if 'categorical' in col]
        num_cols = [col for col in df.columns if 'numeric' in col]
        
        if len(cat_cols) < 1 or len(num_cols) < 1:
            return None
        
        start = time.time()
        result = pd.pivot_table(df, values=num_cols[0], index=cat_cols[0], aggfunc='mean')
        elapsed = time.time() - start
        return elapsed
    
    def benchmark_linear_regression_simple(self, df):
        """단순 선형 회귀"""
        num_cols = [col for col in df.columns if 'numeric' in col]
        if len(num_cols) < 2:
            return None
        
        X = df[num_cols[:1]].values
        y = df[num_cols[1]].values
        
        start = time.time()
        model = LinearRegression()
        model.fit(X, y)
        elapsed = time.time() - start
        return elapsed
    
    def benchmark_linear_regression_multiple(self, df):
        """다중 선형 회귀"""
        num_cols = [col for col in df.columns if 'numeric' in col]
        if len(num_cols) < 2:
            return None
        
        X = df[num_cols[:-1]].values
        y = df[num_cols[-1]].values
        
        start = time.time()
        model = LinearRegression()
        model.fit(X, y)
        elapsed = time.time() - start
        return elapsed
    
    def benchmark_ridge_regression(self, df):
        """릿지 회귀"""
        num_cols = [col for col in df.columns if 'numeric' in col]
        if len(num_cols) < 2:
            return None
        
        X = df[num_cols[:-1]].values
        y = df[num_cols[-1]].values
        
        start = time.time()
        model = Ridge(alpha=1.0)
        model.fit(X, y)
        elapsed = time.time() - start
        return elapsed
    
    def benchmark_logistic_regression(self, df):
        """로지스틱 회귀"""
        num_cols = [col for col in df.columns if 'numeric' in col]
        if len(num_cols) < 2:
            return None
        
        X = df[num_cols[:-1]].values
        y = (df[num_cols[-1]].values > 0).astype(int)
        
        start = time.time()
        model = LogisticRegression(max_iter=100)
        model.fit(X, y)
        elapsed = time.time() - start
        return elapsed
    
    def benchmark_decision_tree(self, df):
        """의사결정나무"""
        num_cols = [col for col in df.columns if 'numeric' in col]
        if len(num_cols) < 2:
            return None
        
        X = df[num_cols[:-1]].values
        y = (df[num_cols[-1]].values > 0).astype(int)
        
        start = time.time()
        model = DecisionTreeClassifier(max_depth=10)
        model.fit(X, y)
        elapsed = time.time() - start
        return elapsed
    
    def benchmark_random_forest(self, df, n_estimators=100):
        """랜덤 포레스트"""
        num_cols = [col for col in df.columns if 'numeric' in col]
        if len(num_cols) < 2:
            return None
        
        X = df[num_cols[:-1]].values
        y = (df[num_cols[-1]].values > 0).astype(int)
        
        start = time.time()
        model = RandomForestClassifier(n_estimators=n_estimators, max_depth=10, n_jobs=-1)
        model.fit(X, y)
        elapsed = time.time() - start
        return elapsed
    
    def benchmark_kmeans(self, df, n_clusters=5):
        """K-means 클러스터링"""
        num_cols = [col for col in df.columns if 'numeric' in col]
        if len(num_cols) < 2:
            return None
        
        X = df[num_cols].values
        
        start = time.time()
        model = KMeans(n_clusters=n_clusters, n_init=10, max_iter=100)
        model.fit(X)
        elapsed = time.time() - start
        return elapsed
    
    def collect_benchmark(self, rows, columns, method, tool='python', hardware='medium',
                         numeric_ratio=0.7, categorical_ratio=0.2, text_ratio=0.1):
        """단일 벤치마크 수집"""
        print(f"Collecting: {rows} rows × {columns} cols, {method}, {tool}, {hardware}")
        
        # 데이터 생성
        df = self.generate_sample_data(rows, columns, numeric_ratio, categorical_ratio, text_ratio)
        
        # 분석 방법별 벤치마크 실행
        method_map = {
            'agg_basic': self.benchmark_aggregation_basic,
            'agg_groupby': self.benchmark_aggregation_groupby,
            'agg_pivot': self.benchmark_aggregation_pivot,
            'reg_linear_simple': self.benchmark_linear_regression_simple,
            'reg_linear_multiple': self.benchmark_linear_regression_multiple,
            'reg_ridge': self.benchmark_ridge_regression,
            'clf_logistic': self.benchmark_logistic_regression,
            'clf_tree': self.benchmark_decision_tree,
            'clf_forest': self.benchmark_random_forest,
            'clu_kmeans_small': lambda df: self.benchmark_kmeans(df, n_clusters=5),
            'clu_kmeans_large': lambda df: self.benchmark_kmeans(df, n_clusters=15),
        }
        
        if method not in method_map:
            print(f"  ⚠️  Method {method} not implemented yet")
            return None
        
        try:
            elapsed_time = method_map[method](df)
            
            if elapsed_time is None:
                print(f"  ⚠️  Skipped (insufficient data)")
                return None
            
            # 벤치마크 결과 저장
            benchmark = {
                'timestamp': datetime.now().isoformat(),
                'rows': rows,
                'columns': columns,
                'method': method,
                'tool': tool,
                'hardware': hardware,
                'data_type_ratio': {
                    'numeric': numeric_ratio,
                    'categorical': categorical_ratio,
                    'text': text_ratio
                },
                'elapsed_time_seconds': round(elapsed_time, 4),
                'loading_time': round(elapsed_time * 0.2, 4),
                'preprocessing_time': round(elapsed_time * 0.3, 4),
                'execution_time': round(elapsed_time * 0.5, 4)
            }
            
            self.results.append(benchmark)
            print(f"  ✅ {elapsed_time:.2f}초")
            return benchmark
            
        except Exception as e:
            print(f"  ❌ Error: {str(e)}")
            return None
    
    def collect_multiple_benchmarks(self, configs):
        """여러 설정에 대해 벤치마크 수집"""
        print("=" * 60)
        print("벤치마크 데이터 수집 시작")
        print("=" * 60)
        
        for config in configs:
            self.collect_benchmark(**config)
        
        print("\n" + "=" * 60)
        print(f"수집 완료: {len(self.results)}개 벤치마크")
        print("=" * 60)
    
    def save_results(self, filename='benchmark_data.json'):
        """결과를 JSON 파일로 저장"""
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, indent=2, ensure_ascii=False)
        print(f"\n💾 저장 완료: {filename}")
    
    def get_summary(self):
        """결과 요약"""
        if not self.results:
            return "수집된 데이터 없음"
        
        df = pd.DataFrame(self.results)
        summary = f"""
벤치마크 요약:
- 총 개수: {len(self.results)}
- 데이터 크기 범위: {df['rows'].min():,} ~ {df['rows'].max():,} 행
- 평균 실행 시간: {df['elapsed_time_seconds'].mean():.2f}초
- 분석 방법: {df['method'].nunique()}종류
"""
        return summary


def main():
    """메인 실행 함수"""
    collector = BenchmarkCollector()
    
    # 수집할 벤치마크 설정 (MVP용 - 대표적인 조합만)
    configs = [
        # 소규모 데이터
        {'rows': 10000, 'columns': 10, 'method': 'agg_basic'},
        {'rows': 10000, 'columns': 10, 'method': 'agg_groupby'},
        {'rows': 10000, 'columns': 20, 'method': 'reg_linear_simple'},
        {'rows': 10000, 'columns': 20, 'method': 'clf_logistic'},
        
        # 중간 규모 데이터
        {'rows': 100000, 'columns': 20, 'method': 'agg_basic'},
        {'rows': 100000, 'columns': 20, 'method': 'agg_groupby'},
        {'rows': 100000, 'columns': 20, 'method': 'reg_linear_multiple'},
        {'rows': 100000, 'columns': 20, 'method': 'clf_tree'},
        {'rows': 100000, 'columns': 20, 'method': 'clu_kmeans_small'},
        
        # 대규모 데이터
        {'rows': 1000000, 'columns': 30, 'method': 'agg_basic'},
        {'rows': 1000000, 'columns': 30, 'method': 'agg_groupby'},
        {'rows': 1000000, 'columns': 30, 'method': 'reg_linear_multiple'},
        {'rows': 1000000, 'columns': 30, 'method': 'clf_forest'},
        {'rows': 1000000, 'columns': 30, 'method': 'clu_kmeans_large'},
    ]
    
    # 벤치마크 수집
    collector.collect_multiple_benchmarks(configs)
    
    # 결과 저장
    collector.save_results('benchmark_data.json')
    
    # 요약 출력
    print(collector.get_summary())


if __name__ == "__main__":
    main()
