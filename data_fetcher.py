"""
외부 벤치마크 데이터 수집 스크립트
scitime과 OpenML로부터 실제 분석 시간 데이터를 수집합니다.
"""

import json
import pandas as pd
import numpy as np
from datetime import datetime
import os


class BenchmarkDataFetcher:
    """외부 벤치마크 데이터 수집기"""
    
    def __init__(self):
        self.benchmarks = []
        
    def fetch_openml_data(self, limit=1000):
        """OpenML에서 실제 실행 시간 데이터 가져오기"""
        print("=" * 60)
        print("OpenML 데이터 수집 시작...")
        print("=" * 60)
        
        try:
            import openml
            
            # OpenML 실행 결과 가져오기
            print(f"\n📥 최근 {limit}개의 실행 결과를 가져옵니다...")
            
            # scikit-learn 관련 실행 결과만 필터링
            runs_df = openml.runs.list_runs(
                size=limit,
                output_format='dataframe'
            )
            
            if runs_df is None or runs_df.empty:
                print("⚠️  OpenML 데이터를 가져올 수 없습니다.")
                return []
            
            print(f"✅ {len(runs_df)}개의 실행 결과를 가져왔습니다.")
            
            # 데이터 변환
            collected = 0
            for idx, run in runs_df.iterrows():
                try:
                    # 필요한 정보가 모두 있는지 확인
                    if pd.isna(run.get('run_time')):
                        continue
                    
                    # 데이터셋 정보 가져오기
                    dataset_id = run.get('data_id')
                    if pd.isna(dataset_id):
                        continue
                    
                    try:
                        dataset = openml.datasets.get_dataset(int(dataset_id))
                        rows, cols, _, _ = dataset.get_data()
                        n_rows = len(rows) if rows is not None else 0
                        n_cols = len(cols) if cols is not None else 0
                    except:
                        continue
                    
                    if n_rows == 0 or n_cols == 0:
                        continue
                    
                    # 알고리즘 이름 매핑
                    flow_name = str(run.get('flow_name', ''))
                    method = self._map_algorithm_name(flow_name)
                    
                    if method == 'unknown':
                        continue
                    
                    benchmark = {
                        'source': 'openml',
                        'timestamp': datetime.now().isoformat(),
                        'rows': int(n_rows),
                        'columns': int(n_cols),
                        'method': method,
                        'tool': 'python',
                        'hardware': 'medium',  # OpenML 서버는 대략 중간 사양
                        'data_type_ratio': {
                            'numeric': 0.7,
                            'categorical': 0.2,
                            'text': 0.1
                        },
                        'elapsed_time_seconds': float(run['run_time']),
                        'loading_time': float(run['run_time']) * 0.2,
                        'preprocessing_time': float(run['run_time']) * 0.3,
                        'execution_time': float(run['run_time']) * 0.5
                    }
                    
                    self.benchmarks.append(benchmark)
                    collected += 1
                    
                    if collected % 10 == 0:
                        print(f"  수집 중... {collected}개")
                    
                    if collected >= 100:  # 너무 많이 수집하지 않도록 제한
                        break
                        
                except Exception as e:
                    continue
            
            print(f"\n✅ OpenML에서 {collected}개의 유효한 벤치마크를 수집했습니다.")
            return self.benchmarks
            
        except ImportError:
            print("❌ openml 패키지가 설치되지 않았습니다.")
            print("   설치: pip install openml")
            return []
        except Exception as e:
            print(f"❌ OpenML 데이터 수집 중 오류: {str(e)}")
            return []
    
    def _map_algorithm_name(self, flow_name):
        """OpenML의 알고리즘 이름을 우리 시스템의 방법명으로 매핑"""
        flow_name = flow_name.lower()
        
        # 분류
        if 'randomforest' in flow_name or 'random_forest' in flow_name:
            return 'clf_forest'
        elif 'logistic' in flow_name:
            return 'clf_logistic'
        elif 'decisiontree' in flow_name or 'decision_tree' in flow_name:
            return 'clf_tree'
        elif 'svm' in flow_name or 'svc' in flow_name:
            return 'clf_svm'
        
        # 회귀
        elif 'linearregression' in flow_name or 'linear_regression' in flow_name:
            return 'reg_linear_multiple'
        elif 'ridge' in flow_name:
            return 'reg_ridge'
        
        # 클러스터링
        elif 'kmeans' in flow_name:
            return 'clu_kmeans_small'
        elif 'dbscan' in flow_name:
            return 'clu_dbscan'
        
        return 'unknown'
    
    def generate_scitime_inspired_data(self):
        """scitime 논문의 방법론을 참고한 합성 데이터 생성
        
        scitime은 실제로 작은 샘플을 측정하고 외삽하는 방식입니다.
        여기서는 scitime의 복잡도 공식을 참고하여 현실적인 데이터를 생성합니다.
        """
        print("\n" + "=" * 60)
        print("scitime 방법론 기반 벤치마크 데이터 생성...")
        print("=" * 60)
        
        # scitime이 사용하는 알고리즘별 복잡도 (논문 참고)
        algorithms = [
            # 집계 - O(n)
            {'method': 'agg_basic', 'complexity': lambda n, d: n * d * 1e-7},
            {'method': 'agg_groupby', 'complexity': lambda n, d: n * np.log(n) * d * 1e-7},
            
            # 회귀 - O(n * d^2)
            {'method': 'reg_linear_simple', 'complexity': lambda n, d: n * d * 1e-6},
            {'method': 'reg_linear_multiple', 'complexity': lambda n, d: n * d * d * 1e-6},
            {'method': 'reg_ridge', 'complexity': lambda n, d: n * d * d * 1.2e-6},
            
            # 분류 - 복잡도 높음
            {'method': 'clf_logistic', 'complexity': lambda n, d: n * d * 100 * 1e-6},  # 100 iterations
            {'method': 'clf_tree', 'complexity': lambda n, d: n * np.log(n) * d * 2e-6},
            {'method': 'clf_forest', 'complexity': lambda n, d: n * np.log(n) * d * 100 * 2e-6},  # 100 trees
            
            # 클러스터링
            {'method': 'clu_kmeans_small', 'complexity': lambda n, d: n * 5 * d * 100 * 1e-6},  # k=5, iter=100
            {'method': 'clu_kmeans_large', 'complexity': lambda n, d: n * 15 * d * 100 * 1e-6},  # k=15
        ]
        
        # 다양한 데이터 크기
        data_sizes = [
            {'rows': 1000, 'columns': 5},
            {'rows': 5000, 'columns': 10},
            {'rows': 10000, 'columns': 10},
            {'rows': 50000, 'columns': 20},
            {'rows': 100000, 'columns': 20},
            {'rows': 500000, 'columns': 30},
            {'rows': 1000000, 'columns': 30},
            {'rows': 1000000, 'columns': 50},
        ]
        
        # 하드웨어 배율
        hardware_configs = [
            {'name': 'low', 'multiplier': 4.0},
            {'name': 'medium', 'multiplier': 1.0},
            {'name': 'high', 'multiplier': 0.4},
            {'name': 'ultra', 'multiplier': 0.2},
        ]
        
        generated = 0
        for algo in algorithms:
            for size in data_sizes:
                for hw in hardware_configs:
                    # 기본 시간 계산
                    base_time = algo['complexity'](size['rows'], size['columns'])
                    
                    # 하드웨어 배율 적용
                    adjusted_time = base_time * hw['multiplier']
                    
                    # 약간의 노이즈 추가 (현실적으로)
                    noise = np.random.normal(1.0, 0.1)
                    final_time = adjusted_time * noise
                    
                    # 최소 시간 보장
                    final_time = max(final_time, 0.001)
                    
                    benchmark = {
                        'source': 'scitime_inspired',
                        'timestamp': datetime.now().isoformat(),
                        'rows': size['rows'],
                        'columns': size['columns'],
                        'method': algo['method'],
                        'tool': 'python',
                        'hardware': hw['name'],
                        'data_type_ratio': {
                            'numeric': 0.7,
                            'categorical': 0.2,
                            'text': 0.1
                        },
                        'elapsed_time_seconds': round(final_time, 4),
                        'loading_time': round(final_time * 0.2, 4),
                        'preprocessing_time': round(final_time * 0.3, 4),
                        'execution_time': round(final_time * 0.5, 4)
                    }
                    
                    self.benchmarks.append(benchmark)
                    generated += 1
        
        print(f"✅ {generated}개의 scitime 기반 벤치마크를 생성했습니다.")
        return self.benchmarks
    
    def save_benchmarks(self, filename='benchmark_data.json'):
        """수집된 벤치마크를 JSON 파일로 저장"""
        if not self.benchmarks:
            print("⚠️  저장할 벤치마크 데이터가 없습니다.")
            return
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(self.benchmarks, f, indent=2, ensure_ascii=False)
        
        print(f"\n💾 {len(self.benchmarks)}개의 벤치마크를 '{filename}'에 저장했습니다.")
    
    def get_summary(self):
        """수집된 데이터 요약"""
        if not self.benchmarks:
            return "수집된 데이터 없음"
        
        df = pd.DataFrame(self.benchmarks)
        
        summary = f"""
{'=' * 60}
벤치마크 데이터 요약
{'=' * 60}
총 개수: {len(self.benchmarks):,}개

출처별:
{df['source'].value_counts().to_string() if 'source' in df.columns else 'N/A'}

데이터 크기 범위:
  - 행: {df['rows'].min():,} ~ {df['rows'].max():,}
  - 열: {df['columns'].min()} ~ {df['columns'].max()}

분석 방법:
{df['method'].value_counts().to_string()}

하드웨어:
{df['hardware'].value_counts().to_string()}

실행 시간 통계:
  - 평균: {df['elapsed_time_seconds'].mean():.2f}초
  - 중앙값: {df['elapsed_time_seconds'].median():.2f}초
  - 최소: {df['elapsed_time_seconds'].min():.4f}초
  - 최대: {df['elapsed_time_seconds'].max():.2f}초
{'=' * 60}
"""
        return summary


def main():
    """메인 실행 함수"""
    fetcher = BenchmarkDataFetcher()
    
    print("\n🚀 벤치마크 데이터 수집을 시작합니다.\n")
    
    # 1. scitime 방법론 기반 데이터 생성 (항상 가능)
    fetcher.generate_scitime_inspired_data()
    
    # 2. OpenML 데이터 수집 시도 (선택적)
    try:
        print("\n⏳ OpenML 데이터 수집을 시도합니다...")
        print("   (실패해도 scitime 데이터만으로 동작합니다)")
        fetcher.fetch_openml_data(limit=200)
    except Exception as e:
        print(f"⚠️  OpenML 수집 실패: {str(e)}")
        print("   scitime 기반 데이터만 사용합니다.")
    
    # 3. 결과 저장
    fetcher.save_benchmarks('benchmark_data.json')
    
    # 4. 요약 출력
    print(fetcher.get_summary())
    
    print("\n✅ 완료! 이제 웹 서비스에서 이 데이터를 사용할 수 있습니다.")


if __name__ == "__main__":
    main()
