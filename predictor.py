"""
분석 시간 예측 엔진
벤치마크 데이터를 기반으로 사용자 입력에 대한 예측을 수행합니다.
ML 모델과 앙상블 예측을 지원합니다.
"""

import json
import numpy as np
from typing import Dict, List, Any


class TimePredictor:
    """분석 시간 예측기 (벤치마크 + ML 앙상블)"""
    
    def __init__(self, benchmark_file='benchmark_data.json', use_ml=True):
        """
        Args:
            benchmark_file: 벤치마크 데이터 JSON 파일 경로
            use_ml: ML 모델 사용 여부 (기본값 True)
        """
        self.benchmarks = []
        self.load_benchmarks(benchmark_file)
        self.use_ml = use_ml
        self.ml_predictor = None
        
        # ML 모델 로드 시도
        if use_ml:
            try:
                from ml_predictor import MLTimePredictor
                self.ml_predictor = MLTimePredictor()
                
                # 저장된 모델 로드 시도
                if not self.ml_predictor.load_model('ml_model.pkl'):
                    # 없으면 학습
                    print("💡 ML 모델을 학습합니다...")
                    if self.ml_predictor.train(benchmark_file):
                        self.ml_predictor.save_model('ml_model.pkl')
            except Exception as e:
                print(f"⚠️  ML 모델 로드 실패: {str(e)}")
                print("   벤치마크 기반 예측만 사용합니다.")
                self.ml_predictor = None
    
    def load_benchmarks(self, filename):
        """벤치마크 데이터 로드"""
        try:
            with open(filename, 'r', encoding='utf-8') as f:
                self.benchmarks = json.load(f)
            print(f"✅ {len(self.benchmarks)}개의 벤치마크 데이터를 로드했습니다.")
        except FileNotFoundError:
            print(f"⚠️  '{filename}' 파일을 찾을 수 없습니다.")
            print("   data_fetcher.py를 먼저 실행하세요.")
            self.benchmarks = []
        except Exception as e:
            print(f"❌ 벤치마크 로드 실패: {str(e)}")
            self.benchmarks = []
    
    def predict(self, user_input: Dict[str, Any], use_ensemble=True) -> Dict[str, Any]:
        """
        사용자 입력에 대한 분석 시간 예측 (앙상블 지원)
        
        Args:
            user_input: {
                'rows': int,
                'columns': int,
                'method': str,
                'tool': str,
                'hardware': str,
                'data_type_ratio': dict (optional)
            }
            use_ensemble: 앙상블 예측 사용 여부 (기본값 True)
        
        Returns:
            {
                'estimated_time_minutes': float,
                'confidence_interval': {'min': float, 'max': float},
                'confidence_level': str,
                'breakdown': {...},
                'similar_cases_count': int,
                'data_source': str
            }
        """
        # ML 모델이 있고 앙상블을 사용하는 경우
        if use_ensemble and self.ml_predictor and self.ml_predictor.is_trained:
            return self._ensemble_predict(user_input)
        
        # 일반 벤치마크 기반 예측
        if not self.benchmarks:
            return self._fallback_prediction(user_input)
        
        # 1. 유사한 케이스 찾기
        similar_cases = self._find_similar_cases(user_input)
        
        if len(similar_cases) < 3:
            # 유사 케이스가 너무 적으면 더 넓게 검색
            similar_cases = self._find_similar_cases(user_input, tolerance=2.0)
        
        if len(similar_cases) < 1:
            # 그래도 없으면 폴백
            return self._fallback_prediction(user_input)
        
        # 2. 예측 계산
        times = [case['elapsed_time_seconds'] for case in similar_cases]
        median_time = np.median(times)
        
        # 3. 신뢰 구간 계산
        percentile_25 = np.percentile(times, 25)
        percentile_75 = np.percentile(times, 75)
        
        # 최소/최대 범위 (보수적으로)
        min_time = max(percentile_25 * 0.7, min(times) * 0.9)
        max_time = min(percentile_75 * 1.3, max(times) * 1.1)
        
        # 4. 신뢰도 계산
        confidence_level = self._calculate_confidence(similar_cases, user_input)
        
        # 5. 단계별 분해
        breakdown = {
            'loading_minutes': round(median_time * 0.2 / 60, 2),
            'preprocessing_minutes': round(median_time * 0.3 / 60, 2),
            'execution_minutes': round(median_time * 0.5 / 60, 2)
        }
        
        return {
            'estimated_time_minutes': round(median_time / 60, 2),
            'confidence_interval': {
                'min_minutes': round(min_time / 60, 2),
                'max_minutes': round(max_time / 60, 2)
            },
            'confidence_level': confidence_level,
            'breakdown': breakdown,
            'similar_cases_count': len(similar_cases),
            'data_source': 'benchmark' if similar_cases else 'fallback'
        }
    
    def _find_similar_cases(self, user_input: Dict[str, Any], tolerance=1.5) -> List[Dict]:
        """유사한 벤치마크 케이스 찾기"""
        similar = []
        
        target_rows = user_input['rows']
        target_cols = user_input['columns']
        target_method = user_input['method']
        target_hardware = user_input['hardware']
        
        for benchmark in self.benchmarks:
            # 1. 분석 방법이 같아야 함
            if benchmark['method'] != target_method:
                continue
            
            # 2. 하드웨어가 같아야 함
            if benchmark['hardware'] != target_hardware:
                continue
            
            # 3. 데이터 크기가 유사해야 함 (tolerance 배 이내)
            row_ratio = benchmark['rows'] / target_rows
            col_ratio = benchmark['columns'] / target_cols
            
            if (1/tolerance <= row_ratio <= tolerance and 
                1/tolerance <= col_ratio <= tolerance):
                
                # 유사도 점수 계산 (크기가 비슷할수록 높음)
                similarity = 1 / (abs(np.log(row_ratio)) + abs(np.log(col_ratio)) + 1)
                benchmark['similarity'] = similarity
                similar.append(benchmark)
        
        # 유사도 순으로 정렬
        similar.sort(key=lambda x: x['similarity'], reverse=True)
        
        # 상위 10개만 사용
        return similar[:10]
    
    def _ensemble_predict(self, user_input: Dict[str, Any]) -> Dict[str, Any]:
        """앙상블 예측: 벤치마크 + ML + 복잡도 기반"""
        predictions = []
        weights = []
        
        # 1. 벤치마크 기반 예측
        benchmark_result = None
        similar_cases = self._find_similar_cases(user_input)
        if len(similar_cases) < 3:
            similar_cases = self._find_similar_cases(user_input, tolerance=2.0)
        
        if len(similar_cases) >= 1:
            times = [case['elapsed_time_seconds'] for case in similar_cases]
            benchmark_time = np.median(times)
            predictions.append(benchmark_time)
            # 유사 케이스가 많을수록 가중치 높음
            weight = min(len(similar_cases) / 10.0, 1.0)
            weights.append(weight * 0.5)  # 최대 50% 가중치
            benchmark_result = {
                'time': benchmark_time,
                'count': len(similar_cases),
                'confidence': self._calculate_confidence(similar_cases, user_input)
            }
        
        # 2. ML 모델 예측
        ml_result = None
        try:
            ml_prediction = self.ml_predictor.predict(user_input)
            ml_time = ml_prediction['estimated_time_minutes'] * 60  # 초로 변환
            predictions.append(ml_time)
            weights.append(0.4)  # 40% 가중치
            ml_result = ml_prediction
        except Exception as e:
            print(f"⚠️  ML 예측 실패: {str(e)}")
        
        # 3. 복잡도 기반 예측
        fallback_result = self._fallback_prediction(user_input)
        fallback_time = fallback_result['estimated_time_minutes'] * 60
        predictions.append(fallback_time)
        weights.append(0.1)  # 10% 가중치
        
        # 가중 평균 계산
        if len(predictions) == 0:
            return fallback_result
        
        # 가중치 정규화
        total_weight = sum(weights)
        normalized_weights = [w / total_weight for w in weights]
        
        # 최종 예측
        final_time = sum(p * w for p, w in zip(predictions, normalized_weights))
        
        # 신뢰 구간 계산 (예측값들의 분산 기반)
        if len(predictions) > 1:
            std_dev = np.std(predictions)
            min_time = max(final_time - 1.5 * std_dev, min(predictions) * 0.7)
            max_time = min(final_time + 1.5 * std_dev, max(predictions) * 1.3)
        else:
            min_time = final_time * 0.7
            max_time = final_time * 1.3
        
        # 신뢰도 결정
        if benchmark_result and benchmark_result['count'] >= 5:
            confidence_level = benchmark_result['confidence']
            confidence_percent = 20 if confidence_level == 'High' else 25 if confidence_level == 'Medium' else 35
        elif ml_result:
            confidence_level = ml_result.get('confidence_level', 'Medium')
            confidence_percent = ml_result.get('confidence_percent', 25)
        else:
            confidence_level = 'Medium'
            confidence_percent = 30
        
        # 단계별 분해
        breakdown = {
            'loading_minutes': round(final_time * 0.2 / 60, 2),
            'preprocessing_minutes': round(final_time * 0.3 / 60, 2),
            'execution_minutes': round(final_time * 0.5 / 60, 2)
        }
        
        return {
            'estimated_time_minutes': round(final_time / 60, 2),
            'confidence_interval': {
                'min_minutes': round(min_time / 60, 2),
                'max_minutes': round(max_time / 60, 2)
            },
            'confidence_level': confidence_level,
            'breakdown': breakdown,
            'similar_cases_count': benchmark_result['count'] if benchmark_result else 0,
            'data_source': 'ensemble',
            'ensemble_details': {
                'benchmark_used': benchmark_result is not None,
                'ml_used': ml_result is not None,
                'weights': {
                    'benchmark': normalized_weights[0] if len(normalized_weights) > 0 else 0,
                    'ml': normalized_weights[1] if len(normalized_weights) > 1 else 0,
                    'complexity': normalized_weights[-1]
                }
            }
        }
    
    def _calculate_confidence(self, similar_cases: List[Dict], user_input: Dict) -> str:
        """예측 신뢰도 계산"""
        if len(similar_cases) == 0:
            return "Low"
        
        # 케이스 수가 많을수록 신뢰도 높음
        if len(similar_cases) >= 5:
            # 데이터 변동성 확인
            times = [case['elapsed_time_seconds'] for case in similar_cases]
            std_dev = np.std(times)
            mean_time = np.mean(times)
            cv = std_dev / mean_time if mean_time > 0 else 1.0  # 변동계수
            
            if cv < 0.3:
                return "High"
            elif cv < 0.5:
                return "Medium"
            else:
                return "Low"
        elif len(similar_cases) >= 3:
            return "Medium"
        else:
            return "Low"
    
    def _fallback_prediction(self, user_input: Dict[str, Any]) -> Dict[str, Any]:
        """벤치마크 데이터가 없을 때 복잡도 기반 예측"""
        rows = user_input['rows']
        cols = user_input['columns']
        method = user_input['method']
        hardware = user_input['hardware']
        
        # 하드웨어 배율
        hw_multiplier = {
            'low': 4.0,
            'medium': 1.0,
            'high': 0.4,
            'ultra': 0.2
        }.get(hardware, 1.0)
        
        # 알고리즘별 복잡도 기반 시간 추정 (초)
        if method.startswith('agg_'):
            base_time = rows * cols * 1e-7
        elif method.startswith('reg_'):
            base_time = rows * cols * cols * 1e-6
        elif method == 'clf_logistic':
            base_time = rows * cols * 100 * 1e-6
        elif method == 'clf_tree':
            base_time = rows * np.log(rows) * cols * 2e-6
        elif method == 'clf_forest':
            base_time = rows * np.log(rows) * cols * 100 * 2e-6
        elif method == 'clf_svm':
            base_time = rows * rows * cols * 1e-8
        elif method.startswith('clu_kmeans'):
            k = 5 if 'small' in method else 15
            base_time = rows * k * cols * 100 * 1e-6
        elif method == 'clu_dbscan':
            base_time = rows * rows * 1e-7
        elif method == 'clu_hierarchical':
            base_time = rows * rows * 1e-7
        elif method.startswith('dl_'):
            base_time = rows * cols * 1000 * 1e-5
        else:
            base_time = rows * cols * 1e-6
        
        # 하드웨어 보정
        estimated_time = base_time * hw_multiplier
        
        # 신뢰 구간 (±40%)
        min_time = estimated_time * 0.6
        max_time = estimated_time * 1.4
        
        return {
            'estimated_time_minutes': round(estimated_time / 60, 2),
            'confidence_interval': {
                'min_minutes': round(min_time / 60, 2),
                'max_minutes': round(max_time / 60, 2)
            },
            'confidence_level': 'Low',
            'breakdown': {
                'loading_minutes': round(estimated_time * 0.2 / 60, 2),
                'preprocessing_minutes': round(estimated_time * 0.3 / 60, 2),
                'execution_minutes': round(estimated_time * 0.5 / 60, 2)
            },
            'similar_cases_count': 0,
            'data_source': 'complexity_based'
        }
    
    def get_optimization_suggestions(self, user_input: Dict[str, Any], 
                                     prediction: Dict[str, Any]) -> List[str]:
        """최적화 제안 생성"""
        suggestions = []
        
        estimated_minutes = prediction['estimated_time_minutes']
        
        # 시간이 오래 걸리는 경우에만 제안
        if estimated_minutes < 5:
            return []
        
        # 하드웨어 업그레이드
        if user_input['hardware'] in ['low', 'medium']:
            suggestions.append(
                f"💻 하드웨어 업그레이드 시 {estimated_minutes * 0.4:.1f}~"
                f"{estimated_minutes * 0.25:.1f}분으로 단축 가능"
            )
        
        # 샘플링 제안
        if user_input['rows'] > 100000:
            sampled_time = estimated_minutes * 0.1
            suggestions.append(
                f"🔬 10% 샘플링 사용 시 약 {sampled_time:.1f}분으로 단축"
            )
        
        # 알고리즘 변경
        if user_input['method'] == 'clf_svm' and user_input['rows'] > 10000:
            suggestions.append(
                "⚡ SVM 대신 랜덤 포레스트 사용 시 5-10배 빠름"
            )
        
        if user_input['method'] == 'clf_forest':
            suggestions.append(
                "🌲 랜덤 포레스트의 트리 개수를 50개로 줄이면 2배 빠름"
            )
        
        # 툴 변경
        if user_input['tool'] == 'python' and user_input['method'].startswith('agg_'):
            suggestions.append(
                "🚀 Python 대신 SQL 사용 시 2-5배 빠름 (집계 작업)"
            )
        
        return suggestions[:3]  # 최대 3개만


def main():
    """테스트 실행"""
    predictor = TimePredictor('benchmark_data.json')
    
    # 테스트 케이스
    test_cases = [
        {
            'rows': 1000000,
            'columns': 50,
            'method': 'clf_forest',
            'tool': 'python',
            'hardware': 'medium'
        },
        {
            'rows': 100000,
            'columns': 20,
            'method': 'reg_linear_multiple',
            'tool': 'python',
            'hardware': 'low'
        },
        {
            'rows': 10000,
            'columns': 10,
            'method': 'agg_basic',
            'tool': 'python',
            'hardware': 'high'
        }
    ]
    
    print("\n" + "=" * 60)
    print("예측 엔진 테스트")
    print("=" * 60)
    
    for i, test_input in enumerate(test_cases, 1):
        print(f"\n📊 테스트 케이스 {i}:")
        print(f"   데이터: {test_input['rows']:,} 행 × {test_input['columns']} 열")
        print(f"   방법: {test_input['method']}")
        print(f"   하드웨어: {test_input['hardware']}")
        
        prediction = predictor.predict(test_input)
        
        print(f"\n   ⏱️  예상 시간: {prediction['estimated_time_minutes']} 분")
        print(f"   📊 신뢰 구간: {prediction['confidence_interval']['min_minutes']} ~ "
              f"{prediction['confidence_interval']['max_minutes']} 분")
        print(f"   🎯 신뢰도: {prediction['confidence_level']}")
        print(f"   📈 유사 케이스: {prediction['similar_cases_count']}개")
        print(f"   🔍 데이터 출처: {prediction['data_source']}")
        
        # 최적화 제안
        suggestions = predictor.get_optimization_suggestions(test_input, prediction)
        if suggestions:
            print(f"\n   💡 최적화 제안:")
            for suggestion in suggestions:
                print(f"      {suggestion}")
    
    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
