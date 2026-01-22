"""
머신러닝 기반 시간 예측 모델 (scitime 방식)
랜덤 포레스트를 사용한 메타 학습으로 분석 시간을 예측합니다.
"""

import json
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from typing import Dict, Any, List
import pickle


class MLTimePredictor:
    """scitime 방식의 머신러닝 시간 예측기"""
    
    def __init__(self):
        self.model = RandomForestRegressor(
            n_estimators=200,
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        )
        self.method_encoder = LabelEncoder()
        self.hardware_encoder = LabelEncoder()
        self.is_trained = False
        
        # 알고리즘과 하드웨어 카테고리 미리 정의
        self.known_methods = [
            'agg_basic', 'agg_groupby', 'agg_pivot',
            'reg_linear_simple', 'reg_linear_multiple', 'reg_ridge', 'reg_polynomial',
            'clf_logistic', 'clf_tree', 'clf_forest', 'clf_svm',
            'clu_kmeans_small', 'clu_kmeans_large', 'clu_dbscan', 'clu_hierarchical',
            'dl_simple', 'dl_deep'
        ]
        self.known_hardware = ['low', 'medium', 'high', 'ultra']
    
    def train(self, benchmark_file='benchmark_data.json'):
        """벤치마크 데이터로 모델 학습"""
        print("\n" + "=" * 60)
        print("머신러닝 모델 학습 시작 (scitime 방식)")
        print("=" * 60)
        
        # 벤치마크 데이터 로드
        try:
            with open(benchmark_file, 'r', encoding='utf-8') as f:
                benchmarks = json.load(f)
        except FileNotFoundError:
            print(f"❌ {benchmark_file}을 찾을 수 없습니다.")
            return False
        
        if len(benchmarks) == 0:
            print("❌ 벤치마크 데이터가 없습니다.")
            return False
        
        print(f"📊 {len(benchmarks)}개의 벤치마크 데이터 로드")
        
        # 인코더 먼저 학습
        self.method_encoder.fit(self.known_methods)
        self.hardware_encoder.fit(self.known_hardware)
        
        # 특성과 타겟 준비
        X = self._prepare_features(benchmarks)
        y = np.log([b['elapsed_time_seconds'] + 1e-10 for b in benchmarks])  # log scale
        
        # 모델 학습
        print("🧠 랜덤 포레스트 모델 학습 중...")
        self.model.fit(X, y)
        self.is_trained = True
        
        # 학습 성능 평가
        train_score = self.model.score(X, y)
        y_pred = self.model.predict(X)
        
        # 실제 시간 스케일로 변환하여 오차 계산
        y_actual = np.exp(y)
        y_pred_actual = np.exp(y_pred)
        
        errors = np.abs(y_pred_actual - y_actual) / y_actual * 100
        mean_error = np.mean(errors)
        median_error = np.median(errors)
        
        print(f"✅ 학습 완료!")
        print(f"   R² Score: {train_score:.3f}")
        print(f"   평균 오차: ±{mean_error:.1f}%")
        print(f"   중앙값 오차: ±{median_error:.1f}%")
        
        # 특성 중요도 출력
        feature_names = self._get_feature_names()
        importances = self.model.feature_importances_
        
        print(f"\n📈 특성 중요도 (Top 5):")
        importance_pairs = sorted(zip(feature_names, importances), 
                                 key=lambda x: x[1], reverse=True)
        for name, importance in importance_pairs[:5]:
            print(f"   {name}: {importance:.3f}")
        
        return True
    
    def _prepare_features(self, data: List[Dict]) -> np.ndarray:
        """특성 벡터 준비"""
        features = []
        
        for item in data:
            # 로그 스케일 특성
            log_rows = np.log10(item['rows'] + 1)
            log_cols = np.log10(item['columns'] + 1)
            
            # 알고리즘 인코딩
            method = item['method']
            if method in self.known_methods:
                method_encoded = self.method_encoder.transform([method])[0]
            else:
                method_encoded = -1  # 알 수 없는 알고리즘
            
            # 하드웨어 인코딩
            hardware = item['hardware']
            if hardware in self.known_hardware:
                hardware_encoded = self.hardware_encoder.transform([hardware])[0]
            else:
                hardware_encoded = 1  # 기본값 medium
            
            # 데이터 타입 비율
            data_type = item.get('data_type_ratio', {})
            numeric_ratio = data_type.get('numeric', 0.7)
            categorical_ratio = data_type.get('categorical', 0.2)
            text_ratio = data_type.get('text', 0.1)
            
            # 특성 벡터 구성
            feature_vector = [
                log_rows,
                log_cols,
                method_encoded,
                hardware_encoded,
                numeric_ratio,
                categorical_ratio,
                text_ratio,
                log_rows * log_cols,  # 상호작용 특성
                log_rows * method_encoded,  # 상호작용 특성
            ]
            
            features.append(feature_vector)
        
        return np.array(features)
    
    def _get_feature_names(self) -> List[str]:
        """특성 이름 목록"""
        return [
            'log_rows',
            'log_cols',
            'method_encoded',
            'hardware_encoded',
            'numeric_ratio',
            'categorical_ratio',
            'text_ratio',
            'rows_x_cols',
            'rows_x_method'
        ]
    
    def predict(self, user_input: Dict[str, Any]) -> Dict[str, Any]:
        """사용자 입력에 대한 시간 예측"""
        if not self.is_trained:
            raise ValueError("모델이 학습되지 않았습니다. train()을 먼저 호출하세요.")
        
        # 특성 준비
        X = self._prepare_features([user_input])
        
        # 예측 (log scale)
        log_time_pred = self.model.predict(X)[0]
        
        # 실제 시간으로 변환
        predicted_time = np.exp(log_time_pred)
        
        # 신뢰 구간 추정 (랜덤 포레스트의 개별 트리 예측 사용)
        tree_predictions = []
        for estimator in self.model.estimators_:
            tree_pred = estimator.predict(X)[0]
            tree_predictions.append(np.exp(tree_pred))
        
        # 백분위수로 신뢰 구간 계산
        percentile_25 = np.percentile(tree_predictions, 25)
        percentile_75 = np.percentile(tree_predictions, 75)
        
        # 보수적으로 확장
        min_time = percentile_25 * 0.7
        max_time = percentile_75 * 1.3
        
        # 신뢰도 계산
        std_dev = np.std(tree_predictions)
        cv = std_dev / predicted_time if predicted_time > 0 else 1.0
        
        if cv < 0.25:
            confidence_level = "High"
            confidence_percent = 20
        elif cv < 0.4:
            confidence_level = "Medium"
            confidence_percent = 30
        else:
            confidence_level = "Low"
            confidence_percent = 40
        
        # 단계별 분해
        breakdown = {
            'loading_minutes': round(predicted_time * 0.2 / 60, 2),
            'preprocessing_minutes': round(predicted_time * 0.3 / 60, 2),
            'execution_minutes': round(predicted_time * 0.5 / 60, 2)
        }
        
        return {
            'estimated_time_minutes': round(predicted_time / 60, 2),
            'confidence_interval': {
                'min_minutes': round(min_time / 60, 2),
                'max_minutes': round(max_time / 60, 2)
            },
            'confidence_level': confidence_level,
            'confidence_percent': confidence_percent,
            'breakdown': breakdown,
            'data_source': 'ml_model'
        }
    
    def save_model(self, filename='ml_model.pkl'):
        """학습된 모델 저장"""
        if not self.is_trained:
            print("⚠️  학습되지 않은 모델은 저장할 수 없습니다.")
            return False
        
        model_data = {
            'model': self.model,
            'method_encoder': self.method_encoder,
            'hardware_encoder': self.hardware_encoder,
            'known_methods': self.known_methods,
            'known_hardware': self.known_hardware,
            'is_trained': self.is_trained
        }
        
        with open(filename, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"💾 모델을 '{filename}'에 저장했습니다.")
        return True
    
    def load_model(self, filename='ml_model.pkl'):
        """저장된 모델 로드"""
        try:
            with open(filename, 'rb') as f:
                model_data = pickle.load(f)
            
            self.model = model_data['model']
            self.method_encoder = model_data['method_encoder']
            self.hardware_encoder = model_data['hardware_encoder']
            self.known_methods = model_data['known_methods']
            self.known_hardware = model_data['known_hardware']
            self.is_trained = model_data['is_trained']
            
            print(f"✅ '{filename}'에서 모델을 로드했습니다.")
            return True
        except FileNotFoundError:
            print(f"❌ '{filename}' 파일을 찾을 수 없습니다.")
            return False
        except Exception as e:
            print(f"❌ 모델 로드 실패: {str(e)}")
            return False


def main():
    """테스트 실행"""
    predictor = MLTimePredictor()
    
    # 모델 학습
    if predictor.train('benchmark_data.json'):
        # 모델 저장
        predictor.save_model()
        
        # 테스트 케이스
        print("\n" + "=" * 60)
        print("예측 테스트")
        print("=" * 60)
        
        test_cases = [
            {
                'rows': 1000000,
                'columns': 50,
                'method': 'clf_forest',
                'hardware': 'medium',
                'data_type_ratio': {'numeric': 0.7, 'categorical': 0.2, 'text': 0.1}
            },
            {
                'rows': 100000,
                'columns': 20,
                'method': 'reg_linear_multiple',
                'hardware': 'low',
                'data_type_ratio': {'numeric': 0.8, 'categorical': 0.15, 'text': 0.05}
            },
            {
                'rows': 50000,
                'columns': 15,
                'method': 'clf_svm',
                'hardware': 'high',
                'data_type_ratio': {'numeric': 0.6, 'categorical': 0.3, 'text': 0.1}
            }
        ]
        
        for i, test_input in enumerate(test_cases, 1):
            print(f"\n📊 테스트 케이스 {i}:")
            print(f"   데이터: {test_input['rows']:,} 행 × {test_input['columns']} 열")
            print(f"   방법: {test_input['method']}")
            print(f"   하드웨어: {test_input['hardware']}")
            
            result = predictor.predict(test_input)
            
            print(f"\n   ⏱️  예상 시간: {result['estimated_time_minutes']} 분")
            print(f"   📊 신뢰 구간: {result['confidence_interval']['min_minutes']} ~ "
                  f"{result['confidence_interval']['max_minutes']} 분")
            print(f"   🎯 신뢰도: {result['confidence_level']} (±{result['confidence_percent']}%)")
            print(f"   🔍 데이터 출처: {result['data_source']}")
        
        print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
