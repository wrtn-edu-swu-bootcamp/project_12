# 데이터 분석 방법 분류 체계

## 🎯 단계적 세분화 전략

### Phase 1: MVP (현재) - 대분류만
```
└── 분석 방법 (5개)
    ├── 단순 집계 (Aggregation)
    ├── 회귀분석 (Regression)
    ├── 분류 (Classification)
    ├── 클러스터링 (Clustering)
    └── 딥러닝 (Deep Learning)
```

**장점**: 간단, 빠른 구현
**단점**: 정확도 낮음 (±50%)

---

### Phase 2: 중분류 추가 (권장)
```
└── 회귀분석
    ├── 선형 회귀
    │   ├── 단순 선형 회귀 (O(n))
    │   ├── 다중 선형 회귀 (O(nm²))
    │   └── 다항 회귀 (degree별)
    ├── 정규화 회귀
    │   ├── 릿지 회귀
    │   └── 라쏘 회귀
    └── 비선형 회귀
        ├── 로지스틱 회귀
        └── 비선형 최소제곱

└── 분류
    ├── 선형 분류기
    │   ├── 로지스틱 회귀
    │   └── 선형 SVM
    ├── 트리 기반
    │   ├── 의사결정나무
    │   ├── 랜덤 포레스트 (n_estimators별)
    │   └── XGBoost (depth별)
    └── 앙상블
        ├── 배깅
        └── 부스팅

└── 클러스터링
    ├── 분할 기반
    │   └── K-means (k, iterations별)
    ├── 계층적
    │   ├── Agglomerative
    │   └── Divisive
    └── 밀도 기반
        ├── DBSCAN
        └── HDBSCAN
```

**장점**: 더 정확 (±30%)
**단점**: 복잡도 증가

---

### Phase 3: 소분류 + 하이퍼파라미터
```
└── K-means 클러스터링
    ├── 클러스터 개수 (k)
    │   ├── k=3: 빠름
    │   ├── k=10: 중간
    │   └── k=50: 느림
    ├── 최대 반복 횟수 (max_iter)
    │   ├── 100: 빠름
    │   ├── 300: 중간
    │   └── 1000: 느림
    └── 초기화 방법
        ├── k-means++: 표준
        └── random: 더 느림
```

**장점**: 매우 정확 (±20%)
**단점**: 사용자가 이해하기 어려움

---

## 🔍 시간 복잡도 기반 분류

### 알고리즘별 시간 복잡도

| 분석 방법 | 시간 복잡도 | 실제 시간 (100만 행) |
|----------|------------|-------------------|
| **단순 집계** |
| 평균, 합계 | O(n) | ~1초 |
| 정렬 | O(n log n) | ~5초 |
| 그룹별 집계 | O(n) | ~2초 |
| **회귀분석** |
| 단순 선형 | O(nm) | ~3초 |
| 다중 선형 | O(nm² + m³) | ~10초 |
| 릿지/라쏘 | O(nm² × iter) | ~30초 |
| **분류** |
| 로지스틱 회귀 | O(nm × iter) | ~20초 |
| 의사결정나무 | O(nm log n) | ~15초 |
| 랜덤포레스트 | O(t × nm log n) | ~60초 (t=100) |
| SVM | O(n² × m) | ~300초 |
| **클러스터링** |
| K-means | O(nkd × iter) | ~40초 (k=5) |
| 계층적 | O(n³) | ~수시간 |
| DBSCAN | O(n log n) | ~50초 |
| **딥러닝** |
| 간단한 신경망 | O(nm × layers × epochs) | ~분 단위 |
| CNN | O(n × pixels × filters × epochs) | ~시간 단위 |

여기서:
- n = 행 수
- m = 열 수
- k = 클러스터 수
- d = 차원 수
- t = 트리 개수
- iter = 반복 횟수

---

## 💻 구현 전략

### 전략 1: 점진적 확장 (추천)

**Step 1: MVP (현재 상태)**
```javascript
// 5개 대분류만
const analysisFactor = {
    'simple': 1,
    'regression': 3,
    'classification': 5,
    'clustering': 7,
    'deep_learning': 15
};
```

**Step 2: 중분류 추가**
```javascript
const analysisDetails = {
    'regression': {
        'linear_simple': 2,      // 단순 선형
        'linear_multiple': 3,    // 다중 선형
        'ridge_lasso': 5,        // 정규화
        'polynomial': 8          // 다항
    },
    'classification': {
        'logistic': 3,
        'decision_tree': 4,
        'random_forest': 8,      // 기본 100 트리
        'svm': 12
    },
    'clustering': {
        'kmeans_small': 5,       // k<10
        'kmeans_large': 10,      // k>10
        'hierarchical': 20,
        'dbscan': 7
    }
};
```

**Step 3: 파라미터 고려**
```javascript
function calculateWithParams(rows, cols, analysis, params) {
    let baseFactor = analysisFactor[analysis];
    
    // 파라미터 조정
    if (analysis === 'clustering' && params.k) {
        baseFactor *= (1 + params.k / 10);
    }
    if (params.iterations) {
        baseFactor *= Math.sqrt(params.iterations / 100);
    }
    
    return baseFactor;
}
```

---

### 전략 2: 벤치마크 데이터베이스

**핵심 아이디어**: 모든 조합을 미리 측정해두기

```json
{
  "benchmarks": [
    {
      "rows": 100000,
      "cols": 20,
      "method": "regression_linear_simple",
      "tool": "python",
      "hardware": "medium",
      "time": 2.5,
      "params": {}
    },
    {
      "rows": 100000,
      "cols": 20,
      "method": "regression_ridge",
      "tool": "python",
      "hardware": "medium",
      "time": 12.3,
      "params": {"alpha": 1.0, "max_iter": 1000}
    },
    {
      "rows": 100000,
      "cols": 20,
      "method": "clustering_kmeans",
      "tool": "python",
      "hardware": "medium",
      "time": 8.7,
      "params": {"n_clusters": 5, "max_iter": 300}
    }
  ]
}
```

**장점**:
- 실측 데이터 기반 → 높은 정확도
- 새로운 조합 추가 가능
- 유사도 기반 예측

**단점**:
- 초기 데이터 수집 필요
- 데이터베이스 관리 필요

---

### 전략 3: 사용자 선택 레벨 조정

**레벨 1: 초보자 모드 (현재)**
```
분석 방법: [회귀분석 ▼]
→ 사용자는 큰 카테고리만 선택
→ 정확도: ±50%
```

**레벨 2: 일반 모드**
```
분석 방법: [회귀분석 ▼]
세부 방법: [다중 선형 회귀 ▼]
→ 더 구체적인 선택
→ 정확도: ±30%
```

**레벨 3: 전문가 모드**
```
분석 방법: [회귀분석 ▼]
세부 방법: [릿지 회귀 ▼]
파라미터:
  - alpha: [1.0]
  - max_iter: [1000]
→ 모든 것 조정 가능
→ 정확도: ±20%
```

---

## 🎓 실무 사례 참고

### scitime 프로젝트의 접근법

scitime은 Scikit-learn 알고리즘별로:
1. **메타 학습 모델** 사용
2. 각 알고리즘의 **시간 복잡도** 반영
3. **실제 측정 데이터** 학습
4. **하이퍼파라미터** 고려

예시:
```python
# scitime의 분류
estimator_map = {
    'LinearRegression': O(nm² + m³),
    'Ridge': O(nm² × iter),
    'RandomForest': O(t × nm log n),
    'KMeans': O(nkd × iter)
}
```

---

## 📊 권장 구현 순서

### 단기 (지금)
1. ✅ 5개 대분류 유지
2. ✅ 명확한 예시 제공
   - "회귀분석 (선형, 다중)" 같이 힌트 표시

### 중기 (2-4주)
1. 중분류 추가 (선택 옵션)
   - "더 정확한 예측이 필요하신가요?" 버튼
   - 클릭 시 세부 옵션 표시
2. 벤치마크 데이터 10-20개 수집

### 장기 (1-3개월)
1. 벤치마크 데이터베이스 50개+
2. 파라미터 입력 옵션
3. 머신러닝 기반 예측 모델

---

## 💡 실용적 조언

### 사용자에게 명확한 가이드 제공

**현재 방식 개선:**
```html
<option value="regression">
  회귀분석 (선형, 다중)
  💡 예: LinearRegression, Ridge
</option>
```

**설명 추가:**
```html
<div class="method-hint">
  ℹ️ 선택한 방법: 회귀분석
  • 포함: 단순/다중 선형 회귀, 릿지, 라쏘
  • 제외: 비선형 회귀, 다항 회귀 (degree > 3)
  • 정확도: ±40%
</div>
```

---

## 🎯 결론

**현실적인 접근:**
1. **MVP**: 대분류만 → 빠른 출시
2. **Phase 2**: 사용자 피드백으로 세분화
3. **Phase 3**: 벤치마크 축적으로 정확도 향상

**핵심 원칙:**
- 완벽보다는 **점진적 개선**
- 사용자에게 **현재 정확도 명시**
- **실제 데이터로 학습**

---

**다음 액션:**
어떤 전략을 선택하시겠어요?
1. 현재 5개 유지하고 힌트만 추가
2. 중분류 10-15개로 확장
3. 벤치마크 시스템 구축
