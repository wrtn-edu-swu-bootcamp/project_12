# Mid-Level Analysis Method Verification Guide

## 🎯 Goal
Verify the prediction accuracy of 17 detailed analysis methods.

## 📊 Mid-Level Classification List

### Simple Aggregation (3 methods)
1. **agg_basic** - Basic aggregation (mean, sum, count)
2. **agg_groupby** - Group-by aggregation
3. **agg_pivot** - Pivot table

### Regression (4 methods)
4. **reg_linear_simple** - Simple linear regression
5. **reg_linear_multiple** - Multiple linear regression
6. **reg_ridge** - Ridge regression
7. **reg_polynomial** - Polynomial regression

### Classification (4 methods)
8. **clf_logistic** - Logistic regression
9. **clf_tree** - Decision tree
10. **clf_forest** - Random forest (100 trees)
11. **clf_svm** - SVM

### Clustering (4 methods)
12. **clu_kmeans_small** - K-means (k < 10)
13. **clu_kmeans_large** - K-means (k > 10)
14. **clu_dbscan** - DBSCAN
15. **clu_hierarchical** - Hierarchical clustering

### Deep Learning (2 methods)
16. **dl_simple** - Simple neural network (2-3 layers)
17. **dl_deep** - Deep neural network (5+ layers)

---

## 🚀 Quick Verification Steps

### Step 1: Run Python Script
```powershell
cd c:\Users\PC\Desktop\porject12
python benchmark_test.py
```

### Step 2: Select Test Case
```
1. Basic aggregation (10,000 rows × 10 cols)
2. Multiple linear regression (100,000 rows × 20 cols)
3. Decision tree (50,000 rows × 15 cols)
4. Random forest (100,000 rows × 20 cols)
5. K-means (50,000 rows × 10 cols)
6. Custom
```

### Step 3: Check Actual Time
Example output:
```
📊 Measurement Results:
  • Total time:     12.5 seconds
```

### Step 4: Predict on Website
1. Open `index.html`
2. Enter same conditions:
   - Data rows: 100,000
   - Data columns: 20
   - Analysis method: **Multiple Linear Regression**
   - Tool: Python
   - Hardware: Medium
3. Click "Predict Time"

### Step 5: Check Accuracy
Enter predicted time in script:
```
Predicted time (seconds): 15.0

🎯 Accuracy Analysis:
  • Actual time:   12.5s
  • Predicted:     15.0s
  • Error:         2.5s
  • Error rate:    20.0%
  • Rating:        ✅ Very Accurate!
```

---

## 📋 Recommended Test Scenarios

### Scenario 1: Light Analysis (1-5s)
```
Data: 10,000 rows × 10 cols
Methods:
  - agg_basic (basic aggregation)
  - reg_linear_simple (simple regression)
  - clf_logistic (logistic regression)

Target: Within ±30%
```

### Scenario 2: Medium Analysis (10-30s)
```
Data: 100,000 rows × 20 cols
Methods:
  - reg_linear_multiple (multiple regression)
  - clf_tree (decision tree)
  - clu_kmeans_small (K-means)

Target: Within ±30%
```

### Scenario 3: Heavy Analysis (30s+)
```
Data: 100,000 rows × 20 cols
Methods:
  - clf_forest (random forest)
  - clu_kmeans_large (K-means large)
  - dl_simple (simple neural network)

Target: Within ±40%
```

---

## 🔍 Method-Specific Verification Tips

### Simple Aggregation
**Characteristics**: Very fast (< 1s)
**Key points**: 
- Linear with data size
- Low column impact

**Expected time**:
- 10,000 rows: ~0.1s
- 100,000 rows: ~0.5s
- 1,000,000 rows: ~2s

### Regression
**Characteristics**: Sensitive to column count
**Key points**:
- simple < multiple < ridge < polynomial
- More columns = slower

**Expected time (100,000 rows × 20 cols)**:
- simple: ~2s
- multiple: ~5s
- ridge: ~8s
- polynomial: ~15s

### Classification
**Characteristics**: Large variance by algorithm
**Key points**:
- logistic < tree < forest < SVM
- Random forest scales with tree count

**Expected time (100,000 rows × 20 cols)**:
- logistic: ~5s
- tree: ~8s
- forest: ~30s
- svm: ~60s (sampled)

### Clustering
**Characteristics**: Sensitive to k value
**Key points**:
- Larger k = slower
- Hierarchical is very slow

**Expected time (50,000 rows × 10 cols)**:
- kmeans (k=5): ~5s
- kmeans (k=20): ~10s
- dbscan: ~8s
- hierarchical: ~60s+ (sampled)

### Deep Learning
**Characteristics**: Scales with layers and epochs
**Key points**:
- More layers = slower
- Linear increase with epochs

**Expected time (100,000 rows × 20 cols)**:
- simple (10 epochs): ~20s
- deep (20 epochs): ~60s

---

## 📊 Verification Results Table

| Method | Data Size | Actual | Predicted | Error % | Rating |
|--------|-----------|--------|-----------|---------|--------|
| agg_basic | 10k × 10 | 0.2s | 0.3s | 50% | ⚠️ |
| reg_linear_multiple | 100k × 20 | 5.2s | 6.0s | 15% | ✅ |
| clf_forest | 100k × 20 | 32.1s | 28.5s | 11% | ✅ |
| ... | ... | ... | ... | ... | ... |

---

## 🎓 Accuracy Improvement Methods

### When Error is Large (> ±50%)

**1. Adjust Coefficients**
Modify `analysisFactor` values in `index.html`:
```javascript
'clf_forest': 8,  // Increase to 10 if too slow
```

**2. Collect Data**
- Average 3-5 actual measurements
- Test with various sizes

**3. Hardware Calibration**
Adjust for your computer specs

---

## 🔧 Troubleshooting

### Python Library Errors
```powershell
pip install --upgrade scikit-learn
pip install --upgrade pandas numpy
```

### Out of Memory
- Test with smaller data
- SVM and hierarchical clustering use sampling

### Takes Too Long
- Press Ctrl+C to cancel
- Start with smaller data

---

## 💡 Advanced Verification

### Test Multiple Sizes
```python
# Custom selection
rows = [10000, 50000, 100000]
for r in rows:
    measure_analysis_time(r, 20, 'reg_linear_multiple')
```

### Accuracy Trend Analysis
- Small data: Higher error (fixed overhead)
- Large data: Lower error (stable prediction)

---

## ✅ Verification Checklist

- [ ] Python installed
- [ ] Libraries installed (`pip install -r requirements.txt`)
- [ ] Test at least 3 methods
- [ ] Each within 30% error
- [ ] Test various data sizes
- [ ] Fill results table

---

**After Verification**: 
Commit results to GitHub and update README with accuracy info!

```powershell
git add .
git commit -m "Test: Mid-level verification complete - avg error 25%"
git push origin main
```
