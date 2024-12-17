import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns  # 히트맵 생성을 위한 라이브러리
import os

# 한글 폰트 설정 (Malgun Gothic 또는 NanumGothic 사용)
plt.rcParams['font.family'] = 'Malgun Gothic'  # 또는 'NanumGothic'
plt.rcParams['axes.unicode_minus'] = False  # 마이너스 기호 깨짐 방지

# 1. 데이터 로드
data = pd.read_csv('./data/SeoulHourlyAvgAirPollution.csv')

# 2. 데이터 전처리
# 데이터 타입 확인 및 변환 (문자열을 숫자로 변환)
for column in data.columns:
    data[column] = pd.to_numeric(data[column], errors='coerce')

# 결측치 처리 (결측치를 평균으로 대체)
data.fillna(data.mean(), inplace=True)

# 특성 및 레이블 분리 (미세먼지 예측)
X = data.drop(['측정일시', '측정소명', '미세먼지(㎍/㎥)'], axis=1)  # 특성
y = data['미세먼지(㎍/㎥)']  # 레이블

# 데이터 정규화
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 3. 데이터 분할
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# 4. 모델 선택 및 훈련
model = LinearRegression()
model.fit(X_train, y_train)

# 5. 예측 및 평가
y_pred = model.predict(X_test)

# 성능 평가
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'Mean Squared Error: {mse}')
print(f'R^2 Score: {r2}')

# 6. 결과 시각화

# 저장할 디렉토리 생성 (존재하지 않는 경우)
output_dir = './results'
os.makedirs(output_dir, exist_ok=True)

# 산점도 시각화
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred)
plt.xlabel('Actual PM10')
plt.ylabel('Predicted PM10')
plt.title('Actual vs Predicted PM10 (Scatter Plot)')
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=2)  # 대각선 추가

# 그래프를 파일로 저장
scatter_save_path = os.path.join(output_dir, 'actual_vs_predicted_pm10_scatter.png')
plt.savefig(scatter_save_path)
plt.show()

# 막대 그래프 시각화 (예측과 실제 값의 차이)
plt.figure(figsize=(12, 6))
bar_width = 0.35
index = np.arange(len(y_test))

# 실제 값과 예측 값의 차이 계산
differences = y_test.values - y_pred

# 막대 그래프 그리기
plt.bar(index, y_test.values, bar_width, label='Actual PM10', color='b')
plt.bar(index + bar_width, y_pred, bar_width, label='Predicted PM10', color='r')

plt.xlabel('Sample Index')
plt.ylabel('PM10 Concentration (ug/m^3)')
plt.title('Actual vs Predicted PM10 (Bar Graph)')
plt.legend()

# 그래프를 파일로 저장
bar_save_path = os.path.join(output_dir, 'actual_vs_predicted_pm10_bar.png')
plt.savefig(bar_save_path)
plt.show()

# 7. 히트맵 생성 및 저장
# 측정소명을 제외한 숫자형 데이터만 선택하여 새로운 데이터프레임 생성
numeric_data = data.drop(columns=['측정소명']).select_dtypes(include=[np.number])

correlation_matrix = numeric_data.corr()  # 상관관계 행렬 계산

plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap='coolwarm', square=True)
plt.title('Correlation Heatmap')

# 히트맵을 파일로 저장
heatmap_save_path = os.path.join(output_dir, 'correlation_heatmap.png')
plt.savefig(heatmap_save_path)
plt.show()

from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import mean_absolute_error

# 8. 성능 평가 지표 계산 및 시각화
# 성능 지표 계산
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mse)

# 성능 지표 출력
print(f'Mean Absolute Error: {mae}')
print(f'Root Mean Squared Error: {rmse}')
print(f'Mean Squared Error: {mse}')
print(f'R^2 Score (Accuracy): {r2}')

# 성능 지표를 DataFrame으로 정리
performance_metrics = pd.DataFrame({
    'Metric': ['MAE', 'RMSE', 'MSE', 'R^2 (Accuracy)'],
    'Value': [mae, rmse, mse, r2]
})

# 성능 지표 시각화
plt.figure(figsize=(10, 6))
sns.barplot(x='Metric', y='Value', data=performance_metrics)
plt.title('Model Performance Metrics')
plt.ylabel('Value')
plt.xlabel('Metric')

# 그래프를 저장
metrics_save_path = os.path.join(output_dir, 'model_performance_metrics.png')
plt.savefig(metrics_save_path)
plt.show()

# 9. KFold 교차 검증
kfold = KFold(n_splits=5, shuffle=True, random_state=42)
cross_val_mse = -cross_val_score(model, X_scaled, y, scoring='neg_mean_squared_error', cv=kfold)
cross_val_rmse = np.sqrt(cross_val_mse)
cross_val_r2 = cross_val_score(model, X_scaled, y, scoring='r2', cv=kfold)

# KFold 결과 출력
print(f"KFold Cross Validation MSE: {cross_val_mse.mean():.4f}")
print(f"KFold Cross Validation RMSE: {cross_val_rmse.mean():.4f}")
print(f"KFold Cross Validation R^2: {cross_val_r2.mean():.4f}")

# KFold 결과를 DataFrame으로 정리
kfold_results = pd.DataFrame({
    'Fold': list(range(1, kfold.get_n_splits() + 1)),
    'MSE': cross_val_mse,
    'RMSE': cross_val_rmse,
    'R^2': cross_val_r2
})

# KFold 결과 시각화
plt.figure(figsize=(10, 6))
sns.lineplot(x='Fold', y='MSE', data=kfold_results, label='MSE', marker='o', color='blue')
sns.lineplot(x='Fold', y='RMSE', data=kfold_results, label='RMSE', marker='o', color='green')
sns.lineplot(x='Fold', y='R^2', data=kfold_results, label='R^2', marker='o', color='red')
plt.title('KFold Cross Validation Results')
plt.ylabel('Metric Value')
plt.xlabel('Fold')
plt.legend()

# 그래프를 저장
kfold_save_path = os.path.join(output_dir, 'kfold_cross_validation_results.png')
plt.savefig(kfold_save_path)
plt.show()

# KFold 결과를 CSV로 저장
kfold_csv_path = os.path.join(output_dir, 'kfold_results.csv')
kfold_results.to_csv(kfold_csv_path, index=False)

# 성능 지표 저장
performance_csv_path = os.path.join(output_dir, 'model_performance_metrics.csv')
performance_metrics.to_csv(performance_csv_path, index=False)

print(f"Performance metrics and KFold results saved to: {output_dir}")
