import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import VotingClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
import seaborn as sns
import matplotlib.pyplot as plt

# 데이터 로드 및 전처리
path = './data/credit_card_prediction/'
datasets = pd.read_csv(path + 'train.csv')
data = datasets.sample(frac=0.1, random_state=123)

x = data[['ID', 'Gender', 'Age', 'Region_Code', 'Occupation', 'Channel_Code',
        'Vintage', 'Credit_Product', 'Avg_Account_Balance', 'Is_Active']]
y = data[['Is_Lead']]

x = x.drop(['ID'], axis=1)

# 데이터 시각화
# sns.pairplot(datasets)
# plt.show()

# for i in range(0, len(datasets.columns), 9):
#     sns.pairplot(data=datasets,
#                  x_vars=x.columns[i:i+9],
#                  y_vars=['y'])
#     plt.show()


ob_col = list(x.dtypes[x.dtypes=='object'].index) # object 컬럼 리스트
for col in ob_col:
    x[col] = LabelEncoder().fit_transform(x[col].values)

x = x.fillna(x.mode().iloc[0])

# 특성 삭제
#x = x.drop(['Age', 'Vintage'], axis=1)


# 데이터 분할
train_data_ratio = 0.8
x_train, x_test, y_train, y_test = train_test_split(
    x, y,
    test_size=1-train_data_ratio,
    random_state=123)

# 상관계수 히트맵(heatmap)
# sns.set(font_scale=1.2)
# sns.set(rc={'figure.figsize':(12, 9)})
# sns.heatmap(
#     data = x.corr(),
#     square = True,
#     annot = True,
#     cbar = True
# )
# plt.show()

# 파이프라인 정의
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('model', VotingClassifier(
        estimators=[('cat', CatBoostClassifier()),
                    ('lgbm', LGBMClassifier()),
                    ('xgb', XGBClassifier())],
        voting='hard',
        n_jobs=-1
    ))
])

# 파이프라인을 사용하여 모델 훈련
pipeline.fit(x_train, y_train)

# 테스트 세트 예측 및 정확도 평가
accuracy = pipeline.score(x_test, y_test)
print("Accuracy:", accuracy)

# Accuracy: 0.8712105798575789(drop X)
# Accuracy: 0.8734486266531027(drop(Age))