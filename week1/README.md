# Week1
##과제 설명
하나(하이나리)는 헬스장에 가서 사람들의 키가 모두 다르다는 것을 알게 되었다. 이후 한명씩 체중계에 올라 자신의 몸무게를 측정하는 것을 보게 되었다.

하나는 평소 과학적으로 분석하는 것을 좋아하여 헬스장에서 사람들의 키와 몸무게에 대한 데이터를 모두 수집하였다.

Column 명|	의미
Weight|	몸무게(kg)
Height|	키(cm)
하나는 어떤 사람의 키를 알면 그 사람의 몸무게를 예측할 수 있다고 생각을 하였다. 하나를 도와 사람의 몸무게를 예측하는 인공지능을 만들어보자!

##Dataset
Dataset은 weight-height.csv을 사용합니다.
남성의 키와 몸무게에 대한 데이터셋은 주어진 csv파일로부터 다음의 코드를 이용하여 불러올 수 있습니다.
```python
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

data = pd.read_csv("weight-height.csv")

X, y =  data['Height'], data['Weight']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```
