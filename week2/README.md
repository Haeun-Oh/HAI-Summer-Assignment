# Week2
## 과제 설명
하나(**하**이**나**리)는 봄이 되어 나들이를 나갔다가 붓꽃(iris)라는 예쁜 꽃을 찾았다. 하나는 이 꽃에 관심이 생겨 꽃의 품종을 구본하고 싶어한다. 붓꽃은 총 다음과 같이 3가지 품종으로 나뉜다.

하나는 평소 과학적으로 분석하는 것을 좋아하여  꽃의 다양한 정보를 수집했다. 수집한 정보는 다음과 같다.

* Iris Versicolor
* Iris Setosa
* Iris Virginica

Column 명|의미
---|---|
SepalLengthCm|꽃받침의 길이(cm)
SepalWidthCmt|꽃받침의 너비(cm)
PetalLengthCm|꽃잎의 길이(cm)
PetalWidthCm|꽃잎의 너비(cm)

하나를 도와 붓꽃의 품종을 분류할 수 있는 인공지능을 만들어 보자!

## Dataset
Dataset은 Iris Species을 사용합니다.

Iris Species 데이터셋은 주어진 csv 링크로부터 다음의 코드를 이용하여 불러올 수 있습니다.

```python
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

iris = load_iris()

X, y =  iris.data, iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```
