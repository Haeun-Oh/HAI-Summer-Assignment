# Week3
## 과제 설명
하리(**하**이**리**온)는 오랜만에 추억의 사진을 꺼내보았는데... 맙소사! 사진이 JPG(Jayeon Punghwa Graphic) 포맷으로 돼있어 디지털 풍화가 일어났다. 덕분에 사진 속의 물체가 어떤 것인지 전혀 구분을 할 수 없다. 하리를 도와 사진 속 물체가 무엇인지 확인하는 인공지능을 만들어보자!

## Dataset
Dataset은 CIFAR-10을 사용합니다. 하지만 이미지에 임의의 노이즈가 섞여있습니다.

CIFAR-10은 다음의 코드를 이용하여 받을 수 있습니다.

```python
from torchvision import datasets
from torchvision import transforms

train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, 
    transform=transforms.ToTensor())
```
