# Week4
## 과제 설명
하이리온의 할아버지가 눈이 많이 안좋으셔, 숫자 읽는 것을 도와달라고 하셨다. 단, 숫자 사진을 읽을 때 전체가 보이는 것이 아니고, 왼쪽에서 오른쪽순으로 보이는 괴상한 사진을 주셨다. RNN 모델을 이용해 숫자를 읽어보시오.

MNIST 데이터셋을 CNN 이 아닌 RNN 으로 Classification 해봅시다. 기존에는 [batch_size, image_dim(width*height)] shape 의 이미지를 2-D Convolution 을 이용해 분류를 하셨다면, 이미지의 width 를 sequence 라고 생각하고(shape: [batch_size(32), sequence_length(image_width=28), image_height(28)]) RNN 을 이용하여 이미지를 분류해보세요
