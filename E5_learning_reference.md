# E5 실험 설계 학습 문서
## SegFormer-B0 기반 Semantic Segmentation — 논문 근거 & 구현 연결

---

## 0. 이 문서의 목적

이 문서는 E5 실험의 각 구성 요소가 **어떤 논문에 근거하는지**, 그리고 **코드에서 어떻게 구현되는지**를 연결하여 설명하는 학습용 레퍼런스다. Claude Code가 실제 코드를 읽은 후 수치와 구현 세부사항을 채워 넣는 방식으로 함께 사용한다.

---

## 1. Encoder: MiT-B0 (Mix Transformer)

### 논문 근거

**SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers**
Xie et al., NeurIPS 2021

MiT(Mix Transformer)는 ViT 계열 encoder를 semantic segmentation에 맞게 재설계한 구조다. 핵심 설계 원칙은 세 가지다.

첫째, 계층적(hierarchical) 구조를 가진다. 4개의 stage가 각각 다른 해상도의 feature map을 생성하며, 이를 통해 고해상도 세부 정보와 저해상도 의미 정보를 동시에 보존한다. 이는 CNN의 FPN과 유사한 동기에서 출발하지만, attention 기반으로 구현된다는 점에서 다르다.

둘째, Efficient Self-Attention을 사용한다. 표준 self-attention의 계산 복잡도는 O(N²)인데, 여기서 N은 sequence length(patch 수)다. 고해상도 feature map에서는 N이 매우 커지기 때문에 계산이 불가능해진다. 이를 해결하기 위해 SR(Sequence Reduction) ratio R을 도입하여 K, V를 R² 배 축소한다. 각 stage의 SR ratio는 [8, 4, 2, 1]로 설정되며, 초기 stage일수록 더 많이 축소한다.

셋째, Mix-FFN을 사용한다. 기존 ViT의 FFN은 위치 정보를 별도 positional encoding으로 넣지만, MiT는 FFN 내부에 3×3 depthwise conv를 삽입하여 위치 정보를 암묵적으로 학습한다. 이를 통해 다양한 해상도 입력에 유연하게 대응할 수 있다.

### B0 기준 주요 수치

| Stage | 출력 해상도 | Hidden dim | SR ratio | Layers |
|-------|------------|------------|----------|--------|
| 1 | H/4 × W/4 | 32 | 8 | 2 |
| 2 | H/8 × W/8 | 64 | 4 | 2 |
| 3 | H/16 × W/16 | 160 | 2 | 2 |
| 4 | H/32 × W/32 | 256 | 1 | 2 |

총 파라미터: 약 3.7M (B0 기준)

### Pretrained Weight 로딩

`nvidia/mit-b0`는 HuggingFace에 공개된 ImageNet-1K pretrained 가중치다. 원 모델은 image classification task로 학습되었기 때문에 마지막 classification head(classifier.weight, classifier.bias)가 포함되어 있다. Segmentation 모델에 로딩할 때 이 두 key는 사용하지 않으므로 UNEXPECTED로 표시되며, 이는 정상적인 동작이다.

나머지 176개 key는 encoder의 patch embedding, attention, FFN 가중치에 해당하며 모두 정상적으로 로딩된다.

### 코드 연결 포인트
```python
# 확인해야 할 부분
# 1. SR ratio가 [8, 4, 2, 1]로 설정되어 있는가
# 2. hidden_dims가 [32, 64, 160, 256]인가
# 3. HuggingFace key remap 로직이 어디에 있는가
```

---

## 2. Decoder: FPN (Feature Pyramid Network)

### 논문 근거

**Feature Pyramid Networks for Object Detection**
Lin et al., CVPR 2017

FPN은 원래 object detection을 위해 제안되었으나, 다중 스케일 feature를 효과적으로 결합한다는 특성 때문에 segmentation에도 널리 적용된다.

핵심 아이디어는 top-down pathway와 lateral connection의 조합이다. Encoder의 각 stage에서 나온 feature map(c1~c4)은 채널 수가 다르고 해상도도 다르다. FPN은 이를 통일된 채널 수로 projection한 뒤, 고수준 feature(저해상도)를 upsample하여 저수준 feature(고해상도)와 element-wise add 또는 concat한다. 이를 통해 각 scale의 feature가 상위 scale의 의미 정보를 함께 포함하게 된다.

### E0 MLP decoder와의 비교

SegFormer 논문의 MLP decoder(E0에서 사용)는 각 stage의 feature를 동일 크기로 upsample한 뒤 단순 concat하고 linear projection하는 방식이다. 이는 파라미터 수가 적고 빠르지만, feature 간 상호작용이 제한적이다.

FPN(E1, E5에서 사용)은 feature 간 top-down 경로를 통해 정보를 전달하므로, 세부 구조(Pole, Fence 등)의 표현에 유리할 수 있다. 실제 실험에서 E1이 E0보다 test mIoU에서 +0.0147 향상을 보인 것이 이를 뒷받침한다.

### 코드 연결 포인트
```python
# 확인해야 할 부분
# 1. forward(self, features: List[Tensor]) 형태인가
# 2. lateral conv의 출력 채널이 통일되어 있는가 (예: 256)
# 3. upsample 방식이 bilinear interpolation인가
# 4. 최종 head의 출력이 (B, num_classes, H, W)인가
```

---

## 3. Loss 함수: CE + Dice + Boundary

### 3-1. Cross Entropy Loss

Segmentation의 표준 loss다. 각 픽셀을 독립적인 분류 문제로 취급하여 픽셀 단위 log-likelihood를 최대화한다.

class imbalance 문제가 있을 때 class weight를 부여하거나, 빈번하지 않은 클래스(void, background)에 ignore_index를 설정한다. CamVid에서는 일반적으로 void 클래스를 무시한다.

### 3-2. Dice Loss

**V-Net: Fully Convolutional Neural Networks for Volumetric Medical Image Segmentation**
Milletari et al., 3DV 2016

Dice Loss는 예측과 정답 간의 overlap을 직접 최대화하도록 설계된 loss다. 수식은 다음과 같다.

```
Dice Loss = 1 - (2 * |P ∩ G| + smooth) / (|P| + |G| + smooth)
```

CE Loss만 사용할 때 클래스 불균형이 심한 경우(작은 객체가 큰 영역에 묻히는 경우), Dice Loss는 각 클래스를 동등하게 취급하는 경향이 있어 보완적으로 작용한다. E3에서 CE 단독(E0) 대비 test mIoU +0.0114를 달성한 것은 이 효과와 연관이 있다.

smooth 값은 일반적으로 1.0 또는 1e-6을 사용하며, 분모가 0이 되는 것을 방지한다.

### 3-3. Boundary Loss

**Boundary Loss for Remote Sensing Imagery Semantic Segmentation**
Kervadec et al., MIDL 2019 (또는 변형 구현)

Boundary Loss는 객체 경계 영역에서의 오분류를 강하게 패널티하는 loss다. 경계 마스크를 생성하는 방식은 구현마다 다르며, 대표적으로 두 가지가 있다.

첫 번째는 ground truth mask에 morphological erosion을 적용하여 경계 픽셀을 추출하는 방식이다. 두 번째는 Laplacian 또는 Sobel 필터를 적용하는 방식이다.

E4의 Epoch 1에서 399초가 걸린 원인은 이 boundary mask 생성 과정에서 GPU 또는 CPU 기반 연산의 초기화 비용이 발생했을 가능성이 높다. 이후 epoch부터 정상 속도로 복귀한 것은 캐싱 또는 JIT 컴파일 효과로 해석 가능하다.

### 복합 Loss 가중치

CE, Dice, Boundary를 결합할 때 일반적으로 다음과 같은 가중합을 사용한다.

```
L_total = w1 * L_CE + w2 * L_Dice + w3 * L_Boundary
```

원 SegFormer 논문에서는 CE 단독을 사용하므로 복합 loss는 이 프로젝트에서 추가 실험한 부분이다. 가중치 비율은 config에서 확인해야 하며, 일반적으로 CE가 주가 되고 나머지가 보조 역할을 한다.

### 코드 연결 포인트
```python
# 확인해야 할 부분
# 1. boundary mask 생성 함수가 어디에 있는가
# 2. CE:Dice:Boundary 가중치 비율이 얼마인가
# 3. Dice loss에서 softmax를 먼저 적용하는가
# 4. ignore_index가 어떻게 처리되는가
```

---

## 4. 학습 전략

### 4-1. Pretrained Encoder + Differential Learning Rate

**근거**: SegFormer 논문 및 공식 mmsegmentation config

Pretrained encoder를 fine-tuning할 때 encoder와 decoder에 동일한 learning rate를 적용하면 두 가지 문제가 발생한다. Encoder에 너무 큰 lr을 적용하면 ImageNet에서 학습된 feature 표현이 손상된다(catastrophic forgetting). 반대로 너무 작은 lr을 적용하면 decoder가 충분히 학습되지 않는다.

이를 해결하기 위해 SegFormer 논문은 encoder lr을 decoder lr의 1/10로 설정하는 differential lr 전략을 사용한다. E5에서는 encoder 6e-5, decoder 6e-4로 설정하였으며, 비율은 논문 기준과 동일하다.

optimizer에서 이를 구현하려면 param_groups를 분리해야 한다.

```python
# 일반적인 구현 패턴
optimizer = AdamW([
    {'params': model.encoder.parameters(), 'lr': 6e-5},
    {'params': model.decoder.parameters(), 'lr': 6e-4},
], weight_decay=0.01)
```

### 4-2. Warmup + Poly Learning Rate Scheduler

**근거**: SegFormer 공식 구현; AdamW + poly schedule (160K iteration 기준)

Warmup은 학습 초기에 lr을 0에서 target lr까지 선형적으로 증가시키는 전략이다. Pretrained encoder는 이미 수렴된 가중치를 갖고 있기 때문에, 학습 초기에 갑자기 큰 gradient update를 받으면 표현이 손상될 수 있다. Warmup은 이 위험을 완화한다.

E5의 epoch 1 lr이 6.07e-06인 것은 warmup이 적용된 결과다. Epoch 10에서 6.00e-05(target encoder lr)에 도달한 후 poly decay가 시작되는 구조로 보인다.

Poly decay는 다음 수식을 따른다.

```
lr(t) = lr_base * (1 - t/T)^power
```

power는 일반적으로 1.0을 사용하며, 선형적으로 감소한다. SegFormer 논문에서는 power=1.0을 사용한다.

E0~E4의 poly scheduler와의 차이는 warmup 구간의 유무다. E0~E4에서는 epoch 1부터 6.00e-04로 시작한 반면, E5는 warmup을 거쳐 점진적으로 올라간다.

### 4-3. Paper-like Augmentation

**근거**: SegFormer 공식 구현; ADE20K/CityScapes training pipeline 참조

SegFormer 논문의 augmentation pipeline은 다음을 포함한다.

| Augmentation | 설정 |
|-------------|------|
| RandomResizeCrop | scale (0.5, 2.0), crop size 512×512 |
| RandomHorizontalFlip | p=0.5 |
| PhotoMetricDistortion | brightness, contrast, saturation, hue |
| Normalize | ImageNet mean/std 또는 데이터셋별 통계 |

CamVid에 적용할 때 crop size와 normalization 통계는 CamVid 데이터셋에 맞게 조정해야 한다. ImageNet mean/std를 그대로 사용할 수도 있으나, CamVid 전용 통계를 사용하면 더 안정적인 학습이 가능하다.

Augmentation 강화는 E5의 val-test 하락폭이 E0~E4(평균 0.077) 대비 0.047로 줄어든 원인 중 하나로 추정된다.

### 코드 연결 포인트
```python
# 확인해야 할 부분
# 1. param_groups 분리 로직의 코드 위치
# 2. warmup 구간이 몇 epoch인가 (로그상 epoch 10에서 lr=6e-5 도달)
# 3. poly power 값이 얼마인가
# 4. augmentation pipeline이 어느 파일에 정의되어 있는가
# 5. normalize mean/std가 ImageNet 기준인가 CamVid 기준인가
```

---

## 5. 전체 실험 설계 원칙 요약

### 단일 변수 원칙 (E0~E4)

| 실험 | 변경 요소 | 고정 요소 |
|------|-----------|-----------|
| E0 | 없음 (baseline) | MLP, CE, poly, scratch |
| E1 | Decoder: MLP → FPN | CE, poly, scratch |
| E2 | Loss: CE → Focal | MLP, poly, scratch |
| E3 | Loss: CE → CE+Dice | MLP, poly, scratch |
| E4 | Loss: CE → CE+Boundary | MLP, poly, scratch |

이 원칙을 지키는 이유는, 두 개 이상의 변수가 동시에 바뀌면 어느 변수가 성능 변화를 유발했는지 알 수 없기 때문이다. E0~E4는 각 변수의 독립적인 효과를 측정하는 controlled experiment다.

### E5의 위치

E5는 E0~E4에서 확인된 최선 구성(FPN + CE+Dice+Boundary)을 선택하고, 단일 변수 실험에서 배제했던 학습 전략 강화를 동시에 적용한 응용 실험이다. 단일 변수 원칙의 위반이 아니라 의도된 복합 실험이며, 비교 대상은 E0~E4가 아니라 "현재 설계로 달성 가능한 최대 성능"이다.

---

## 6. 실험 결과와 설계의 연결

| 실험 설계 의도 | 실험 결과 | 해석 |
|---------------|-----------|------|
| FPN이 MLP보다 다중 스케일 표현에 유리할 것 | E1 test mIoU +0.0147 vs E0 | 가설 지지 |
| Focal loss가 소수 클래스에 유리할 것 | E2 test mIoU -0.0013 vs E0 | 가설 불지지. 일반화 실패 |
| CE+Dice가 overlap 최적화에 유리할 것 | E3 test mIoU +0.0114 vs E0 | 가설 지지, 가장 안정적 |
| CE+Boundary가 경계 표현에 유리할 것 | E4 test mIoU +0.0023 vs E0 | 제한적 지지 |
| pretrained encoder가 소수 클래스에 기여할 것 | E5 SignSymbol 0.22→0.55, Pedestrian 0.27→0.61 | 강하게 지지 |

---

## 7. Claude Code 분석 후 채워야 할 항목

이 문서는 Claude Code가 실제 코드를 읽은 후 아래 항목을 채워 완성한다.

- [ ] Encoder의 실제 SR ratio 값 (코드에서 확인)
- [ ] FPN lateral conv의 출력 채널 수
- [ ] CE:Dice:Boundary 가중치 비율 (config에서 확인)
- [ ] Boundary mask 생성 방식 (morphological / laplacian / 기타)
- [ ] Warmup epoch 수 (코드 또는 config에서 확인)
- [ ] Poly power 값
- [ ] Augmentation 전체 목록 및 파라미터
- [ ] Normalize mean/std 값
- [ ] E4 Epoch 1 지연의 코드 수준 원인
- [ ] E5 epoch 100 per-class IoU 이중 출력의 코드 위치

---

## 참고 문헌

1. Xie, E., Wang, W., Yu, Z., Anandkumar, A., Alvarez, J. M., & Luo, P. (2021). SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers. NeurIPS 2021.
2. Lin, T. Y., Dollár, P., Girshick, R., He, K., Hariharan, B., & Belongie, S. (2017). Feature Pyramid Networks for Object Detection. CVPR 2017.
3. Milletari, F., Navab, N., & Ahmadi, S. A. (2016). V-Net: Fully Convolutional Neural Networks for Volumetric Medical Image Segmentation. 3DV 2016.
4. Kervadec, H., Bouchtiba, J., Desrosiers, C., Granger, E., Dolz, J., & Ben Ayed, I. (2019). Boundary Loss for Remote Sensing Imagery Semantic Segmentation. MIDL 2019.
5. Loshchilov, I., & Hutter, F. (2019). Decoupled Weight Decay Regularization. ICLR 2019. (AdamW)