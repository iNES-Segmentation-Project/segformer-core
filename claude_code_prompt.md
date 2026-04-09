# Claude Code 분석 지시 프롬프트

## 역할 및 목적

너는 이 프로젝트의 E5 실험 코드를 분석하는 전문 리뷰어다.
목적은 코드 작성자가 **"왜 이렇게 구현했는가"**를 논문 근거와 함께 이해하도록 돕는 것이다.
단순한 코드 요약이 아니라, 각 구현 선택이 어떤 논문/이론적 근거에서 비롯되었는지를 연결하여 설명해야 한다.

---

## 프로젝트 구조 파악

먼저 아래 명령으로 전체 구조를 파악하라.

```
find . -type f -name "*.py" | sort
```

이후 다음 순서대로 파일을 읽어라.

1. `configs/e5_best.yaml` (또는 e5 관련 config 파일)
2. `models/encoder/` 하위 전체
3. `models/decoder/` 하위 전체
4. `models/loss/` 하위 전체
5. `train.py` 또는 `trainer.py`
6. `datasets/` 하위 전체

---

## 분석 항목 및 출력 형식

각 항목에 대해 아래 형식으로 설명하라.
이모지 없이, 논문/보고서 스타일로 작성한다.

---

### 항목 1. Encoder (MiT-B0) 구조 분석

다음을 설명하라.

- Patch Embedding이 stage별로 어떻게 다르게 구현되었는가 (stride, kernel size)
- Efficient Self-Attention의 SR ratio가 각 stage에서 몇으로 설정되었는가
- Mix-FFN에서 depthwise conv가 어떤 역할을 하는가
- 4개 stage의 hidden dimension이 각각 얼마인가 (32, 64, 160, 256 여부 확인)
- pretrained weight 로딩 로직: HuggingFace `nvidia/mit-b0`에서 어떤 key를 remapping하는가

**근거로 연결할 논문**: SegFormer (Xie et al., NeurIPS 2021), Section 3.1

---

### 항목 2. Decoder (FPN) 구조 분석

다음을 설명하라.

- 4개 feature map [c1, c2, c3, c4]를 어떻게 받아서 처리하는가
- 각 feature map의 채널을 통일하는 방식 (1x1 conv 등)
- upsample 후 feature를 합산 또는 concat하는 방식
- 최종 segmentation head의 구조 (채널 수, 출력 shape)
- MLP decoder(E0)와의 구조적 차이를 코드 레벨에서 비교 설명

**근거로 연결할 논문**: FPN (Lin et al., CVPR 2017); SegFormer MLP head와의 대비

---

### 항목 3. Loss 함수 분석 (CE + Dice + Boundary)

각 loss 구성 요소에 대해 다음을 설명하라.

**CE Loss**
- ignore_index 처리 여부
- class weight 적용 여부

**Dice Loss**
- softmax 또는 sigmoid 중 어떤 방식을 사용하는가
- smooth 값이 얼마로 설정되어 있는가
- 분모에 합산 방식 (sum vs mean)

**Boundary Loss**
- boundary mask를 어떻게 생성하는가 (morphological erosion, laplacian, canny 등)
- boundary region에만 loss를 적용하는가, 아니면 가중치를 높이는가
- E4 Epoch 1에서 399초가 걸린 원인이 이 구현과 관련이 있는지 확인

**복합 loss 가중치**
- CE : Dice : Boundary의 가중치 비율이 얼마인가
- config에서 조정 가능한 구조인가

**근거로 연결할 논문**: Dice Loss (Milletari et al., 2016); Boundary Loss (Kervadec et al., 2019)

---

### 항목 4. 학습 전략 분석

**Pretrained Encoder**
- HuggingFace에서 로딩하는 코드 위치와 key remapping 로직
- encoder와 decoder의 weight 초기화 방식이 다른가
- UNEXPECTED key (classifier.bias, classifier.weight)가 발생하는 이유

**Differential Learning Rate**
- encoder lr과 decoder lr이 각각 얼마로 설정되어 있는가 (6e-5 / 6e-4 여부)
- optimizer에서 param_groups를 어떻게 분리하는가 (코드 라인 직접 인용)

**Warmup + Poly Scheduler**
- warmup 구간이 몇 epoch 또는 몇 iteration인가
- warmup 종료 후 poly decay의 power 값이 얼마인가
- E0~E4의 poly scheduler와 구현 방식이 어떻게 다른가

**Augmentation**
- 적용된 augmentation 종류를 모두 나열하라
- RandomResizeCrop의 scale range가 얼마인가
- CamVid에 맞게 mean/std normalization 값이 수정되어 있는가

**근거로 연결할 논문**: SegFormer 공식 구현 (mmsegmentation config 기준); AdamW + poly schedule

---

### 항목 5. 전체 학습 루프 분석

다음을 설명하라.

- train / val / test loop의 구조
- best model 저장 기준 (val mIoU 기준인지 확인)
- per-class IoU 계산 방식 (confusion matrix 기반인지 확인)
- E5 로그에서 epoch 100 이후 per-class IoU가 두 번 출력된 이유가 코드 어디에 있는가

---

### 항목 6. 설계 원칙 준수 여부 검증

아래 항목을 코드에서 직접 확인하고 준수 여부를 명시하라.

| 규칙 | 확인 방법 | 준수 여부 |
|------|-----------|-----------|
| Encoder는 MiT-B0와 동일한 구조 | hidden_dim, SR ratio, stage 수 확인 | |
| Encoder/Decoder/Loss가 완전히 분리 | import 구조, 클래스 독립성 확인 | |
| Decoder input: [c1,c2,c3,c4] list | forward 함수 signature 확인 | |
| MMSegmentation 미사용 | import 전체 검색 | |
| 단일 변수 원칙 (E0~E4) | config 파일 비교 | |

---

## 최종 출력 형식

위 항목을 모두 분석한 후, 아래 구조로 md 파일을 생성하라.

파일명: `E5_code_review.md`

```
# E5 코드 분석 보고서

## 1. Encoder 분석
## 2. Decoder 분석
## 3. Loss 함수 분석
## 4. 학습 전략 분석
## 5. 학습 루프 분석
## 6. 설계 원칙 준수 검증
## 7. 종합 평가
```

각 섹션은 다음 구조를 따른다.

```
### [구현 내용]
- 코드에서 확인한 사실을 기술

### [논문/이론 근거]
- 해당 구현이 어떤 논문 또는 이론에 근거하는지 설명

### [설계 의도 해석]
- 왜 이 방식을 선택했는지, 다른 방식 대비 장단점
```

---

## 주의사항

- 코드에서 직접 확인하지 못한 내용은 추측이라고 명시하라
- 수치(lr, epoch, ratio 등)는 반드시 코드에서 읽은 실제 값을 사용하라
- 논문 reference는 저자, 연도, venue를 포함하라
- 이모지 사용 금지
- 과장 없이 사실 기반으로 작성하라