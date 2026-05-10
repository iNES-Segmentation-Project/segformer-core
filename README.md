## Project Overview

SegFormer-B0 기반 경량 시맨틱 세그멘테이션 성능 개선을 목적으로 한다.
Decoder 구조와 Loss 함수를 단일 변수 원칙에 따라 체계적으로 비교하고,
최종적으로 최적 조합(E5)을 통해 성능 상한을 검증한다.

- Encoder: SegFormer-B0 구조 고정 (수정 불가)
- Decoder: MLP (baseline) vs FPN
- Loss: CE / Focal / CE+Dice / CE+Boundary / CE+Dice+Boundary
- Dataset: CamVid (11 classes) — Train 369 / Val 100 / Test 232

---

## Project Structure

```
segformer-core/
├── models/
│   ├── encoder/        # MiT-B0 (고정)
│   ├── decoder/        # MLP, FPN
│   └── loss/           # CE, Focal, Dice, Boundary
├── data/
├── configs/
├── utils/
├── scripts/
├── tests/
└── outputs/
```

---

## Setup

```bash
git clone https://github.com/iNES-Segmentation-Project/segformer-core
cd segformer-core
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

---

## Training & Evaluation

```bash
python scripts/train.py --config configs/b0_mlp_ce.yaml
python scripts/evaluate.py --config configs/b0_mlp_ce.yaml
```

---

## Experiments

| ID | Encoder | Decoder | Loss | 비고 |
|----|---------|---------|------|------|
| E0 | MiT-B0 | MLP | CE | baseline |
| E1 | MiT-B0 | FPN | CE | decoder 변경 |
| E2 | MiT-B0 | MLP | Focal | loss 변경 |
| E3 | MiT-B0 | MLP | CE+Dice | loss 변경 |
| E4 | MiT-B0 | MLP | CE+Boundary | loss 변경 |
| E5 | MiT-B0 | FPN | CE+Dice+Boundary | 복합 (pretrained + aug + diff-LR) |

E0\~E4는 단일 변수 비교 실험이며, E5는 E0\~E4 결과에 근거한 확장 실험이다.

---

## Results

### Val / Test mIoU

| ID | Val Best mIoU | Test mIoU | Val→Test 하락폭 |
|----|---------------|-----------|-----------------|
| E0 | 0.6369 | 0.5682 | -0.0687 |
| E1 | 0.6626 | **0.5829** | -0.0797 |
| E2 | 0.6503 | 0.5669 | -0.0834 |
| E3 | 0.6518 | 0.5796 | -0.0722 |
| E4 | 0.6510 | 0.5705 | -0.0805 |
| E5 | **0.8043** | **0.7572** | -0.0471 |

### Per-class IoU (Test 기준, 주요 클래스)

| Class | E0 | E1 | E3 | E5 |
|-------|----|----|----|----|
| Sky | 0.9243 | 0.9259 | 0.9241 | 0.9348 |
| Road | 0.9342 | 0.9466 | 0.9414 | 0.9682 |
| Building | 0.7892 | 0.8006 | 0.7926 | 0.8900 |
| Pole | 0.2120 | 0.2695 | 0.2546 | 0.4707 |
| SignSymbol | 0.2229 | 0.2445 | 0.2498 | 0.5455 |
| Pedestrian | 0.2734 | 0.3090 | 0.3095 | 0.6137 |
| Bicyclist | 0.3916 | 0.3925 | 0.3743 | 0.6956 |

---

## Key Findings

**Decoder 효과 (E0 → E1)**

FPN이 MLP 대비 val +0.0257, test +0.0147 향상.
두 단계에서 일관된 우위로 신뢰도가 높다.

**Loss 효과 (E2~E4 vs E0)**

- CE+Dice (E3): val +0.0149 → test +0.0114. 가장 안정적으로 일반화된 조합.
- Focal (E2): val +0.0134였으나 test -0.0013으로 역전. 일반화 실패로 신뢰도 낮음.
- CE+Boundary (E4): val +0.0141 → test +0.0023. test에서 효과 반감.

**E5 복합 실험**

test mIoU 0.7572로 E1(0.5829) 대비 +0.1743 향상.
pretrained encoder 도입으로 epoch 10 시점에서 이미 val mIoU 0.7135를 기록 — E0\~E4 최종값을 초과.
소수 클래스(Pole, SignSymbol, Pedestrian)의 대폭 개선과 val-test 하락폭 0.0471(E0\~E4 평균 0.077 대비 감소)은
pretrained encoder와 augmentation 강화의 복합 효과로 해석된다.

**구조적 한계**

E0\~E4의 val-test 하락폭(평균 0.077)은 369개 학습 데이터로 11개 클래스를 scratch 학습하는
조건에서 비롯된 구조적 한계이며, 단일 변수 변경만으로는 해소되지 않는다.

---

## Metrics

- mIoU, Per-class IoU
- Val-Test 하락폭
- FPS / FLOPs / Parameters

---

## Team

| 담당자 | 역할 |
|--------|------|
| MJ | Baseline 재현, Encoder 모듈화, 학습 파이프라인 |
| 찬호 | FPN Decoder, Loss 함수, 시각화 |

---

## Notes

- Encoder는 SegFormer-B0와 기능적으로 동일하게 유지한다.
- E0\~E4는 단일 변수 원칙을 엄수한다 (Decoder 또는 Loss 중 하나만 변경).
- E5는 E0\~E4 결과에 근거한 논리적 확장이며, 단순 비교 대상에서 분리한다.
- MMSegmentation 의존성 없이 순수 PyTorch로 구현한다.
