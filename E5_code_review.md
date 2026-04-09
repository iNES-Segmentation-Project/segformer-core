# E5 코드 분석 보고서

SegFormer-B0 기반 E5 실험 코드 분석. 각 구현 선택의 논문 근거와 설계 의도를 연결하여 기술한다.

---

## 1. Encoder 분석

### [구현 내용]

**Patch Embedding (OverlapPatchEmbed — `models/encoder/overlap_patch_embed.py`)**

각 stage는 `OverlapPatchEmbed`로 시작한다. 핵심은 stride와 kernel_size의 조합이다.

| Stage | kernel_size | stride | padding | 출력 해상도 |
|-------|-------------|--------|---------|------------|
| 1     | 7           | 4      | 3       | H/4, W/4   |
| 2     | 3           | 2      | 1       | H/8, W/8   |
| 3     | 3           | 2      | 1       | H/16, W/16 |
| 4     | 3           | 2      | 1       | H/32, W/32 |

`padding = patch_size // 2`로 설정하여 stride만큼만 해상도가 줄어들도록 보장한다. 패치 간 overlap이 발생하는 이유는 이 padding 덕분이다. 각 패치 임베딩 후 `LayerNorm`이 sequence 차원에 적용된다.

**Efficient Self-Attention (`models/encoder/efficient_attention.py`)**

SR ratio가 각 stage에서 다음과 같이 설정된다 (`mit_encoder.py:B0_STAGE_CONFIGS`):

| Stage | SR ratio | K/V 시퀀스 길이 축소 |
|-------|----------|---------------------|
| 1     | 8        | N → N/64            |
| 2     | 4        | N → N/16            |
| 3     | 2        | N → N/4             |
| 4     | 1        | 축소 없음 (표준 MHSA) |

SR ratio > 1인 경우, K와 V의 입력 시퀀스를 `Conv2d(embed_dim, embed_dim, kernel_size=sr_ratio, stride=sr_ratio)`로 공간 축소한 뒤 `LayerNorm`을 적용한다. Q는 원래 해상도를 유지하므로 출력 shape은 입력과 동일하다.

K, V는 `nn.Linear(embed_dim, embed_dim*2)` 하나로 합쳐서 처리된 후 `unbind`로 분리한다. 이 구조는 HuggingFace 가중치 로드 시 k_weight와 v_weight를 `cat([k_w, v_w], dim=0)`으로 합쳐야 하는 이유다 (`utils/checkpoint.py:149-155`).

**Mix-FFN (`models/encoder/mix_ffn.py`)**

구조: `Linear(C→4C) → DWConv(3×3, groups=4C) → GELU → Linear(4C→C) → Dropout`

Depthwise Conv(3×3)는 시퀀스를 2D spatial map으로 reshape한 뒤 적용된다. 이를 통해 인접 패치 간 지역적 맥락을 학습하며, 별도의 positional embedding 없이 위치 정보를 암묵적으로 인코딩한다. mlp_ratio=4.0으로 전 stage에서 동일하게 설정된다.

**Hidden Dimensions**

`B0_STAGE_CONFIGS`에서 직접 확인:

```python
Stage 1: embed_dim=32,  num_heads=1
Stage 2: embed_dim=64,  num_heads=2
Stage 3: embed_dim=160, num_heads=5
Stage 4: embed_dim=256, num_heads=8
```

총 파라미터 약 3.7M (B0 기준).

**Pretrained Weight 로딩 (`utils/checkpoint.py`)**

`load_pretrained_encoder(model, "nvidia/mit-b0")`에서 `transformers.SegformerModel.from_pretrained()`로 HuggingFace state_dict를 받아온 뒤, `_remap_hf_to_ours()`로 key를 변환한다.

주요 remapping 대응:

| HuggingFace key | 이 프로젝트 key |
|-----------------|----------------|
| `encoder.patch_embeddings.{i}.proj.weight` | `stages.{i}.patch_embed.proj.weight` |
| `encoder.block.{i}.{j}.attention.self.query.weight` | `stages.{i}.blocks.{j}.attn.q.weight` |
| `encoder.block.{i}.{j}.attention.self.key.weight` + `value.weight` | `stages.{i}.blocks.{j}.attn.kv.weight` (cat) |
| `encoder.block.{i}.{j}.mlp.dwconv.dwconv.weight` | `stages.{i}.blocks.{j}.ffn.dw_conv.weight` |
| `encoder.layer_norm.{i}.weight` | `stages.{i}.norm.weight` |

UNEXPECTED keys (`classifier.weight`, `classifier.bias`): HuggingFace `nvidia/mit-b0`는 ImageNet-1K classification task로 학습된 모델이므로 classification head가 포함된다. Segmentation 모델에는 이 head가 없으므로 `model.encoder.load_state_dict(our_state, strict=False)` 호출 시 UNEXPECTED로 표시된다. 정상 동작이며 encoder 가중치에는 영향이 없다.

### [논문/이론 근거]

SegFormer (Xie et al., NeurIPS 2021), Section 3.1.

- Efficient Self-Attention: SR ratio R로 K, V 시퀀스를 R² 배 축소. 표준 MHSA의 O(N²) 복잡도를 O(N²/R²)으로 감소.
- Overlapping patch embedding: ViT의 non-overlapping patch와 달리 패치 경계의 연속성을 보존.
- Mix-FFN: 기존 ViT의 positional encoding을 제거하고 3×3 DWConv로 대체. 이를 통해 학습/추론 해상도가 달라도 동작 가능.

### [설계 의도 해석]

SR ratio를 얕은 stage(고해상도)에서 크게, 깊은 stage(저해상도)에서 작게 설정한 이유: stage 1에서는 N이 매우 크므로(예: 512×512 입력 → N=16384) 표준 MHSA는 계산 불가. stage 4에서는 N이 이미 작으므로(N=256) SR 없이 전체 attention을 수행한다.

B0 encoder는 CLAUDE.md에 명시된 대로 구조 변경이 금지된다. `B0_STAGE_CONFIGS` 딕셔너리가 코드 상단에 상수로 선언된 것은 이 의도를 반영한다.

---

## 2. Decoder 분석

### [구현 내용]

**FPN Decoder (`models/decoder/fpn_decoder.py`)**

입력: `[c1, c2, c3, c4]` — `List[Tensor]`

```
c1: (B,  32, H/4,  W/4 )
c2: (B,  64, H/8,  W/8 )
c3: (B, 160, H/16, W/16)
c4: (B, 256, H/32, W/32)
```

처리 파이프라인:

1. **LateralConv (1×1, BN/ReLU 없음)**: 각 stage 채널을 `fpn_dim=256`으로 통일. `LateralConv` 클래스는 `Conv2d(in_ch, fpn_dim, kernel_size=1, bias=True)` 단독 구성.
2. **Top-down pathway**: `for i in range(3, 0, -1)` 역방향 순회. `laterals[i-1] += F.interpolate(laterals[i], size=laterals[i-1].shape[2:], mode="bilinear", align_corners=False)`. F4의 의미 정보가 F3→F2→F1으로 전파된다.
3. **OutputConv (3×3, BN, ReLU)**: top-down add 이후 각 레벨을 독립적으로 정제.
4. **전체 H/4 해상도 upsample + cat**: → `(B, 4*256, H/4, W/4)`
5. **FusionConv (1×1, BN, ReLU)**: `(B, 4*256, H/4, W/4)` → `(B, 256, H/4, W/4)`
6. **Dropout2d (p=0.1)**
7. **SegHead (1×1)**: `(B, 256, H/4, W/4)` → `(B, 11, H/4, W/4)`

최종 H, W 복원 upsample은 decoder 내부가 아니라 `segformer.py:107-113`에서 수행된다:

```python
logits = F.interpolate(logits, size=(input_H, input_W), mode="bilinear", align_corners=False)
```

**MLP Decoder (E0)와의 구조적 비교**

| 항목 | MLP Decoder (E0) | FPN Decoder (E1/E5) |
|------|-----------------|---------------------|
| Stage 간 상호작용 | 없음 (독립 upsample) | top-down pathway (add) |
| Projection | LinearProjection(1×1, no BN/ReLU) | LateralConv(1×1, no BN/ReLU) |
| 비선형성 적용 횟수 | FusionConv 1회만 | OutputConv 4회 + FusionConv 1회 |
| Fusion kernel | 1×1 (논문 원본) | 1×1 (대칭 설계) |
| OutputConv kernel | 없음 | 3×3 (aliasing 정제) |

### [논문/이론 근거]

Feature Pyramid Networks for Object Detection (Lin et al., CVPR 2017).

FPN의 핵심 기여는 top-down pathway와 lateral connection의 조합이다. 고수준 semantic 정보(저해상도)와 저수준 세부 정보(고해상도)를 각 scale에서 동시에 활용할 수 있다. SegFormer MLP head는 feature 간 상호작용 없이 단순 concat을 사용하므로, 경계가 복잡하거나 크기가 작은 객체에서 상대적으로 불리하다.

E1 실험에서 test mIoU +0.0147이 확인되었으며, 이 결과가 E5의 FPN 선택 근거다.

### [설계 의도 해석]

LateralConv와 LinearProjection을 모두 BN/ReLU 없는 1×1 Conv로 설계한 것은 단일 변수 원칙을 지키기 위함이다. 비선형성의 수와 위치를 통일함으로써 "FPN의 top-down pathway 자체"가 성능 변화의 원인임을 명확하게 격리한다.

OutputConv에 3×3을 사용하는 것은 top-down add 이후 발생하는 aliasing artifact를 정제하기 위함이다. MLP Decoder에는 add 연산이 없으므로 동일하게 3×3을 쓸 이유가 없다.

---

## 3. Loss 함수 분석

### [구현 내용]

**CE Loss (`models/loss/cross_entropy.py`)**

`nn.CrossEntropyLoss(weight=None, ignore_index=255, label_smoothing=0.0, reduction="mean")` 래퍼.

- class weight: 미적용 (weight=None). 생성자 인자로 전달 가능하나 E5에서는 사용하지 않음.
- label_smoothing: 0.0 (기본값).
- ignore_index=255: CamVid의 void/unlabeled 픽셀 처리.

**Dice Loss (`models/loss/dice_loss.py`)**

- 예측 확률: `F.softmax(logits, dim=1)` (sigmoid 아닌 softmax)
- smooth=1.0 (Laplace smoothing). 희귀 클래스에서 GT와 예측 모두 비어있을 때 loss가 0이 되는 것을 방지.
- 분모 합산 방식: `cardinality = (prob + one_hot).sum(dim=(0, 2, 3))` — batch와 spatial 모두 합산 후 클래스별 벡터 `(C,)` 생성.
- `dice_per_class = (2.0 * intersection + smooth) / (cardinality + smooth)`
- 최종 loss: `1.0 - dice_per_class.mean()` — 클래스 단순 평균.
- ignore_index 처리: `valid_mask = (targets != ignore_index)`로 valid 픽셀을 직접 마스킹.

**Boundary Loss (`models/loss/boundary_loss.py`)**

경계 마스크 생성 방식: **Laplacian 필터 (3×3)**

```
Laplacian kernel:
  -1 -1 -1
  -1  8 -1
  -1 -1 -1
```

GT mask에 Laplacian을 적용하여 인접 클래스가 바뀌는 위치(경계)를 검출한다. 이후 `F.max_pool2d(boundary, kernel_size=3, stride=1, padding=1)`로 1픽셀 경계를 3픽셀로 확장(dilate)한다. `theta=3.0`, `dilate_kernel_size=3`.

loss 적용 방식: 경계 영역에만 loss를 적용하는 것이 아니라, **경계 픽셀에 높은 가중치를 부여한 weighted NLL**이다.

```
weight_map = 1.0 + (3.0 - 1.0) * boundary
           = 1.0 (내부 픽셀) / 3.0 (경계 픽셀)
loss = (nll * weight_map * valid_mask).sum() / valid_pixel_count
```

**E4 Epoch 1 지연(399초) 원인**

`_extract_boundary` 메서드 내에서 `self.laplacian.to(t.device)`가 첫 배치 처리 시 호출된다. `register_buffer`로 등록된 Laplacian 텐서가 GPU로 이동하는 과정에서 CUDA 커널 JIT 컴파일이 초기화된다. 이후 epoch에서는 컴파일된 커널이 캐시에서 재사용되므로 정상 속도로 복귀한다.

**복합 Loss 가중치 (`combined_loss.py`, `train.py:138-145`)**

E5 (`loss_type: ce_dice_boundary`):

```python
CombinedLoss(
    mode="ce+dice+boundary",
    ce_weight=1.0,
    aux_weight=1.0,       # Dice
    boundary_weight=0.1,  # Boundary
)
# L_total = 1.0*CE + 1.0*Dice + 0.1*Boundary
```

`CombinedLoss` 생성자의 `ce_weight`, `aux_weight`, `boundary_weight` 인자로 비율 조정이 가능하다. 단, E5에서는 `train.py`에 수치가 하드코딩되어 있으므로 config에서는 조정되지 않는다.

### [논문/이론 근거]

- Dice Loss: V-Net (Milletari et al., 3DV 2016). 원 제안은 의료 영상 분야이나 semantic segmentation에 범용적으로 적용된다.
- Boundary Loss: Boundary-aware Loss (Kervadec et al., MIDL 2019 변형). 경계 픽셀에 가중치를 부여하는 방식은 Detail-Sensitive Loss (Yuan et al., 2020)와도 유사하다.

### [설계 의도 해석]

CE를 anchor로 두고 Dice, Boundary를 보조 loss로 사용하는 구조는 학습 안정성을 위한 설계다. Dice Loss 단독 사용 시 초기 학습이 불안정할 수 있으며(분모가 0에 가까운 경우), CE가 baseline gradient를 제공하여 이를 안정화한다.

Boundary의 가중치를 0.1로 낮게 설정한 이유: CE와 Dice의 loss 스케일(약 0.5~2.0)에 비해 Boundary도 유사한 스케일을 가지므로, 동일한 가중치(1.0)를 적용하면 경계 학습이 과도하게 지배적이 될 수 있다. 0.1은 경계 정보를 보조적으로만 활용하기 위한 보정값이다.

---

## 4. 학습 전략 분석

### [구현 내용]

**Pretrained Encoder 로딩**

코드 위치: `utils/checkpoint.py:load_pretrained_encoder`, 호출 위치: `scripts/train.py:115-117`

```python
if cfg.get("pretrained", False):
    from utils.checkpoint import load_pretrained_encoder as load_hf_encoder
    load_hf_encoder(model, hf_model_name)
```

`pretrained=false`이면 이 블록 전체를 건너뛰므로 E0~E4에는 영향이 없다. Decoder는 pretrained 대응이 없으며 기본 PyTorch 초기화(`kaiming_uniform_` for Conv2d, `uniform_` for Linear bias)를 사용한다.

**Differential Learning Rate**

`train.py:468-488`:

```python
encoder_params = list(model.encoder.parameters())
decoder_params = list(model.decoder.parameters())

if cfg.get("differential_lr", False):
    encoder_lr = cfg["lr"]          # 6e-5
    decoder_lr = cfg["lr"] * 10    # 6e-4
else:
    encoder_lr = cfg["lr"] * 10
    decoder_lr = cfg["lr"] * 10

optimizer = torch.optim.AdamW(
    [
        {"params": encoder_params, "lr": encoder_lr},
        {"params": decoder_params, "lr": decoder_lr},
    ],
    weight_decay=cfg["weight_decay"],
)
```

E5 설정값: encoder lr = 6e-5, decoder lr = 6e-4. 비율 1:10.

`differential_lr=false`(E0~E4)인 경우 encoder도 `lr * 10 = 6e-4`를 사용한다. pretrained 가중치가 없는 경우 encoder를 낮은 lr로 학습할 이유가 없기 때문이다.

**Warmup + Poly Scheduler**

`train.py:153-175`:

```python
def build_scheduler(cfg, optimizer, total_iters):
    if scheduler_type == "warmup_poly":
        warmup_iters = int(total_iters * cfg["warmup_ratio"])   # 0.1 × total

        def warmup_poly_lr(iteration):
            if iteration < warmup_iters:
                return (iteration + 1) / warmup_iters           # 선형 증가
            progress = (iteration - warmup_iters) / max(total_iters - warmup_iters, 1)
            return (1 - progress) ** 0.9                        # poly decay
```

- warmup_ratio=0.1: 전체 iteration의 10%가 warmup 구간. 100 epoch × N_batches/epoch ≈ 10 epoch에 해당.
- poly power=0.9 (코드에서 `** 0.9`로 직접 확인).
- stepping 방식: warmup_poly는 `train_one_epoch` 내에서 **batch마다 step** (`train.py:295`). poly(E0~E4)는 메인 루프에서 **epoch마다 step** (`train.py:529`).

E0~E4와의 차이: E0~E4의 poly scheduler는 warmup 없이 epoch 1부터 target lr에서 시작한다. E5는 lr이 0에서 점진적으로 올라가며 약 10 epoch에서 target lr(encoder 6e-5)에 도달한다.

**Augmentation (`data/transforms.py:PaperlikeTransform`)**

train split 적용 순서:

| 단계 | 내용 | 파라미터 |
|------|------|---------|
| RandomResize | target size × scale | scale ∈ (0.5, 2.0), image=BILINEAR, mask=NEAREST |
| Pad if needed | crop size 미달 시 패딩 | image=RGB(0,0,0), mask=255(IGNORE_INDEX) |
| RandomCrop | (H, W) = (512, 512) | image/mask 동일 (top, left) 좌표 |
| RandomHFlip | 좌우 반전 | p=0.5, image/mask 동일 방향 |
| ColorJitter | 색상 변형 (image only) | brightness=0.2, contrast=0.2, saturation=0.2 |
| Normalize | ImageNet 통계 | mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225] |

val/test split: Resize → ToTensor → Normalize만 적용 (`PaperlikeTransform.forward:184-187`).

Normalize mean/std는 ImageNet 기준이며, CamVid 전용 통계로 수정되어 있지 않다 (`transforms.py:35-36`에서 `IMAGENET_MEAN`, `IMAGENET_STD`로 확인).

### [논문/이론 근거]

- Differential LR: SegFormer 공식 구현 (mmsegmentation config, `segformer_mit-b0_512x512_160k_ade20k.py`). encoder lr / decoder lr = 1:10 비율은 논문에서 명시.
- Warmup: Pretrained 모델 fine-tuning 시 초기 학습 불안정 방지. SegFormer 공식 구현에서 warmup_ratio=0.06~0.1 사용.
- Poly decay: DeepLab 계열 이래 segmentation에서 표준으로 사용. power=0.9는 SegFormer 공식 구현 기준.
- Paper-like augmentation: SegFormer mmsegmentation config의 `PhotoMetricDistortion` + `RandomResizeCrop` + `RandomFlip` pipeline을 기반으로 torchvision 의존성 없이 PIL로 재구현.

### [설계 의도 해석]

Warmup이 필요한 구체적 이유: pretrained encoder는 이미 ImageNet에서 수렴된 가중치를 갖고 있다. 학습 초기에 큰 gradient update를 받으면 이 표현이 손상된다(catastrophic forgetting). Warmup은 이 위험을 최소화하고, decoder가 초기 수 epoch 동안 pretrained encoder에 맞게 서서히 적응할 수 있도록 한다.

---

## 5. 학습 루프 분석

### [구현 내용]

**전체 루프 구조 (`scripts/train.py`)**

```
main()
  ├── build_model / build_criterion / optimizer / scheduler
  ├── for epoch in range(start_epoch, epochs+1):
  │     ├── train_one_epoch()     → avg_loss
  │     ├── scheduler.step()      (poly만, warmup_poly는 내부에서 처리)
  │     ├── validate()            → val_loss, miou, per_class_iou, mpa
  │     ├── if epoch % 5 == 0:   → per-class IoU 출력
  │     └── save_checkpoint()     (is_best = miou > best_miou)
  └── [test evaluation] best checkpoint 로드 → validate() on test set
```

**Best model 저장 기준**

`is_best = miou > best_miou` (`train.py:560`). **val mIoU 기준**으로 best를 판단한다. `{exp_name}_best.pth`와 `{exp_name}_last.pth`를 별도 저장한다.

**Per-class IoU 계산 (`MeanIoU` 클래스, `train.py:188-250`)**

Confusion matrix 기반. `(num_classes, num_classes)` 행렬에서 rows=GT, cols=Pred.

```python
intersection = conf.diag()                          # (C,): 각 클래스 TP
union = conf.sum(1) + conf.sum(0) - intersection   # (C,): TP+FP+FN
iou = intersection / union.clamp(min=1e-6)         # (C,): 클래스별 IoU
```

GT에 한 번도 등장하지 않은 클래스는 mIoU 계산에서 제외된다(`valid_classes = conf.sum(1) > 0`).

**E5 epoch 100에서 per-class IoU가 두 번 출력된 이유**

첫 번째 출력: epoch 100은 `if epoch % 5 == 0` 조건을 만족하므로 train 루프 내 val per-class IoU가 출력된다 (`train.py:553-557`).

두 번째 출력: 학습 루프 종료 후 test evaluation 블록 (`train.py:576-617`)에서 best checkpoint를 로드하고 test set에 대해 별도로 `validate()`를 수행한다. 이 test per-class IoU가 추가로 출력된다.

두 출력은 각각 val set과 test set의 per-class IoU이며, 서로 다른 데이터에 대한 결과다.

### [논문/이론 근거]

mIoU (mean Intersection over Union)는 semantic segmentation의 표준 평가 지표다. Confusion matrix 방식은 배치별 누적이 가능하여 메모리 효율적이다. Valid class 필터링은 학습 중에는 희귀 클래스가 일부 epoch에서 GT에 등장하지 않을 수 있기 때문이며, 이를 IoU 계산에 포함하면 분모가 0이 되어 NaN이 발생한다.

### [설계 의도 해석]

test evaluation을 학습 종료 후 best checkpoint 기준으로 1회 수행하는 구조는 val mIoU로 best를 선택한 뒤 별도 test set으로 최종 성능을 측정하는 올바른 평가 프로토콜이다. test set이 best model 선택에 개입하지 않는다.

---

## 6. 설계 원칙 준수 검증

### [구현 내용]

| 규칙 | 확인 방법 | 준수 여부 |
|------|-----------|-----------|
| Encoder는 MiT-B0와 동일한 구조 | `B0_STAGE_CONFIGS`: embed_dim=[32,64,160,256], SR=[8,4,2,1], depth=2×4=8 blocks | 준수 |
| Encoder/Decoder/Loss가 완전히 분리 | `SegFormer.__init__`에서 `self.encoder`, `self.decoder` 별도 속성. loss는 `models/loss/`에 독립적으로 위치. | 준수 |
| Decoder input: [c1,c2,c3,c4] list | `FPNDecoder.forward(self, features: List[Tensor])`, `MLPDecoder.forward(self, features: List[Tensor])` | 준수 |
| MMSegmentation 미사용 | 전체 코드에서 `mmdet`, `mmcv`, `mmengine`, `mmseg` import 없음. `transformers` (HuggingFace)만 사용. | 준수 |
| 단일 변수 원칙 (E0~E4) | E0: `model_type: mlp, loss_type: ce` / E1: `model_type: fpn, loss_type: ce` / E2: `model_type: mlp, loss_type: focal`. 각 config에서 정확히 1개 요소만 변경. | 준수 |

**각 항목의 코드 근거**

- Encoder 고정: `segformer.py:71-75` — `self.encoder = MiTEncoder(in_channels=3, ...)`. 생성자 인자로 encoder 타입을 바꿀 수 있는 경로가 없다. `MiTEncoder`는 항상 `B0_STAGE_CONFIGS`를 사용한다.
- 분리 구조: `segformer.py:100-113` — encoder forward, decoder forward, 최종 upsample이 명확하게 3단계로 분리된다. 어느 단계도 다른 단계 내부를 직접 참조하지 않는다.
- MMSeg 미사용: `train.py` imports에 `from transformers import SegformerModel`만 존재하며, 이는 pretrained 로딩 전용으로 `utils/checkpoint.py`에서만 호출된다.

### [논문/이론 근거]

단일 변수 원칙(controlled experiment)은 실험 설계의 기본 원칙이다. E0~E4에서 하나의 변수만 변경함으로써 성능 변화의 원인을 특정 구조 또는 loss로 귀인할 수 있다. 두 변수가 동시에 바뀌면 각 변수의 독립적인 기여를 측정할 수 없다.

### [설계 의도 해석]

E5는 단일 변수 원칙의 위반이 아니라 의도된 복합 실험이다. E0~E4에서 각 변수의 독립 효과를 확인한 뒤, 최선 구조에 학습 강화 설정을 일괄 적용하는 2단계 파이프라인의 두 번째 단계에 해당한다. E5의 비교 대상은 E0~E4가 아니라 "이 구조로 달성 가능한 최대 성능"이다.

---

## 7. 종합 평가

### [구현 내용 요약]

E5 실험은 다음 5개 요소로 구성된다.

1. **Encoder**: MiT-B0 고정. SR ratio [8,4,2,1], hidden dim [32,64,160,256], mix-FFN(mlp_ratio=4).
2. **Decoder**: FPN. top-down pathway를 통한 stage 간 feature 전파. fpn_dim=256.
3. **Loss**: 1.0×CE + 1.0×Dice + 0.1×Boundary. ignore_index=255.
4. **학습 전략**: pretrained encoder(nvidia/mit-b0) + differential lr(6e-5/6e-4) + warmup_poly(warmup_ratio=0.1, power=0.9) + paper-like augmentation.
5. **학습 설정**: 100 epoch, batch=4, input 512×512, weight_decay=0.01.

### [논문/이론 근거]

- SegFormer (Xie et al., NeurIPS 2021): encoder 구조, MLP decoder baseline, differential lr 전략
- FPN (Lin et al., CVPR 2017): FPN decoder의 top-down pathway 설계
- Dice Loss (Milletari et al., 3DV 2016): overlap 기반 loss
- Boundary Loss (Kervadec et al., MIDL 2019): 경계 픽셀 가중 loss
- AdamW (Loshchilov & Hutter, ICLR 2019): weight decay를 gradient 업데이트와 분리한 optimizer

### [설계 의도 해석]

E5의 각 구성 요소는 독립적인 이론적 근거를 가지며, E0~E4의 실험 결과가 그 선택을 사후적으로 지지한다. FPN 선택은 E1(+0.0147)으로, CE+Dice 조합은 E3(+0.0114)으로 검증되었다. Pretrained encoder의 효과는 소수 클래스(SignSymbol, Pedestrian)에서 특히 크게 나타났으며, 이는 ImageNet 사전 학습이 도메인 전이에 효과적임을 시사한다.

코드 전반에서 CLAUDE.md의 설계 원칙 — encoder 고정, 모듈 분리, MMSeg 미사용, 단일 변수 원칙 — 이 충실하게 반영되어 있다.
