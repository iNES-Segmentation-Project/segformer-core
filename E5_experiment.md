# E5 실험 설계 (확정)

## 1. 실험 목적

E0~E4 내부 실험에서 선정된 최적 구조에 paper-like 설정을 적용하여
**SegFormer-B0의 최대 성능을 추출**하는 실험.

- 구조 비교가 아닌 성능 극대화가 목적
- pretrained backbone + strong augmentation + differential LR의 복합 효과 검증

---

## 2. 확정 설정

### Model
| 항목 | 값 | 비고 |
|------|-----|------|
| encoder | MiT-B0 (고정) | 구조 변경 불가 |
| decoder | FPN | E0~E4 결과 기준 best |
| loss | **CE + Dice + Boundary** | 아래 가중치 설계 참고 |
| pretrained | true | ImageNet-1K MiT-B0 (HF: nvidia/mit-b0) |

### Loss 가중치 설계
```
Total Loss = 1.0 × CE + 1.0 × Dice + 0.1 × Boundary
```
- **CE (1.0):** 전체 클래스 분류 안정성 — anchor
- **Dice (1.0):** 클래스 불균형 해소, 소형 객체(Pedestrian, Bicyclist) IoU 견인
- **Boundary (0.1):** FPN이 복원한 해상도 위에서 경계 정밀도 강화
  - 0.1로 낮춘 이유: boundary loss 단독 스케일이 작아 학습 불안정 방지

구현: `CombinedLoss(mode="ce+dice+boundary", ce_weight=1.0, aux_weight=1.0, boundary_weight=0.1)`
config: `loss_type: ce_dice_boundary`

### Training 설정
| 항목 | E0~E4 (internal) | E5 (paper-like) |
|------|-----------------|-----------------|
| pretrained | false | **true** |
| augmentation | basic | **paperlike** |
| scheduler | poly | **warmup_poly** |
| differential LR | false | **true** |
| encoder lr | 6e-4 | **6e-5** (미세조정) |
| decoder lr | 6e-4 | **6e-4** (처음부터 학습) |
| epochs | 40 | **100** |
| input_size | 360×480 | **512×512** |
| warmup_ratio | — | 0.1 (전체 iter의 10%) |

### Differential LR 근거
SegFormer 논문 원칙과 동일:
`backbone lr = head lr / 10`
→ pretrained backbone은 낮은 lr로 미세조정, decoder는 높은 lr로 수렴 유도.

---

## 3. Augmentation (Paper-like)

`data/transforms.py` — `PaperlikeTransform` / config: `augmentation_type: paperlike`

### Train split 적용 순서

| 단계 | 방법 | 논문 대응 |
|------|------|----------|
| 1. RandomResize | scale ∈ [0.5, 2.0] / image=BILINEAR / mask=NEAREST | Multi-scale training |
| 2. Pad | crop size보다 작으면 패딩 (image=0, mask=255) | — |
| 3. RandomCrop | 512×512, image/mask 동일 좌표 | Random crop |
| 4. RandomHFlip | p=0.5, image/mask 동일 방향 | Random flip |
| 5. ColorJitter | brightness/contrast/saturation 각 ±0.2, 랜덤 순서 | PhotoMetricDistortion |
| 6. Normalize | ImageNet mean/std | — |

Val/Test: Resize(512×512) → Normalize only (augmentation 없음)

> hue jitter 제외: CamVid 소규모(367장)에서 효과 불명확, PIL numpy HSV 변환 대비 이득 적음

---

## 4. 평가 지표 (확정)

| 지표 | 측정 대상 | 비고 |
|------|----------|------|
| **mIoU** | 전체 클래스 평균 IoU | 핵심 지표 |
| **mPA** | 클래스별 픽셀 정확도 평균 | Dice Loss 효과 검증 |
| **Per-class IoU** | Pole / Pedestrian / Bicyclist 집중 | 소형 객체 성능 |
| **GFLOPs / Params** | 연산량 및 파라미터 수 | 경량성 확인 |

---

## 5. Pretrained Weight 로드 구조

### HF key → 프로젝트 모듈 매핑

우리 인코더는 5개 모듈로 분리되어 있고, `model.encoder.state_dict()` key 경로가
HF key와 다르기 때문에 `utils/checkpoint.py`에서 remapping을 수행한다.

```
MiTEncoder
└── stages: ModuleList
    └── [i] MiTStage
        ├── patch_embed: OverlapPatchEmbed
        │   ├── proj  (Conv2d)
        │   └── norm  (LayerNorm)
        ├── blocks: ModuleList
        │   └── [j] TransformerBlock
        │       ├── norm1 (LayerNorm)
        │       ├── attn: EfficientSelfAttention
        │       │   ├── q        (Linear: dim → dim)
        │       │   ├── kv       (Linear: dim → dim×2)  ← HF의 k+v를 cat
        │       │   ├── proj     (Linear: dim → dim)
        │       │   ├── sr       (Conv2d, sr_ratio>1만)
        │       │   └── sr_norm  (LayerNorm, sr_ratio>1만)
        │       ├── norm2 (LayerNorm)
        │       └── ffn: MixFFN
        │           ├── fc1     (Linear)
        │           ├── dw_conv (Conv2d, depthwise)
        │           └── fc2     (Linear)
        └── norm (LayerNorm)
```

### Key 변환 테이블

| HF key (encoder.block.{i}.{j}.*) | 우리 key (stages.{i}.blocks.{j}.*) | 특이사항 |
|---|---|---|
| `attention.self.query.*` | `attn.q.*` | — |
| `attention.self.key.*` + `attention.self.value.*` | `attn.kv.*` | **k,v를 cat(dim=0)** |
| `attention.output.dense.*` | `attn.proj.*` | — |
| `attention.self.sr.*` | `attn.sr.*` | stage4(sr_ratio=1)는 HF에도 없음 |
| `attention.self.layer_norm.*` | `attn.sr_norm.*` | 이름 불일치 → 별도 처리 |
| `layer_norm_1.*` | `norm1.*` | — |
| `layer_norm_2.*` | `norm2.*` | — |
| `mlp.dense1.*` | `ffn.fc1.*` | — |
| `mlp.dwconv.dwconv.*` | `ffn.dw_conv.*` | HF 내부 이중 래핑 |
| `mlp.dense2.*` | `ffn.fc2.*` | — |

| HF key | 우리 key | 특이사항 |
|---|---|---|
| `encoder.patch_embeddings.{i}.proj.*` | `stages.{i}.patch_embed.proj.*` | — |
| `encoder.patch_embeddings.{i}.layer_norm.*` | `stages.{i}.patch_embed.norm.*` | — |
| `encoder.layer_norm.{i}.*` | `stages.{i}.norm.*` | — |

### kv 합치기 핵심

HF는 key, value를 별도 Linear로 가지지만, 우리 모델은 하나로 합친 `kv = nn.Linear(dim, dim×2)`를 사용:

```python
# HF: key.weight (dim, dim) + value.weight (dim, dim)
# 우리: kv.weight (dim×2, dim)
kv.weight = torch.cat([key.weight, value.weight], dim=0)  # (dim×2, dim) ✓
kv.bias   = torch.cat([key.bias,   value.bias],   dim=0)  # (dim×2,)    ✓
```

### 로딩 흐름 요약

```
train.py:build_model()
    └── pretrained=true → utils/checkpoint.py:load_pretrained_encoder(model, "nvidia/mit-b0")
            ├── os.path.isfile("nvidia/mit-b0") = False
            │   → SegformerModel.from_pretrained("nvidia/mit-b0")  [HF 다운로드/캐시]
            ├── _remap_hf_to_ours(hf_state)
            │   ├── _RE_PATCH 파싱 → patch_embed 매핑
            │   ├── _RE_BLOCK 파싱 → blocks 매핑 (kv_buffer에서 k+v cat)
            │   └── _RE_NORM  파싱 → stage norm 매핑
            └── model.encoder.load_state_dict(our_state, strict=False)
                    ├── Missing keys   : 0  (전부 로드됨)
                    ├── Unexpected keys: 0
                    └── decoder는 건드리지 않음 (random init 유지)
```

---

## 6. 실행 방법

```bash
python scripts/train.py --config configs/e5_best.yaml
```

---

## 7. 단일 변수 원칙 준수

E5는 E0~E4와 **완전히 분리**된 실험.
E0~E4에는 pretrained / paperlike / differential_lr / warmup 적용 금지.