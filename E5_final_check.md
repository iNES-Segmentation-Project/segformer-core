너는 SegFormer E5 실험 코드의 **최종 검수자**다.

목표:
E5_experiment.md 기준으로 **실험 실행 전에 필수 조건이 모두 충족되었는지 검증**하는 것.

설명 금지. 반드시 **코드 기반 검증 + 결과 + 판정** 형태로 작성해라.

---

# 🔍 검증 항목 (2개 핵심 영역)

## [1] checkpoint + resume (exp_name 기반)

다음 항목을 모두 검증:

### 1-1. 저장 경로

* last checkpoint가 `f"{exp_name}_last.pth"` 형태인지
* best checkpoint가 `f"{exp_name}_best.pth"` 형태인지

코드에서 실제 경로 생성 부분 확인

---

### 1-2. resume 로드 경로

* resume 시 `f"{exp_name}_last.pth"` 기준으로 로드하는지
* `"last.pth"` 같은 고정 파일명 사용 ❌

---

### 1-3. resume 동작 검증

다음이 실제 코드에 존재하는지 확인:

* `start_epoch = ckpt["epoch"] + 1`
* `model.load_state_dict(...)`
* `optimizer.load_state_dict(...)`
* `scheduler.load_state_dict(...)`

---

### 1-4. 학습 루프 연결

```python
for epoch in range(start_epoch, epochs + 1):
```

형태로 이어지는지 확인

---

### 1-5. 로그 출력

resume 발생 시 아래 출력 존재 여부:

```
[Resume] Loaded checkpoint from ...
[Resume] Start from epoch ...
```

---

## [2] 평가 지표 구현 (E5 필수)

다음 4개 지표가 **모두 구현 + 실제 사용되는지** 검증:

### 2-1. mIoU

* 구현되어 있고 validation에서 계산되는지

---

### 2-2. mPA (중요)

* confusion matrix 기반으로 계산되는지

정확한 식 확인:

```
mPA = mean(diag(conf_mat) / sum(conf_mat, axis=1))
```

---

### 2-3. Per-class IoU

* 클래스별 IoU 반환되는지
* 최소 Pole / Pedestrian / Bicyclist 확인 가능해야 함

---

### 2-4. GFLOPs / Params

* 모델 기준으로 계산 코드 존재하는지
* 입력 크기 (512×512) 기준인지
* 출력 또는 로그에 표시되는지

---

# ⚠️ 중요한 제한

* "구현된 것 같음", "정상으로 보임" 금지
* 반드시 코드 위치 + 검증 결과로 판단
* 없으면 FAIL로 명확히 표시

---

# 📌 출력 형식

각 항목마다 아래 형식:

[항목 이름]
코드 위치:
...
검증 결과:
...
판정:
PASS / FAIL

---

# 🧾 최종 결론

다음 3단계 중 하나로 판정:

* ❌ FAIL — 실행 불가 (핵심 누락)
* ⚠️ PARTIAL — 실행 가능하지만 부족
* ✅ READY — E5 실험 실행 가능

---

이 검증은 E5 실험 시작 전 마지막 단계다.
절대 추측하지 말고, 실제 코드 기준으로만 판단해라.