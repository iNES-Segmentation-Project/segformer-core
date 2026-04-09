너는 지금 SegFormer-B0 encoder pretrained 로딩이 “정확히 성공했는지”를 검증하는 역할이다.

설명이 아니라 반드시 **코드 + 검증 결과 기반으로 판단**해야 한다.

다음 항목을 모두 검증하고, 각 항목마다:

* 실제 확인 코드
* 출력 예시
* 정상 기준
* 판정 (PASS / FAIL)

형태로 답해라.

---

# 1. load_state_dict 결과 검증

조건:

* strict=False로 encoder에 pretrained를 로드했을 때
* missing keys / unexpected keys 출력

요구:

* 실제 코드 작성
* expected 정상 상태 설명
* 결과 해석

주의:

* "missing=0이면 좋다" 같은 추측 금지
* encoder-only 로딩 기준으로 정상 패턴 설명

---

# 2. Weight 분포 검증 (가장 중요)

다음 파라미터를 반드시 확인:

model.encoder.stages[0].patch_embed.norm.weight

요구:

* mean / std / min / max 출력 코드 작성
* pretrained vs random 초기화 분포 차이 설명

판정 기준 반드시 포함:

* pretrained일 때 mean ≈ 1 근처인 이유 설명
* random init일 때 값 분포 설명

---

# 3. KV 매핑 검증 (핵심 구조 검증)

HF:

* key.weight
* value.weight

우리 모델:

* kv.weight (dim×2)

요구:

* kv.weight shape 확인 코드
* 실제 dim×2인지 검증
* 잘못 매핑됐을 때 발생하는 문제 설명

---

# 4. 특정 파라미터 직접 비교 (가능하면)

HF 모델과 우리 모델에서 다음 비교:

* stages[0].patch_embed.proj.weight 일부 값 비교
* 또는 norm.weight 비교

요구:

* 실제 값 비교 코드
* 동일/유사 여부 판단

---

# 5. forward 안정성 검증

요구:

* dummy input (2,3,512,512)
* forward → loss 계산

판정 기준:

* loss 값이 정상 범위인지
* NaN / 폭발 여부 확인

---

# 6. 최종 판정

다음 3단계로 결론을 내려라:

* ❌ 로딩 실패 (구조 문제)
* ⚠️ 부분 성공 (일부 mismatch)
* ✅ 정상 로딩 완료

---

# 매우 중요한 제한 조건

* "아마", "추정", "가능성" 금지
* 반드시 코드 기반 검증으로만 결론
* 설명보다 검증 결과 중심으로 작성
* 실제 실험에서 바로 실행 가능한 코드로 작성

---

# 출력 형식

각 섹션을 아래 형식으로 작성:

[항목 이름]
코드:
...
출력 예시:
...
정상 기준:
...
판정:
...

---

이 검증은 E5 실험 실행 전 마지막 sanity check다.
절대 대충 설명하지 말고, 실제 검증 가능한 수준으로 작성해라.