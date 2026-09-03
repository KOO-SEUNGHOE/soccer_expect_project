# football-predictor

<!-- BADGE:START -->
_(아직 실전 예측 채점 결과가 없습니다 — 주간 파이프라인이 몇 라운드 돌아간 뒤 표시됩니다)_
<!-- BADGE:END -->

해외 5대리그(우선 EPL) 경기 결과를 포아송 회귀로 예측하고, 북메이커 배당과
비교해 모델 성능을 검증하는 프로젝트. 상세 컨벤션은 `CLAUDE.md` 참고.

## 지금까지 만든 것

- `src/ingest/download.py` — football-data.co.uk에서 시즌별 CSV를 받는 스크립트
  (실제 환경에서 `python src/ingest/download.py --seasons 2223 2324 2425 2526` 실행)
- `src/features/build_features.py` — raw CSV 정리/여러 시즌 병합 + 배당에서
  마진 제거한 "진짜" 시장 확률 계산
- `src/features/rolling_features.py` — 데이터 누수 없이(예측 시점 이전 경기만
  사용) 팀별 최근 폼(승점)/홈-원정 득실차/휴식일수를 계산
- `src/model/poisson_model.py` — 팀별 공격력/수비력을 포아송 회귀로 추정해
  승/무/패 확률을 내는 베이스라인 모델. 위 롤링 피처를 추가 공변량으로 넣을 수 있음
- `src/evaluate/backtest.py` — 시간순 walk-forward 백테스트. 라운드(10경기)
  단위로 재학습하며 Brier score / 로그손실로 모델과 시장(배당)을 비교
- `src/evaluate/run_backtest.py` — 2022-23~2025-26 시즌 전체 데이터로
  베이스라인과 피처 추가 모델을 한 번에 비교 실행 (`python src/evaluate/run_backtest.py`)
- `tests/` — 마진 제거 로직, 포아송 확률 합=1, 롤링 피처의 미래 데이터 누수
  여부, 파이프라인 DB/채점 로직에 대한 유닛 테스트 (`python -m pytest tests/`, 26개 모두 통과)
- `src/pipeline/db.py` — 예측 이력을 저장하는 SQLite(`logs/predictions.db`) 스키마/CRUD
- `src/pipeline/predict_next_round.py` — 보유 데이터 전체로 학습한 베이스라인
  모델로 다음 라운드 예측을 만들어 DB에 저장 (football-data.org에서 예정 경기 조회)
- `src/pipeline/collect_results.py` — 시즌 CSV를 다시 받아 지난 라운드 실제
  결과를 DB의 예측과 매칭해 채점
- `src/evaluate/generate_badge.py` — DB에 쌓인 실전 채점 결과로 이 README 상단
  배지를 자동 갱신 (하드코딩 없음)
- `.github/workflows/weekly_pipeline.yml` — 매주 월요일 결과 수집, 금요일 다음
  라운드 예측을 cron으로 실행
- `dashboard/app.py` — 백테스트/실전 성능/다음 라운드 예측을 보는 Streamlit
  대시보드 (`streamlit run dashboard/app.py`)

## 백테스트 결과 (2022-23~2025-26 시즌 전체, 1520경기 / 최소 1시즌 학습 후 라운드 단위 재학습)

| 지표 | 베이스라인(팀 더미만) | +최근 폼/홈-원정 편차/휴식일수 | 시장(배당, 마진 제거) |
|---|---|---|---|
| 평가 경기 수 | 1135 | 1088 | - |
| Brier score (낮을수록 좋음) | 0.590 | 0.600 | 0.575~0.577 |
| Log loss (낮을수록 좋음) | 1.007 | 1.032 | 0.967~0.969 |

**해석**:
- 시즌 하나만 학습하고 고정해 4시즌을 예측했던 이전 방식 대신, 라운드마다
  재학습하도록 백테스트를 개선하니 베이스라인 모델이 시장에 더 가까워졌습니다
  (이전 Brier 0.605 → 0.590).
- 반면 CLAUDE.md가 제안했던 "최근 폼/홈-원정 편차/휴식일수"를 단순 선형
  공변량으로 추가한 모델은 오히려 베이스라인보다 소폭 나빠졌습니다(0.600).
  회의적으로 원인을 확인해본 결과, 승격팀처럼 표본이 아주 적은 팀은 롤링
  피처(직전 5경기)를 채울 수 없어 학습에서 통째로 빠지면서 그 팀의 공격/수비
  계수가 불안정해지는 문제가 있었습니다 (`PoissonFootballModel`에 안전장치를
  추가해 그런 경우는 예측을 아예 건너뛰도록 처리했습니다). 즉, 이번 결과는
  "피처가 쓸모없다"는 결론이 아니라 "이 형태(단순 선형 추가)로는 아직
  개선이 안 됐다"는 정직한 중간 결과입니다.
- 여전히 모델이 배당(시장)을 이기지 못하는 것은 정상적인 결과입니다 — 배당
  시장은 팀 뉴스, 부상 등 훨씬 많은 정보를 반영합니다.

## 다음 단계

1. ~~**데이터 완성**~~ (완료): 2022-23~2025-26 시즌 EPL 전체 데이터 확보.
2. ~~**피처 추가 시도**~~ (완료, 효과는 아직 미확인): 최근 폼/홈-원정 편차/
   휴식일수를 추가했지만 위 표처럼 아직 배당을 따라잡지 못함. 다음으로 시도해볼
   것: 팀별 신뢰도 가중치(표본 적은 팀에 축소 추정/regularization), 피처
   상호작용, xG 기반 피처(Understat 크롤링 필요).
3. ~~**자동화**~~ (코드 작성 완료, **실제 실행은 미검증**): `.github/workflows/weekly_pipeline.yml`이
   매주 월요일 결과 수집, 금요일 다음 라운드 예측을 cron으로 실행하도록 되어
   있음. 다만 이 개발 환경엔 git 저장소/GitHub 원격 저장소가 없어 실제
   push나 Actions 실행으로 검증하지 못했고, football-data.org API 키
   (`FOOTBALL_DATA_API_KEY`)도 없어 `predict_next_round.py`의 예정 경기 조회와
   팀 이름 매핑(`src/ingest/fixtures.py`의 `FDORG_TO_FDCOUK_NAME`)도 실제
   응답으로 확인하지 못했음. **저장소를 만들고 API 키를 등록한 뒤
   `workflow_dispatch`로 한 번 수동 실행해 검증할 것.**
4. ~~**대시보드**~~ (완료): `streamlit run dashboard/app.py`로 실행, 백테스트/실전
   성능/다음 라운드 예측 3개 탭. 브라우저로 렌더링 확인함.

## 다음으로 시도해볼 것

- 위 자동화 파이프라인의 실제 GitHub Actions 실행 검증 (저장소 생성 필요)
- 피처가 배당을 못 이기는 문제: 표본 적은 팀에 대한 축소추정/regularization,
  xG 기반 피처(Understat 크롤링) 등

## Git으로 시작하기

```bash
cd football-predictor
git init
git add .
git commit -m "feat: 포아송 베이스라인 모델과 배당 대비 백테스트 파이프라인 추가"
```

이후 GitHub에 저장소를 만들고 원격을 연결한 뒤:
1. Settings → Secrets and variables → Actions에 `FOOTBALL_DATA_API_KEY` 등록
   (https://www.football-data.org/client/register 에서 무료 발급)
2. Actions 탭에서 `Weekly Pipeline` 워크플로를 `workflow_dispatch`로 한 번
   수동 실행해 `predict`/`collect` 각각이 정상 동작하는지 확인
