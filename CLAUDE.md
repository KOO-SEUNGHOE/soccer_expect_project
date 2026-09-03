# CLAUDE.md

이 파일은 Claude Code가 이 저장소에서 작업할 때 따라야 할 규칙을 정의합니다.

## 프로젝트 개요

**이름**: football-predictor
**목표**: 해외 5대리그(EPL 등) 경기 결과를 예측하는 모델을 구축하고, 매주 자동으로
데이터를 갱신하며 예측 정확도를 누적 기록한다. 또한 북메이커 배당(implied
probability)과 모델 예측을 비교하여 시장 효율성을 분석한다.

**최종 산출물**:
- 자동화된 주간 예측 파이프라인 (GitHub Actions)
- 예측 vs 실제 결과 누적 로그 (SQLite)
- 대시보드 (예측 정확도, 배당 대비 성능 시각화)
- README에 자동 갱신되는 성능 배지

**현재 진행 상황** (참고): 2022-23~2025-26 시즌 EPL 전체 데이터 확보 완료.
라운드 단위로 재학습하는 walk-forward 백테스트로 개선, 최근 폼/홈-원정 편차/
휴식일수 피처를 추가해봤으나 아직 배당(시장)을 이기지 못함
(베이스라인 Brier 0.590, 피처 추가 0.600 vs 시장 0.575~0.577 — 자세한 내용은
README 참고). 다음 단계는 GitHub Actions 자동화와 대시보드.

---

## 폴더 구조

```
football-predictor/
├── CLAUDE.md
├── README.md
├── data/
│   ├── raw/                # 원본 CSV (football-data.co.uk 등)
│   └── processed/          # 피처 엔지니어링 결과 (parquet 권장)
├── src/
│   ├── ingest/              # 데이터 수집 (raw CSV 다운로드, football-data.org 예정 경기 API)
│   ├── features/            # 피처 생성 (롤링 폼/홈-원정 편차/휴식일수, 마진 제거 등)
│   ├── model/                # 학습/추론 코드 (포아송 회귀 등)
│   ├── evaluate/            # 백테스트, Brier score, 배당 대비 성능, README 배지 생성
│   ├── pipeline/             # 주간 파이프라인 (다음 라운드 예측 생성, 지난 결과 수집, DB)
│   └── api/                  # FastAPI 서빙 레이어 (아직 미구현)
├── dashboard/
│   └── app.py                 # Streamlit 대시보드 (백테스트/실전 성능/다음 라운드 예측)
├── logs/
│   └── predictions.db        # 예측 이력 누적 DB (주간 파이프라인이 최초 실행 시 생성)
├── .github/workflows/
│   └── weekly_pipeline.yml    # 매주 월/금 cron (실제 GitHub 저장소에 push 후 동작 —
│                               # 이 환경엔 git 저장소가 없어 실제 실행은 아직 검증 안 됨.
│                               # football-data.org API 키(FOOTBALL_DATA_API_KEY)도 필요)
└── tests/                     # pytest 유닛 테스트 (마진 제거, 포아송 확률, 롤링 피처 누수,
                                # 파이프라인 DB/채점 로직 등)
```

---

## 기술 스택

- **언어**: Python 3.11+
- **데이터 처리**: pandas
- **모델링**: statsmodels(포아송 회귀 GLM), scipy
- **DB**: SQLite (`logs/predictions.db`, 스키마/CRUD `src/pipeline/db.py`)
- **API**: FastAPI (아직 미구현)
- **대시보드**: Streamlit (`dashboard/app.py`)
- **자동화**: GitHub Actions (`.github/workflows/weekly_pipeline.yml`, cron —
  실제 GitHub 저장소 push 후 동작 검증 필요)
- **패키지 관리**: pip + `requirements.txt`
- **테스트**: pytest (`tests/`, `python -m pytest tests/`)
- **린트/포맷**: ruff (`python -m ruff check src/ tests/`)

---

## 코딩 컨벤션

- 함수/변수명은 영어, 스네이크케이스 (`calculate_team_form`, `home_xg`)
- 타입 힌트 필수 (`def predict(match: MatchInput) -> PredictionResult:`)
- 모든 데이터 처리 함수는 순수 함수로 작성 (부작용 없이 입력→출력)하고,
  I/O(파일 읽기/쓰기, DB 접근)는 별도 레이어로 분리
- 매직 넘버 금지 — 설정값은 `config.yaml` 또는 `src/config.py`로 분리
- 각 모듈에 docstring 필수. 통계적 가정이 들어간 부분(예: 포아송 독립성 가정,
  배당 마진 제거 로직)은 근거를 주석으로 남길 것

---

## Git / 커밋 컨벤션

- 브랜치 전략: `main` (배포 가능 상태 유지) / `feature/*` (기능 단위) → PR 머지
- 브랜치 예시: `feature/poisson-model`, `feature/odds-comparison`, `feature/weekly-cron`
- 커밋 메시지: [Conventional Commits](https://www.conventionalcommits.org/) 형식
  - `feat: 포아송 회귀 기반 승/무/패 확률 계산 추가`
  - `fix: 홈/원정 xG 컬럼 매핑 오류 수정`
  - `docs: README 성능 배지 갱신`
  - `chore: 의존성 업데이트`
- 커밋은 논리적 단위로 쪼갤 것 — 예: "데이터 수집 스크립트"와 "피처 엔지니어링"은
  별도 커밋. Claude Code에게 작업을 맡길 때 "이 작업을 논리적 커밋 단위로
  나눠서 커밋해줘"라고 명시적으로 요청할 것.
- 이슈 트래커 활용: 기능 제안 → 이슈 생성 → 구현 → PR 연결 → 클로즈 흐름 유지

---

## 데이터 소스

| 용도 | 소스 | 형식 |
|---|---|---|
| 경기 결과 + 배당 | football-data.co.uk (URL 패턴: `mmz4281/{season}/E0.csv`) | CSV (시즌별) |
| xG/xGA 고급 스탯 | Understat | 크롤링 필요 (아직 미구현) |
| 예정 경기 일정 | football-data.org API (`src/ingest/fixtures.py`) | JSON (무료 티어, API 키 필요 — `FOOTBALL_DATA_API_KEY`. 이 개발 환경엔 키가 없어 실제 응답으로 검증 못함, 팀 이름 매핑도 미검증) |

- 원본 데이터는 `data/raw/`에 시즌/리그별로 저장, 절대 수정하지 않음
- 전처리 결과만 `data/processed/`에 저장, 재현 가능하도록 전처리 스크립트를 통해서만 생성
- `data/raw/E0_{2223,2324,2425,2526}.csv`: 2022-23~2025-26 시즌 EPL 전체 데이터
  확보 완료 (각 381줄, 헤더+380경기). `src/ingest/download.py --seasons <code>`로
  시즌 추가/갱신 가능.

---

## 모델링 원칙

1. **베이스라인 우선**: 복잡한 모델(ML)로 바로 가지 말고, 포아송 회귀
   베이스라인의 성능(Brier score, 로그손실)을 먼저 확보하고 비교 기준으로 삼는다.
2. **배당(implied probability) 처리**: 북메이커 마진(overround)을 반드시
   제거한 뒤 모델 확률과 비교한다 (`src/features/build_features.py`의
   `implied_probabilities` 참고).
3. **평가지표**: Brier score, 로그손실(log loss), 캘리브레이션 플롯을 기본으로 사용.
   단순 정확도(accuracy)만으로 평가하지 않는다 (승/무/패 클래스 불균형 문제).
4. **데이터 누수 금지**: 예측 시점에 알 수 없는 정보(경기 후 확정되는 xG 등)를
   피처로 사용하지 않는다. 반드시 "예측 시점까지의 데이터만" 사용해서 피처 생성.
5. **백테스트는 시간순으로**: 랜덤 train/test split이 아니라 시즌/라운드 기준
   시간순 분리로 검증한다 (`src/evaluate/backtest.py`의 walk-forward 방식 참고).

---

## 지속 운영 파이프라인 (핵심, 코드는 구현됨 — 실제 GitHub Actions 실행은 미검증)

- **매주 월요일**: 지난 라운드 실제 결과 수집(`src/pipeline/collect_results.py`,
  현재 시즌 CSV 재다운로드로 해결) → 저장했던 예측과 비교 →
  `logs/predictions.db`에 정확도 지표 기록 → `src/evaluate/generate_badge.py`로
  README 배지 갱신
- **매주 금요일**: 다음 라운드 예측 생성(`src/pipeline/predict_next_round.py`,
  football-data.org API로 예정 경기 조회) → `logs/predictions.db`에 저장
- 이 파이프라인은 `.github/workflows/weekly_pipeline.yml`에서 cron으로 실행된다.
  단, 이 개발 환경에는 git 저장소/GitHub 원격 저장소가 없고 football-data.org
  API 키도 없어 실제 워크플로 실행으로 검증하지 못했다 — 저장소를 만들고
  `FOOTBALL_DATA_API_KEY` 시크릿을 등록한 뒤 `workflow_dispatch`로 한 번
  수동 실행해 확인할 것.
- 사람 개입 없이 돌아가는 것이 목표

---

## Claude Code 작업 시 유의사항

- 새 기능 작업 전, 관련 폴더의 기존 코드 스타일을 먼저 확인하고 따를 것
  (특히 `src/model/poisson_model.py`의 docstring 스타일)
- 크롤러/API 호출 코드 작성 시 요청 간 딜레이(rate limiting) 반드시 포함
  (`src/ingest/download.py`의 `sleep_sec` 참고)
- 모델링 코드 작성 후에는 반드시 간단한 유닛 테스트를 같이 생성할 것
  (예: 포아송 확률 합이 1이 되는지, 마진 제거 로직이 올바른지) — 현재 `tests/`는 비어있음
- 회의적 태도 유지: 백테스트 성능이 비정상적으로 좋으면(예: 배당 대비 ROI가
  크게 플러스) 데이터 누수를 먼저 의심하고 검증할 것
- README의 성능 배지/지표는 하드코딩하지 말고 스크립트로 자동 생성할 것

---

## 다음 작업 우선순위

1. ~~`src/ingest/download.py`로 2022-23 ~ 2025-26 시즌 전체 데이터 받기~~ (완료)
2. ~~`requirements.txt` 작성~~ (완료)
3. ~~최근 5경기 폼, 홈/원정 편차 등 피처 추가해서 배당 대비 성능 개선 시도~~
   (완료, 효과는 아직 없음 — README "백테스트 결과" 참고. 표본 적은 팀에
   대한 regularization, xG 피처 등이 다음 후보)
4. ~~GitHub Actions 주간 자동화 파이프라인 구축~~ (코드/워크플로 작성 완료,
   `logs/predictions.db` SQLite 스키마 포함. **미검증**: git 저장소가 없어
   실제 push/Actions 실행을 못 해봤고, football-data.org API 키가 없어
   `predict_next_round.py`의 예정 경기 조회 및 팀 이름 매핑도 실호출로
   확인하지 못함 — 저장소 생성 + API 키 등록 후 반드시 한 번 수동 검증할 것)
5. ~~Streamlit 대시보드~~ (완료, `dashboard/app.py`, 브라우저로 3개 탭 모두
   렌더링 확인함 — `streamlit run dashboard/app.py`)

## 다음으로 시도해볼 것 (우선순위 밖, 후보)

- 위 자동화 파이프라인을 실제 GitHub 저장소에서 한 번 수동 실행(`workflow_dispatch`)해
  end-to-end 검증
- 피처 추가가 아직 배당을 못 이기는 문제: 표본 적은 팀에 대한 축소추정/
  regularization, xG 기반 피처(Understat 크롤링) 등
