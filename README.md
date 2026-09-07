# football-predictor

<!-- BADGE:START -->
![실전 예측 성능](https://img.shields.io/badge/%EC%8B%A4%EC%A0%84%20%EC%98%88%EC%B8%A1%201%EA%B2%BD%EA%B8%B0%20Brier-0.093-blue)
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
- `dashboard/app.py` — 백테스트/실전 성능/다음 라운드 예측/배트맨 프로토 배당을
  보는 Streamlit 대시보드 (`streamlit run dashboard/app.py`,
  [Streamlit Cloud에 배포됨](https://soccerexpectproject-dj8mjhiqg8hnnjvgjhkpwj.streamlit.app))
- `src/betman/` — **메인 파이프라인과 완전히 별개인 섹션.** 한국 공식
  스포츠토토(배트맨)의 "프로토 승부식"에서 EPL 경기의 실제 고정 배당(승/무/패)을
  가져온다. 자세한 내용은 아래 "배트맨 프로토 승부식 섹션" 참고.

## 백테스트 결과 (2022-23~2025-26 시즌 전체, 1520경기 / 최소 1시즌 학습 후 라운드 단위 재학습)

| 모델 | 평가 경기 수 | Brier score | Log loss |
|---|---|---|---|
| 베이스라인 (팀 더미만) | 1135 | 0.590 | 1.007 |
| + 최근 폼/홈-원정 편차/휴식일수 | 1088 | 0.600 | 1.032 |
| + 상대전적(H2H)만 | 1044 | 0.596 | 1.034 |
| + 홈-원정 편차 + 상대전적(H2H) | 1047 | 0.597 | 1.035 |
| 시장(배당, 마진 제거) | - | 0.575~0.577 | 0.967~0.969 |

**해석**:
- 시즌 하나만 학습하고 고정해 4시즌을 예측했던 이전 방식 대신, 라운드마다
  재학습하도록 백테스트를 개선하니 베이스라인 모델이 시장에 더 가까워졌습니다
  (이전 Brier 0.605 → 0.590).
- **지금까지 시도한 피처 4개(최근 폼, 홈-원정 편차, 휴식일수, 상대전적) 중
  베이스라인을 이긴 게 하나도 없습니다.** 우연이라기보다 패턴으로 보입니다 —
  회의적으로 원인을 확인해본 결과 두 가지가 겹쳐 있습니다:
  1. 승격팀처럼 표본이 아주 적은 팀은 피처(직전 5경기, 맞대결 이력 등)를
     채울 수 없어 학습에서 통째로 빠지면서 표본이 줄어듭니다 (예: H2H 추가 시
     1135건 → 1044건으로 감소).
  2. 팀 더미 변수가 이미 각 팀의 실력 차이 대부분을 설명하고 있어서, 추가
     피처들이 새 정보를 별로 못 주면서 계수 추정만 불안정하게 만드는 것으로
     보입니다 (반복적으로 뜨는 "디자인 행렬이 특이함" 경고가 근거).
  `PoissonFootballModel`에는 이런 경우 예측을 아예 건너뛰는 안전장치를 넣어뒀다.
- **정규화(ridge/L2)도 시도했지만 역시 베이스라인을 못 이겼다** (`PoissonFootballModel(l2_alpha=...)`
  로 구현, alpha를 0.00005~0.3까지 촘촘히 스윕): 베이스라인과 H2H는 alpha=0(정규화
  없음)이 최적이었고, 정규화를 걸수록 계속 나빠졌다. 가장 불안정했던
  "폼+홈원정편차+휴식일수" 조합만 alpha≈0.0005에서 0.600→0.598로 아주
  미세하게 개선됐지만 베이스라인(0.590)에는 못 미쳤다. 팀 더미 전체에 균일한
  강도로 정규화를 걸다 보니, 문제 있는 소수 팀은 안정화돼도 이미 잘 추정되던
  대다수 팀 계수까지 같이 깎여서 순손실이 더 컸던 것으로 보인다 — 애초에
  발산 문제는 `_expected_goals`의 안전장치(발산 시 예측 스킵)로 이미 완화돼
  있었다는 점도 있다.
- 여전히 모델이 배당(시장)을 이기지 못하는 것은 정상적인 결과입니다 — 배당
  시장은 팀 뉴스, 부상 등 훨씬 많은 정보를 반영합니다.

## 다음 단계

1. ~~**데이터 완성**~~ (완료): 2022-23~2025-26 시즌 EPL 전체 데이터 확보.
2. ~~**피처 추가 시도**~~ (완료, 효과는 아직 미확인): 최근 폼/홈-원정 편차/
   휴식일수를 추가했지만 위 표처럼 아직 배당을 따라잡지 못함. 다음으로 시도해볼
   것: 팀별 신뢰도 가중치(표본 적은 팀에 축소 추정/regularization), 피처
   상호작용, xG 기반 피처(Understat 크롤링 필요).
3. ~~**자동화**~~ (완료, 실제 GitHub Actions 실행으로 검증함): `.github/workflows/weekly_pipeline.yml`이
   매주 월요일 결과 수집, 금요일 다음 라운드 예측을 cron으로 실행. 실제 저장소에서
   `workflow_dispatch`로 여러 차례 수동 실행해 `predict`/`collect` 모두 정상
   동작 확인 (도중에 numpy Python 버전 문제, 승격팀 이름 매핑 3건 발견해 수정함).
4. ~~**대시보드**~~ (완료): `streamlit run dashboard/app.py`로 로컬 실행 가능,
   [Streamlit Cloud에도 배포](https://soccerexpectproject-dj8mjhiqg8hnnjvgjhkpwj.streamlit.app)해
   상시 접근 가능. 백테스트/실전 성능/다음 라운드 예측/배트맨 프로토 4개 탭.
5. ~~**배트맨 프로토 배당 섹션**~~ (완료): 아래 "배트맨 프로토 승부식 섹션" 참고.

## 배트맨 프로토 승부식 섹션 (메인 파이프라인과 완전히 별개)

한국 공식 스포츠토토 배트맨(betman.co.kr)에서 EPL 경기의 실제 배당을 가져와
보여주는 별도 섹션. 위 모델/백테스트/자동화와는 데이터 소스도, 실행 방식도
완전히 다르므로 섞지 않는다.

- **왜 "승무패"가 아니라 "프로토 승부식"인가**: 배트맨의 "축구토토 승무패"는
  실제 배당이 아니라 베팅 금액 비율(투표율%)만 제공한다 — 이 프로젝트가
  전제하는 "배당 마진 제거 → 시장 확률"과 성격이 다르다. 반면 "프로토 승부식"은
  실시간으로 오르내리는 진짜 소수점 고정 배당(예: 1.90/3.30/3.40)을 쓴다.
- **자동화 파이프라인에 포함하지 않는 이유**: 이 데이터는 페이지 로드 후 내부
  API 호출로 채워져서 Playwright(헤드리스 브라우저)가 필요하다. GitHub Actions
  같은 클라우드 IP는 이런 사이트의 봇 차단에 걸리기 쉬워서, 로컬/수동 실행
  전용으로 뒀다.
- **사용법**:
  ```bash
  pip install -r requirements-betman.txt
  playwright install chromium   # 최초 1회
  python src/betman/proto_odds.py          # 예정 경기 배당 수집
  python src/ingest/download.py --seasons 2526   # 결과 채점 전 최신화
  python src/betman/score_odds.py          # 결과 확정된 경기 채점
  ```
  실행 후 `logs/betman_odds.db`가 갱신되며, 대시보드의 "배트맨 프로토" 탭에
  바로 반영된다 (배포된 Streamlit Cloud에 반영하려면 이 DB 변경을 커밋/push).
- **팀 이름 매핑 주의**: `src/betman/proto_odds.py`의 `KR_TO_FDCOUK_NAME`은
  실제 화면에서 확인하며 채운 목록이라 완전하지 않을 수 있다 (승격/강등으로
  팀이 바뀌면 특히). 매핑에 없는 팀은 전체 실행을 막지 않고 그 경기만 건너뛰고
  경고를 남긴다 — 경고가 뜨면 해당 팀 한글 표기를 매핑에 추가할 것.

## 다음으로 시도해볼 것

- ~~정규화(ridge/L2) 도입~~ (시도함, 도움 안 됨): `PoissonFootballModel(l2_alpha=...)`로
  구현하고 alpha를 0.00005~0.3까지 스윕했지만 베이스라인/H2H는 정규화 없음이
  최적이었고, 폼+홈원정편차+휴식일수 조합만 미세하게 개선(0.600→0.598)됐을 뿐
  베이스라인(0.590)에는 못 미쳤다. 자세한 내용은 위 "백테스트 결과" 참고.
- **다음 후보**: 팀 더미 전체가 아니라 문제 있는 소수 팀(표본 부족)에만
  선택적으로 축소추정을 적용, xG 기반 피처(Understat 크롤링 필요) 등 —
  피처/정규화보다 근본적으로 다른 데이터(선수 단위 정보)가 필요할 수 있다.

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
