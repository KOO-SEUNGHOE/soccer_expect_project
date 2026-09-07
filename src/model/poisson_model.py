"""팀별 공격력(attack)/수비력(defense) 파라미터를 포아송 회귀로 추정하고,
승/무/패 확률을 계산하는 베이스라인 모델.

아이디어 (Maher 1982 / Dixon-Coles 계열의 단순화 버전):
    log(홈팀 득점 기대값) = intercept + home_advantage + attack[홈팀] - defense[원정팀]
    log(원정팀 득점 기대값) = intercept + attack[원정팀] - defense[홈팀]

각 팀의 공격력/수비력을 "홈에서 넣은 골"과 "원정에서 넣은 골"을 모두 팀 더미
변수로 취급해 하나의 포아송 회귀로 동시에 추정한다.

feature_cols로 "최근 폼/홈-원정 편차/휴식일수" 같은 팀 관점의 추가 공변량을
넣을 수 있다 (src/features/rolling_features.py 참고). 학습 데이터에는
home_{name}/away_{name} 컬럼이 있어야 하고, 예측 시에는 predict_proba에
같은 이름의 키를 가진 home_features/away_features 딕셔너리를 넘겨야 한다.

l2_alpha(기본 0, 정규화 없음)로 L2(ridge) 정규화를 켤 수 있다. 백테스트에서
확인된 문제(README "백테스트 결과" 참고) — 정규화 없는 GLM은 표본이 아주
적은 팀(승격 직후 등)의 계수가 극단값으로 튀기 쉬운데, ridge는 그 계수들을
0 쪽으로 당겨서 이 불안정성을 완화하는 게 목적이다. 절편(Intercept)은
관례상 규제하지 않는다.

use_dixon_coles=True로 Dixon-Coles(1997) 저득점 보정을 켤 수 있다. 홈/원정
득점을 완전히 독립인 포아송으로 가정하면 저득점 스코어(0-0, 1-1)의 실제
상관관계를 놓쳐서 무승부 확률을 구조적으로 과소평가하는 경향이 있다(사용자
피드백: 정배 위주 예측이 아니라 무승부를 더 잘 잡고 싶음). (0,0)/(1,0)/(0,1)/
(1,1) 네 스코어에만 보정 계수 τ를 곱해 이를 완화한다. τ의 강도를 정하는
파라미터 ρ는 fit() 시점에 이미 추정된 λ(기대득점)를 고정한 채, 실제 결과에
대한 로그우도를 최대화하는 값으로 별도 추정한다(2단계 근사 — 전체 파라미터를
한 번에 재추정하는 원 논문의 완전 결합 추정보다 간단하지만, 기존 GLM 적합
로직을 그대로 재사용할 수 있다).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy.optimize import minimize_scalar
from scipy.stats import poisson

# Dixon-Coles 원 논문 및 후속 연구에서 실제로 추정되는 ρ는 보통 -0.2~0.2 범위를
# 크게 벗어나지 않는다. 범위를 더 넓게(-1~1) 두면 표본이 작을 때 노이즈에 과적합돼
# ρ가 극단값(예: -0.999)으로 튀면서 특정 예측에서 τ가 음수가 되는(=확률이 음수가
# 되는) 사고가 날 수 있다 (유닛 테스트의 작은 합성 데이터에서 실제로 재현됨).
_DC_RHO_BOUNDS = (-0.2, 0.2)


def _dc_tau(x: int, y: int, lambda_home: float, lambda_away: float, rho: float) -> float:
    """Dixon-Coles 저득점 보정 계수. 네 스코어 외에는 보정하지 않는다(τ=1)."""
    if x == 0 and y == 0:
        return 1 - lambda_home * lambda_away * rho
    if x == 0 and y == 1:
        return 1 + lambda_home * rho
    if x == 1 and y == 0:
        return 1 + lambda_away * rho
    if x == 1 and y == 1:
        return 1 - rho
    return 1.0


def _dc_tau_vectorized(x: np.ndarray, y: np.ndarray, lam_h: np.ndarray, lam_a: np.ndarray, rho: float) -> np.ndarray:
    """_dc_tau의 배열 버전 (ρ 추정 시 전체 학습 데이터에 대해 한 번에 계산)."""
    tau = np.ones_like(lam_h, dtype=float)
    m00 = (x == 0) & (y == 0)
    m01 = (x == 0) & (y == 1)
    m10 = (x == 1) & (y == 0)
    m11 = (x == 1) & (y == 1)
    tau[m00] = 1 - lam_h[m00] * lam_a[m00] * rho
    tau[m01] = 1 + lam_h[m01] * rho
    tau[m10] = 1 + lam_a[m10] * rho
    tau[m11] = 1 - rho
    return tau


class PoissonFootballModel:
    def __init__(
        self,
        feature_cols: list[str] | None = None,
        l2_alpha: float = 0.0,
        use_dixon_coles: bool = False,
    ) -> None:
        self.model = None
        self.teams: list[str] = []
        self.feature_cols = feature_cols or []
        self.l2_alpha = l2_alpha
        self.use_dixon_coles = use_dixon_coles
        self.rho = 0.0

    def _to_long_format(self, matches: pd.DataFrame) -> pd.DataFrame:
        """각 경기를 '홈팀 득점' 행 1개 + '원정팀 득점' 행 1개로 풀어쓴다.
        (Dixon-Coles/Maher 스타일 포아송 회귀에서 표준적으로 쓰는 변환)

        추가 공변량은 "그 행이 어느 팀 관점인가"에 맞춰 home_{name}/away_{name}
        컬럼에서 값을 가져온다 (홈팀 행은 home_{name}, 원정팀 행은 away_{name}).
        """
        home_extra = {name: matches[f"home_{name}"].to_numpy() for name in self.feature_cols}
        away_extra = {name: matches[f"away_{name}"].to_numpy() for name in self.feature_cols}

        home = pd.DataFrame({
            "team": matches["HomeTeam"],
            "opponent": matches["AwayTeam"],
            "goals": matches["FTHG"],
            "is_home": 1,
            **home_extra,
        })
        away = pd.DataFrame({
            "team": matches["AwayTeam"],
            "opponent": matches["HomeTeam"],
            "goals": matches["FTAG"],
            "is_home": 0,
            **away_extra,
        })
        return pd.concat([home, away], ignore_index=True)

    def fit(self, matches: pd.DataFrame) -> PoissonFootballModel:
        """학습 시점까지의 경기 결과(matches)로 팀별 공격력/수비력을 추정한다.

        matches는 반드시 예측 시점 이전 경기만 포함해야 한다 (데이터 누수 방지).
        feature_cols에 해당하는 값이 없는 행(시즌 초반 표본 부족 등)은 이 학습에서
        제외한다 — 결측을 임의로 채우지 않고 표본이 있는 행만 사용한다.
        """
        self.teams = sorted(set(matches["HomeTeam"]) | set(matches["AwayTeam"]))
        long_df = self._to_long_format(matches)
        if self.feature_cols:
            long_df = long_df.dropna(subset=list(self.feature_cols)).reset_index(drop=True)
        long_df["team"] = pd.Categorical(long_df["team"], categories=self.teams)
        long_df["opponent"] = pd.Categorical(long_df["opponent"], categories=self.teams)

        formula = "goals ~ is_home + team + opponent"
        if self.feature_cols:
            formula += " + " + " + ".join(self.feature_cols)

        model = smf.glm(formula=formula, data=long_df, family=sm.families.Poisson())
        if self.l2_alpha > 0:
            # L1_wt=0.0 -> 순수 L2(ridge). Intercept 위치만 alpha=0으로 둬서
            # 절편은 규제하지 않는다 (나머지 팀 더미/is_home/추가 피처는 동일 강도로 규제).
            alpha = np.array([0.0 if name == "Intercept" else self.l2_alpha for name in model.exog_names])
            self.model = model.fit_regularized(alpha=alpha, L1_wt=0.0)
        else:
            self.model = model.fit()

        if self.use_dixon_coles:
            self.rho = self._fit_dc_rho(matches)
        return self

    def _predict_all(self, matches: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        """matches의 모든 경기에 대해 (한 경기당 하나씩) λ_home/λ_away 배열을 반환한다.
        _to_long_format은 dropna를 하지 않으므로 앞 N개가 홈, 뒤 N개가 원정 행으로
        matches와 순서가 그대로 대응된다.
        """
        n = len(matches)
        long_df = self._to_long_format(matches)
        long_df["team"] = pd.Categorical(long_df["team"], categories=self.teams)
        long_df["opponent"] = pd.Categorical(long_df["opponent"], categories=self.teams)
        predicted = self.model.predict(long_df).to_numpy()
        return predicted[:n], predicted[n:]

    def _fit_dc_rho(self, matches: pd.DataFrame) -> float:
        """이미 적합된 self.model의 λ를 고정한 채, Dixon-Coles ρ를 실제 결과에 대한
        로그우도 최대화로 추정한다. τ가 음수가 되는 ρ 후보는 큰 페널티를 줘서 배제한다
        (그런 ρ는 확률이 음수가 되는 무의미한 값이기 때문).
        """
        lambda_home, lambda_away = self._predict_all(matches)
        actual_home = matches["FTHG"].to_numpy()
        actual_away = matches["FTAG"].to_numpy()

        valid = np.isfinite(lambda_home) & np.isfinite(lambda_away)
        lambda_home, lambda_away = lambda_home[valid], lambda_away[valid]
        actual_home, actual_away = actual_home[valid], actual_away[valid]

        def neg_log_likelihood(rho: float) -> float:
            tau = _dc_tau_vectorized(actual_home, actual_away, lambda_home, lambda_away, rho)
            if np.any(tau <= 0):
                return 1e10
            base = poisson.pmf(actual_home, lambda_home) * poisson.pmf(actual_away, lambda_away)
            return float(-np.sum(np.log(tau * base + 1e-300)))

        result = minimize_scalar(neg_log_likelihood, bounds=_DC_RHO_BOUNDS, method="bounded")
        return float(result.x)

    def _expected_goals(
        self,
        home_team: str,
        away_team: str,
        home_features: dict[str, float] | None = None,
        away_features: dict[str, float] | None = None,
    ) -> tuple[float, float]:
        if self.model is None:
            raise RuntimeError("먼저 fit()을 호출하세요.")
        # pd.Categorical(categories=...)는 목록에 없는 값을 조용히 NaN으로 바꿔버려서
        # (예외를 던지지 않음) 학습 데이터에 없던 팀(승격팀 등)을 여기서 명시적으로 걸러낸다.
        unknown = {t for t in (home_team, away_team) if t not in self.teams}
        if unknown:
            raise ValueError(f"학습 데이터에 없는 팀: {unknown}")

        home_features = home_features or {}
        away_features = away_features or {}
        home_extra = {name: [home_features[name]] for name in self.feature_cols}
        away_extra = {name: [away_features[name]] for name in self.feature_cols}

        home_row = pd.DataFrame({
            "team": pd.Categorical([home_team], categories=self.teams),
            "opponent": pd.Categorical([away_team], categories=self.teams),
            "is_home": [1],
            **home_extra,
        })
        away_row = pd.DataFrame({
            "team": pd.Categorical([away_team], categories=self.teams),
            "opponent": pd.Categorical([home_team], categories=self.teams),
            "is_home": [0],
            **away_extra,
        })
        lambda_home = float(self.model.predict(home_row).iloc[0])
        lambda_away = float(self.model.predict(away_row).iloc[0])

        if not (np.isfinite(lambda_home) and np.isfinite(lambda_away)):
            # 표본이 아주 적은 팀(승격 직후 등)이 껴 있으면 학습 시 디자인 행렬이
            # 특이(singular)해져 그 팀의 GLM 계수가 극단값으로 튀고, 예측 기대득점이
            # 발산(overflow)할 수 있다. 조용히 NaN을 반환하는 대신 명시적으로 실패시켜
            # (predict_proba가 아니라 여기서) 호출 측이 학습 데이터 부족을 인지하게 한다.
            raise ValueError(
                f"기대 득점 계산이 발산함 (home={lambda_home}, away={lambda_away}) "
                f"— {home_team} 또는 {away_team}의 학습 표본이 부족할 수 있음"
            )
        return lambda_home, lambda_away

    def predict_proba(
        self,
        home_team: str,
        away_team: str,
        home_features: dict[str, float] | None = None,
        away_features: dict[str, float] | None = None,
        max_goals: int = 10,
    ) -> dict[str, float]:
        """홈승/무/원정승 확률을 반환한다.

        두 팀의 득점을 독립 포아송으로 가정하고, 가능한 모든 스코어(0~max_goals)에
        대한 결합확률을 더해 승/무/패 확률을 계산한다. use_dixon_coles=True면
        저득점 스코어 4개(0-0/1-0/0-1/1-1)에 τ 보정을 곱한 뒤 계산한다.
        """
        lambda_home, lambda_away = self._expected_goals(home_team, away_team, home_features, away_features)

        home_probs = poisson.pmf(np.arange(max_goals + 1), lambda_home)
        away_probs = poisson.pmf(np.arange(max_goals + 1), lambda_away)
        score_matrix = np.outer(home_probs, away_probs)

        if self.use_dixon_coles:
            # ρ는 학습 데이터 전체에 대해 고정된 값이라, 학습 때 보지 못한 극단적인
            # λ 조합(예: 아주 강팀 vs 아주 약팀)에서는 τ가 이론상 음수가 될 수 있다.
            # 0으로 클리핑해 "그 스코어는 사실상 안 나온다"로 처리한다(음수 확률 방지).
            for x, y in [(0, 0), (0, 1), (1, 0), (1, 1)]:
                score_matrix[x, y] *= max(0.0, _dc_tau(x, y, lambda_home, lambda_away, self.rho))

        p_home = float(np.tril(score_matrix, -1).sum())
        p_draw = float(np.trace(score_matrix))
        p_away = float(np.triu(score_matrix, 1).sum())

        # 잘림 오차(max_goals 초과) 보정을 위해 정규화
        total = p_home + p_draw + p_away
        return {"H": p_home / total, "D": p_draw / total, "A": p_away / total}
