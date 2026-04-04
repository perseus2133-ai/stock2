"""
저평가 점수 산출 모듈
업종 내 Z-score 기반 복합 점수 계산
"""

import pandas as pd
import numpy as np


def compute_undervaluation_scores(df: pd.DataFrame) -> pd.DataFrame:
    """
    업종 내 저평가 점수 계산
    높은 점수 = 더 저평가 (좋은 실적 대비 낮은 시총)
    """
    df = df.copy()

    # 기본 필터: PER, PBR > 0 (흑자 기업만)
    valid = df[
        (df['PER'] > 0) & (df['PBR'] > 0) & (df['EPS'] > 0)
    ].copy()

    if valid.empty:
        df['저평가점수'] = 0
        df['업종순위'] = 999
        return df

    # 수익성 지표 계산
    valid['earnings_yield'] = 1 / valid['PER']  # 높을수록 저평가
    valid['book_yield'] = 1 / valid['PBR']       # 높을수록 저평가
    valid['log_mcap'] = np.log10(valid['시가총액'].clip(lower=1))

    # 업종별 Z-score 계산
    score_cols = ['earnings_yield', 'book_yield', 'EPS', 'DIV']
    weights = {'earnings_yield': 0.30, 'book_yield': 0.25, 'EPS': 0.20, 'DIV': 0.10}
    # 시총 역수 가중 (같은 실적이면 시총 작을수록 저평가)
    mcap_weight = 0.15

    def sector_zscore(group):
        if len(group) < 3:
            group['저평가점수'] = 50
            group['업종순위'] = range(1, len(group) + 1)
            return group

        composite = pd.Series(0.0, index=group.index)

        for col in score_cols:
            if col in group.columns:
                mean = group[col].mean()
                std = group[col].std()
                if std > 0:
                    z = (group[col] - mean) / std
                else:
                    z = 0
                composite += weights.get(col, 0.15) * z

        # 시총 역Z-score (시총이 낮을수록 점수 높음 → 같은 실적이면 저평가)
        mcap_mean = group['log_mcap'].mean()
        mcap_std = group['log_mcap'].std()
        if mcap_std > 0:
            mcap_z = -(group['log_mcap'] - mcap_mean) / mcap_std
            composite += mcap_weight * mcap_z

        # 0~100 스케일 변환
        min_s, max_s = composite.min(), composite.max()
        if max_s > min_s:
            normalized = ((composite - min_s) / (max_s - min_s)) * 100
        else:
            normalized = 50

        group['저평가점수'] = normalized.round(1)
        group['업종순위'] = normalized.rank(ascending=False, method='min').astype(int)

        return group

    import warnings
    # pandas 버전별 호환 처리
    pd_version = tuple(int(x) for x in pd.__version__.split('.')[:2])
    try:
        if pd_version >= (2, 2):
            scored = valid.groupby('업종', group_keys=False).apply(
                sector_zscore, include_groups=False
            )
            scored['업종'] = valid['업종']
        else:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                scored = valid.groupby('업종', group_keys=False).apply(sector_zscore)
    except TypeError:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            scored = valid.groupby('업종', group_keys=False).apply(sector_zscore)

    # 점수 컬럼만 원본에 병합
    score_cols_result = [c for c in ['저평가점수', '업종순위'] if c in scored.columns]
    if score_cols_result:
        df = df.merge(
            scored[score_cols_result],
            left_index=True, right_index=True, how='left'
        )
    else:
        df['저평가점수'] = 0
        df['업종순위'] = 999

    df['저평가점수'] = df['저평가점수'].fillna(0)
    df['업종순위'] = df['업종순위'].fillna(999).astype(int)

    return df


def get_top_sectors(df: pd.DataFrame, n: int = 10) -> list:
    """시총 상위 N개 업종 반환"""
    sector_caps = df.groupby('업종')['시가총액'].sum().sort_values(ascending=False)
    # '기타' 제외
    sector_caps = sector_caps[sector_caps.index != '기타']
    return sector_caps.head(n).index.tolist()


def get_sector_summary(df: pd.DataFrame) -> pd.DataFrame:
    """업종별 요약 통계"""
    summary = df.groupby('업종').agg(
        총시가총액=('시가총액', 'sum'),
        종목수=('Code', 'count'),
        평균PER=('PER', lambda x: x[x > 0].mean()),
        평균PBR=('PBR', lambda x: x[x > 0].mean()),
        평균DIV=('DIV', 'mean'),
        평균EPS=('EPS', lambda x: x[x > 0].mean()),
    ).reset_index()

    summary = summary.sort_values('총시가총액', ascending=False)
    summary['총시가총액_조'] = (summary['총시가총액'] / 1e12).round(1)

    return summary
