"""
KRX 저평가 종목 탐색기
업종별 저평가 종목을 찾아 분석하는 Streamlit 앱
"""

import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(
    page_title="KRX 저평가 종목 탐색기",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

from data_loader import build_master_dataframe, load_price_history
from scoring import compute_undervaluation_scores, get_top_sectors, get_sector_summary
from visualization import (
    create_radar_chart,
    create_price_chart,
    create_sector_treemap_3d,
)


# ──────────────────────────────────────
# 커스텀 CSS
# ──────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Noto+Sans+KR:wght@400;500;700;900&display=swap');

    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem 2.5rem;
        border-radius: 16px;
        color: white;
        margin-bottom: 2rem;
        box-shadow: 0 10px 40px rgba(102,126,234,0.4);
        position: relative;
        overflow: hidden;
    }
    .main-header::before {
        content: '';
        position: absolute;
        top: -50%;
        right: -20%;
        width: 300px;
        height: 300px;
        background: rgba(255,255,255,0.08);
        border-radius: 50%;
    }
    .main-header h1 { margin: 0; font-size: 2.2rem; font-weight: 900; text-shadow: 0 2px 4px rgba(0,0,0,0.2); }
    .main-header p { margin: 0.5rem 0 0 0; opacity: 0.9; font-size: 1rem; }

    .sector-card {
        background: white;
        border: none;
        border-radius: 16px;
        padding: 1.5rem;
        text-align: center;
        transition: all 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275);
        box-shadow: 0 4px 15px rgba(0,0,0,0.08);
        cursor: pointer;
        min-height: 160px;
        position: relative;
        overflow: hidden;
    }
    .sector-card::before {
        content: '';
        position: absolute;
        top: 0; left: 0; right: 0;
        height: 4px;
        border-radius: 16px 16px 0 0;
    }
    .sector-card:hover {
        transform: translateY(-8px) scale(1.02);
        box-shadow: 0 20px 40px rgba(102,126,234,0.25);
    }
    .sector-card .rank {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        width: 28px; height: 28px;
        border-radius: 50%;
        background: linear-gradient(135deg, #667eea, #764ba2);
        color: white;
        font-size: 0.75rem;
        font-weight: 700;
        margin-bottom: 0.3rem;
    }
    .sector-card h3 { margin: 0.3rem 0 0.5rem 0; color: #1A1A2E; font-size: 1.05rem; font-weight: 700; }
    .sector-card .cap {
        background: linear-gradient(135deg, #667eea, #764ba2);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 1.5rem;
        font-weight: 900;
    }
    .sector-card .info { color: #888; font-size: 0.8rem; margin-top: 0.4rem; }

    .sector-card.c0::before { background: linear-gradient(90deg, #667eea, #764ba2); }
    .sector-card.c1::before { background: linear-gradient(90deg, #f093fb, #f5576c); }
    .sector-card.c2::before { background: linear-gradient(90deg, #4facfe, #00f2fe); }
    .sector-card.c3::before { background: linear-gradient(90deg, #43e97b, #38f9d7); }
    .sector-card.c4::before { background: linear-gradient(90deg, #fa709a, #fee140); }
    .sector-card.c5::before { background: linear-gradient(90deg, #a18cd1, #fbc2eb); }
    .sector-card.c6::before { background: linear-gradient(90deg, #fccb90, #d57eeb); }
    .sector-card.c7::before { background: linear-gradient(90deg, #e0c3fc, #8ec5fc); }
    .sector-card.c8::before { background: linear-gradient(90deg, #f5576c, #ff6f91); }
    .sector-card.c9::before { background: linear-gradient(90deg, #667eea, #00f2fe); }

    .metric-box {
        background: white;
        border: none;
        border-radius: 14px;
        padding: 1.2rem;
        text-align: center;
        box-shadow: 0 4px 15px rgba(0,0,0,0.06);
        transition: all 0.3s ease;
    }
    .metric-box:hover {
        transform: translateY(-3px);
        box-shadow: 0 8px 25px rgba(0,0,0,0.12);
    }
    .metric-box .label { color: #999; font-size: 0.75rem; font-weight: 500; text-transform: uppercase; letter-spacing: 0.5px; }
    .metric-box .value { color: #1A1A2E; font-size: 1.5rem; font-weight: 900; margin-top: 0.3rem; }

    .score-badge {
        display: inline-flex;
        align-items: center;
        gap: 4px;
        padding: 6px 16px;
        border-radius: 24px;
        font-weight: 700;
        font-size: 0.9rem;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
    }
    .score-high { background: linear-gradient(135deg, #43e97b, #38f9d7); color: #065f46; }
    .score-mid { background: linear-gradient(135deg, #fccb90, #fee140); color: #92400e; }
    .score-low { background: linear-gradient(135deg, #fecaca, #fca5a5); color: #991b1b; }

    .stock-row {
        background: white;
        border-radius: 12px;
        padding: 1rem 1.2rem;
        margin-bottom: 0.5rem;
        box-shadow: 0 2px 8px rgba(0,0,0,0.04);
        transition: all 0.3s ease;
        border-left: 4px solid transparent;
    }
    .stock-row:hover {
        box-shadow: 0 6px 20px rgba(102,126,234,0.15);
        border-left-color: #667eea;
        transform: translateX(4px);
    }

    .rank-circle {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        width: 36px; height: 36px;
        border-radius: 50%;
        font-weight: 900;
        font-size: 0.9rem;
        color: white;
    }
    .rank-1 { background: linear-gradient(135deg, #FFD700, #FFA500); box-shadow: 0 3px 10px rgba(255,215,0,0.4); }
    .rank-2 { background: linear-gradient(135deg, #C0C0C0, #A0A0A0); box-shadow: 0 3px 10px rgba(192,192,192,0.4); }
    .rank-3 { background: linear-gradient(135deg, #CD7F32, #B8860B); box-shadow: 0 3px 10px rgba(205,127,50,0.4); }
    .rank-other { background: linear-gradient(135deg, #667eea, #764ba2); box-shadow: 0 3px 10px rgba(102,126,234,0.3); }

    .naver-link {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        background: linear-gradient(135deg, #03C75A, #02B350);
        color: white !important;
        padding: 8px 18px;
        border-radius: 10px;
        text-decoration: none;
        font-weight: 600;
        font-size: 0.85rem;
        transition: all 0.3s ease;
        box-shadow: 0 3px 10px rgba(3,199,90,0.3);
    }
    .naver-link:hover { transform: translateY(-2px); box-shadow: 0 6px 20px rgba(3,199,90,0.4); color: white !important; }

    .analysis-box {
        background: linear-gradient(135deg, #f8faff, #eef2ff);
        border-left: 4px solid #667eea;
        padding: 1.2rem 1.5rem;
        border-radius: 0 12px 12px 0;
        margin: 0.5rem 0;
        box-shadow: 0 2px 8px rgba(102,126,234,0.08);
    }

    div[data-testid="stHorizontalBlock"] > div { min-width: 0; }

    .section-title {
        display: flex;
        align-items: center;
        gap: 10px;
        margin: 1.5rem 0 1rem 0;
    }
    .section-title .icon {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        width: 40px; height: 40px;
        border-radius: 12px;
        font-size: 1.2rem;
    }
    .section-title h3 { margin: 0; font-weight: 700; }
</style>
""", unsafe_allow_html=True)


# ──────────────────────────────────────
# 세션 상태 초기화
# ──────────────────────────────────────
if 'view' not in st.session_state:
    st.session_state.view = 'sectors'
if 'selected_sector' not in st.session_state:
    st.session_state.selected_sector = None
if 'selected_stock' not in st.session_state:
    st.session_state.selected_stock = None


def go_to_sectors():
    st.session_state.view = 'sectors'
    st.session_state.selected_sector = None
    st.session_state.selected_stock = None

def go_to_sector(sector):
    st.session_state.view = 'sector_detail'
    st.session_state.selected_sector = sector
    st.session_state.selected_stock = None

def go_to_stock(code):
    st.session_state.view = 'stock_detail'
    st.session_state.selected_stock = code


# ──────────────────────────────────────
# 사이드바
# ──────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ 설정")

    if st.button("🔄 데이터 새로고침", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

    market_filter = st.selectbox("📈 시장", ["전체", "KOSPI", "KOSDAQ"])

    min_mcap = st.slider(
        "💰 최소 시가총액 (억 원)",
        min_value=0, max_value=10000, value=100, step=100,
        help="이 금액 이상의 종목만 분석"
    )

    with st.expander("🎚️ 고급 설정: 가중치 조정"):
        w_ey = st.slider("수익률(1/PER) 가중치", 0.0, 1.0, 0.30, 0.05)
        w_by = st.slider("장부가(1/PBR) 가중치", 0.0, 1.0, 0.25, 0.05)
        w_eps = st.slider("EPS 가중치", 0.0, 1.0, 0.20, 0.05)
        w_div = st.slider("배당률 가중치", 0.0, 1.0, 0.10, 0.05)
        w_mcap = st.slider("시총역수 가중치", 0.0, 1.0, 0.15, 0.05)

    st.markdown("---")
    st.markdown(
        '<p style="color:#aaa;font-size:0.75rem;text-align:center;">'
        '📊 KRX 데이터 기반<br>네이버증권 + FinanceDataReader'
        '</p>',
        unsafe_allow_html=True,
    )


# ──────────────────────────────────────
# 데이터 로딩
# ──────────────────────────────────────
@st.cache_data(ttl=3600, show_spinner="전체 데이터를 분석하고 있습니다...")
def get_processed_data(market_filter_val, min_mcap_val):
    from datetime import datetime
    result = build_master_dataframe()
    if isinstance(result, tuple):
        df, debug_log = result
    else:
        df, debug_log = result, []

    if df is None or (isinstance(df, pd.DataFrame) and df.empty):
        return pd.DataFrame(), pd.DataFrame(), [], "", debug_log

    data_date = df.attrs.get('date', datetime.now().strftime('%Y-%m-%d'))

    # 시장 필터
    if market_filter_val != "전체":
        market_col = None
        for c in ['Market', 'market', '시장']:
            if c in df.columns:
                market_col = c
                break
        if market_col:
            df = df[df[market_col].str.upper().str.contains(market_filter_val)]

    # 시총 필터
    df = df[df['시가총액'] >= min_mcap_val * 1e8]

    # 점수 계산
    df = compute_undervaluation_scores(df)

    summary = get_sector_summary(df)
    top_sectors = get_top_sectors(df, 10)

    return df, summary, top_sectors, data_date, debug_log


with st.spinner("데이터를 불러오고 있습니다..."):
    master_df, sector_summary, top_sectors, data_date, debug_log = get_processed_data(market_filter, min_mcap)

# 사이드바에 디버그 정보
if debug_log:
    with st.sidebar:
        with st.expander("🔍 데이터 로딩 로그", expanded=False):
            for msg in debug_log:
                st.caption(msg)

if master_df.empty:
    st.error("데이터를 불러올 수 없습니다. 잠시 후 다시 시도해주세요.")
    st.stop()

if not data_date:
    data_date = 'N/A'

# 통계
total_stocks = len(master_df)
scored_stocks = len(master_df[master_df.get('저평가점수', pd.Series([0])) > 0]) if '저평가점수' in master_df.columns else 0


# ──────────────────────────────────────
# 헤더
# ──────────────────────────────────────
st.markdown(f"""
<div class="main-header">
    <h1>🔍 KRX 저평가 종목 탐색기</h1>
    <p>업종별 펀더멘탈 분석 기반 저평가 종목 발굴 · 기준일: {data_date} · 총 {total_stocks:,}개 종목 분석 · {scored_stocks:,}개 스코어링 완료</p>
</div>
""", unsafe_allow_html=True)


# ══════════════════════════════════════
# VIEW: 업종 리스트
# ══════════════════════════════════════
if st.session_state.view == 'sectors':

    # 업종별 시가총액 바 차트
    top_summary = sector_summary[sector_summary['업종'].isin(top_sectors)]
    if not top_summary.empty:
        fig = create_sector_treemap_3d(top_summary)
        st.plotly_chart(fig, use_container_width=True, key="sector_treemap")

    st.markdown("""
    <div class="section-title">
        <div class="icon" style="background:linear-gradient(135deg,#667eea,#764ba2);color:white;">🏆</div>
        <h3>시가총액 상위 10대 업종</h3>
    </div>
    <p style="color:#888;margin-bottom:1rem;">업종을 클릭하면 해당 업종의 저평가 종목을 확인할 수 있습니다</p>
    """, unsafe_allow_html=True)

    # 업종 카드 그리드 (2행 x 5열)
    for row in range(2):
        cols = st.columns(5)
        for col_idx in range(5):
            idx = row * 5 + col_idx
            if idx >= len(top_sectors):
                break
            sector = top_sectors[idx]
            s_info = top_summary[top_summary['업종'] == sector].iloc[0] if len(top_summary[top_summary['업종'] == sector]) > 0 else None

            with cols[col_idx]:
                if s_info is not None:
                    cap_str = f"{s_info['총시가총액_조']:.0f}조"
                    cnt = int(s_info['종목수'])
                    per_str = f"PER {s_info['평균PER']:.1f}" if pd.notna(s_info['평균PER']) else ""

                    st.markdown(f"""
                    <div class="sector-card c{idx}">
                        <div class="rank">{idx+1}</div>
                        <h3>{sector}</h3>
                        <div class="cap">{cap_str}</div>
                        <div class="info">{cnt}개 종목 {' · ' + per_str if per_str else ''}</div>
                    </div>
                    """, unsafe_allow_html=True)

                    if st.button(f"📊 {sector} 분석", key=f"sector_{idx}", use_container_width=True):
                        go_to_sector(sector)
                        st.rerun()


# ══════════════════════════════════════
# VIEW: 업종 상세 (종목 순위)
# ══════════════════════════════════════
elif st.session_state.view == 'sector_detail':
    sector = st.session_state.selected_sector

    col1, col2 = st.columns([1, 6])
    with col1:
        if st.button("← 업종 목록", use_container_width=True):
            go_to_sectors()
            st.rerun()
    with col2:
        st.markdown(f"## 📂 {sector}")

    # 업종 요약 정보
    s_info = sector_summary[sector_summary['업종'] == sector]
    sector_df = master_df[master_df['업종'] == sector].copy()

    if not s_info.empty:
        s = s_info.iloc[0]
        m1, m2, m3, m4, m5 = st.columns(5)
        with m1:
            st.markdown(f'<div class="metric-box"><div class="label">💎 총 시가총액</div><div class="value">{s["총시가총액_조"]:.1f}조</div></div>', unsafe_allow_html=True)
        with m2:
            st.markdown(f'<div class="metric-box"><div class="label">📋 종목 수</div><div class="value">{int(s["종목수"])}개</div></div>', unsafe_allow_html=True)
        with m3:
            per_v = f'{s["평균PER"]:.1f}' if pd.notna(s["평균PER"]) else "N/A"
            st.markdown(f'<div class="metric-box"><div class="label">📊 평균 PER</div><div class="value">{per_v}</div></div>', unsafe_allow_html=True)
        with m4:
            pbr_v = f'{s["평균PBR"]:.2f}' if pd.notna(s["평균PBR"]) else "N/A"
            st.markdown(f'<div class="metric-box"><div class="label">📈 평균 PBR</div><div class="value">{pbr_v}</div></div>', unsafe_allow_html=True)
        with m5:
            div_v = f'{s["평균DIV"]:.2f}%' if pd.notna(s["평균DIV"]) else "N/A"
            st.markdown(f'<div class="metric-box"><div class="label">💰 평균 배당률</div><div class="value">{div_v}</div></div>', unsafe_allow_html=True)

    st.markdown("")

    # 종목 순위 테이블
    st.markdown("""
    <div class="section-title">
        <div class="icon" style="background:linear-gradient(135deg,#43e97b,#38f9d7);color:#065f46;">🏅</div>
        <h3>저평가 종목 순위</h3>
    </div>
    <p style="color:#888;">저평가 점수가 높을수록 업종 내 실적 대비 시가총액이 낮은 종목입니다</p>
    """, unsafe_allow_html=True)

    sort_col = '저평가점수' if '저평가점수' in sector_df.columns else '시가총액'
    ranked = sector_df.sort_values(sort_col, ascending=False).reset_index(drop=True)

    for i, (_, row) in enumerate(ranked.iterrows()):
        score = row.get('저평가점수', 0) if '저평가점수' in ranked.columns else 0
        rank = i + 1

        if score >= 70:
            badge_class = 'score-high'
            badge_icon = '🔥'
        elif score >= 40:
            badge_class = 'score-mid'
            badge_icon = '⚡'
        else:
            badge_class = 'score-low'
            badge_icon = '💤'

        if rank == 1:
            rank_class = 'rank-1'
        elif rank == 2:
            rank_class = 'rank-2'
        elif rank == 3:
            rank_class = 'rank-3'
        else:
            rank_class = 'rank-other'

        mcap_억 = row['시가총액'] / 1e8

        with st.container():
            c1, c2, c3, c4, c5, c6, c7 = st.columns([0.5, 2.2, 1.5, 1.2, 1, 1, 1])

            with c1:
                st.markdown(f'<div class="rank-circle {rank_class}">{rank}</div>', unsafe_allow_html=True)
            with c2:
                if st.button(f"📌 {row['Name']}", key=f"stock_{row['Code']}"):
                    go_to_stock(row['Code'])
                    st.rerun()
            with c3:
                st.markdown(f'<span class="score-badge {badge_class}">{badge_icon} {score:.1f}점</span>', unsafe_allow_html=True)
            with c4:
                st.caption(f"💰 {mcap_억:,.0f}억")
            with c5:
                per_v = f"{row['PER']:.1f}" if row['PER'] > 0 else "N/A"
                st.caption(f"PER {per_v}")
            with c6:
                pbr_v = f"{row['PBR']:.2f}" if row['PBR'] > 0 else "N/A"
                st.caption(f"PBR {pbr_v}")
            with c7:
                naver_url = f"https://finance.naver.com/item/main.naver?code={row['Code']}"
                st.markdown(f'<a href="{naver_url}" target="_blank" class="naver-link">N</a>', unsafe_allow_html=True)

        if i < len(ranked) - 1 and i < 49:
            st.divider()

        if i >= 49:
            break


# ══════════════════════════════════════
# VIEW: 종목 상세
# ══════════════════════════════════════
elif st.session_state.view == 'stock_detail':
    code = st.session_state.selected_stock
    stock_data = master_df[master_df['Code'] == code]

    if stock_data.empty:
        st.error("종목 정보를 찾을 수 없습니다.")
        if st.button("← 돌아가기"):
            go_to_sectors()
            st.rerun()
        st.stop()

    stock = stock_data.iloc[0]
    sector = stock['업종']

    # 네비게이션
    col1, col2, col3 = st.columns([1, 1, 5])
    with col1:
        if st.button("← 업종 목록"):
            go_to_sectors()
            st.rerun()
    with col2:
        if st.button(f"← {sector}"):
            go_to_sector(sector)
            st.rerun()

    # 종목 헤더
    st.markdown(f"## 🏢 {stock['Name']} ({code})")

    naver_url = f"https://finance.naver.com/item/main.naver?code={code}"
    st.markdown(f'<a href="{naver_url}" target="_blank" class="naver-link">🔗 네이버 증권에서 보기</a>', unsafe_allow_html=True)
    st.markdown("")

    # 핵심 지표
    score = stock.get('저평가점수', 0) if '저평가점수' in master_df.columns else 0
    if score >= 70:
        badge = '<span class="score-badge score-high">🔥 강력 저평가</span>'
    elif score >= 40:
        badge = '<span class="score-badge score-mid">⚡ 적정~저평가</span>'
    else:
        badge = '<span class="score-badge score-low">💤 고평가 또는 데이터 부족</span>'

    st.markdown(f"### 저평가 점수: {score:.1f}점 {badge}", unsafe_allow_html=True)

    mcap_억 = stock['시가총액'] / 1e8
    cols = st.columns(6)
    metric_data = [
        ("💎 시가총액", f"{mcap_억:,.0f}억"),
        ("📊 PER", f"{stock['PER']:.1f}" if stock['PER'] > 0 else "N/A"),
        ("📈 PBR", f"{stock['PBR']:.2f}" if stock['PBR'] > 0 else "N/A"),
        ("💵 EPS", f"{stock['EPS']:,.0f}" if stock['EPS'] > 0 else "N/A"),
        ("🏦 BPS", f"{stock.get('BPS', 0):,.0f}"),
        ("💰 배당률", f"{stock.get('DIV', 0):.2f}%"),
    ]
    for col, (label, value) in zip(cols, metric_data):
        with col:
            st.markdown(f'<div class="metric-box"><div class="label">{label}</div><div class="value">{value}</div></div>', unsafe_allow_html=True)

    st.markdown("")

    # 분석 탭
    tab1, tab2, tab3 = st.tabs(["📈 주가 차트", "🎯 업종 비교 레이더", "📋 종합 분석"])

    with tab1:
        price_df = load_price_history(code)
        if not price_df.empty:
            fig = create_price_chart(price_df, stock['Name'])
            st.plotly_chart(fig, use_container_width=True, key="stock_price_chart")

            close_col = 'Close' if 'Close' in price_df.columns else '종가'
            if close_col in price_df.columns:
                pc1, pc2, pc3, pc4 = st.columns(4)
                current = price_df[close_col].iloc[-1]
                high_52 = price_df[close_col].max()
                low_52 = price_df[close_col].min()
                avg_52 = price_df[close_col].mean()

                with pc1:
                    st.metric("현재가", f"{current:,.0f}원")
                with pc2:
                    st.metric("52주 최고", f"{high_52:,.0f}원", f"{((current/high_52)-1)*100:.1f}%")
                with pc3:
                    st.metric("52주 최저", f"{low_52:,.0f}원", f"{((current/low_52)-1)*100:.1f}%")
                with pc4:
                    st.metric("52주 평균", f"{avg_52:,.0f}원", f"{((current/avg_52)-1)*100:.1f}%")
        else:
            st.info("주가 데이터를 불러올 수 없습니다.")

    with tab2:
        s_info = sector_summary[sector_summary['업종'] == sector]
        if not s_info.empty:
            sector_avg = s_info.iloc[0].to_dict()
            sector_df_temp = master_df[(master_df['업종'] == sector) & (master_df['BPS'] > 0)] if 'BPS' in master_df.columns else pd.DataFrame()
            sector_avg['평균BPS'] = sector_df_temp['BPS'].mean() if not sector_df_temp.empty else 1
            fig = create_radar_chart(stock, sector_avg, sector)
            st.plotly_chart(fig, use_container_width=True, key="stock_radar_chart")

            st.markdown("""
            <div class="analysis-box">
            <b>🎯 레이더 차트 해석</b><br>
            - 100 = 업종 평균 수준<br>
            - 100 이상 = 업종 평균보다 우수<br>
            - PER/PBR 역수가 높을수록 → 같은 이익 대비 주가가 저렴<br>
            - EPS/BPS가 높을수록 → 기업의 실적/자산가치가 우수
            </div>
            """, unsafe_allow_html=True)

    with tab3:
        st.markdown("""
        <div class="section-title">
            <div class="icon" style="background:linear-gradient(135deg,#667eea,#764ba2);color:white;">📋</div>
            <h3>종합 분석 리포트</h3>
        </div>
        """, unsafe_allow_html=True)

        sector_peers = master_df[(master_df['업종'] == sector) & (master_df['PER'] > 0)]
        if not sector_peers.empty:
            avg_per = sector_peers['PER'].mean()
            avg_pbr = sector_peers['PBR'].mean()
            avg_div = sector_peers['DIV'].mean()
            avg_mcap = sector_peers['시가총액'].mean() / 1e8
            total_in_sector = len(sector_peers)
            rank = stock.get('업종순위', 'N/A') if '업종순위' in master_df.columns else 'N/A'

            st.markdown(f"""
            <div class="analysis-box">
            <b>📍 업종 내 위치</b><br>
            {sector} 업종 내 {total_in_sector}개 종목 중 <b>{rank}위</b> (저평가 순)
            </div>
            """, unsafe_allow_html=True)

            if stock['PER'] > 0:
                per_diff = ((stock['PER'] / avg_per) - 1) * 100
                per_status = "할인" if per_diff < 0 else "프리미엄"
                emoji = "✅" if per_diff < 0 else "⚠️"
                st.markdown(f"""
                <div class="analysis-box">
                <b>{emoji} PER 분석</b><br>
                종목 PER: {stock['PER']:.1f} | 업종 평균: {avg_per:.1f}<br>
                업종 평균 대비 <b>{abs(per_diff):.1f}% {per_status}</b> 거래 중
                {'→ 이익 대비 주가가 저렴합니다' if per_diff < 0 else '→ 이익 대비 주가가 비쌉니다'}
                </div>
                """, unsafe_allow_html=True)

            if stock['PBR'] > 0:
                pbr_diff = ((stock['PBR'] / avg_pbr) - 1) * 100
                pbr_status = "할인" if pbr_diff < 0 else "프리미엄"
                emoji = "✅" if pbr_diff < 0 else "⚠️"
                st.markdown(f"""
                <div class="analysis-box">
                <b>{emoji} PBR 분석</b><br>
                종목 PBR: {stock['PBR']:.2f} | 업종 평균: {avg_pbr:.2f}<br>
                업종 평균 대비 <b>{abs(pbr_diff):.1f}% {pbr_status}</b> 거래 중
                {'→ 자산가치 대비 주가가 저렴합니다' if pbr_diff < 0 else '→ 자산가치 대비 주가가 비쌉니다'}
                </div>
                """, unsafe_allow_html=True)

            div_val = stock.get('DIV', 0)
            if div_val > 0:
                div_diff = div_val - avg_div
                emoji = "✅" if div_diff > 0 else "💤"
                st.markdown(f"""
                <div class="analysis-box">
                <b>{emoji} 배당 분석</b><br>
                종목 배당률: {div_val:.2f}% | 업종 평균: {avg_div:.2f}%<br>
                업종 평균 대비 <b>{abs(div_diff):.2f}%p {'높은' if div_diff > 0 else '낮은'}</b> 배당률
                </div>
                """, unsafe_allow_html=True)

            mcap_ratio = (mcap_억 / avg_mcap) * 100 if avg_mcap > 0 else 100
            emoji = "✅" if mcap_ratio < 80 else "📊"
            st.markdown(f"""
            <div class="analysis-box">
            <b>{emoji} 시가총액 분석</b><br>
            종목 시총: {mcap_억:,.0f}억 | 업종 평균: {avg_mcap:,.0f}억<br>
            업종 평균 시총의 <b>{mcap_ratio:.0f}%</b> 수준
            {'→ 실적 대비 시총이 낮아 저평가 가능성이 있습니다' if mcap_ratio < 80 else '→ 업종 평균 수준 또는 그 이상입니다'}
            </div>
            """, unsafe_allow_html=True)

            # 종합 시그널
            signals = []
            if stock['PER'] > 0 and stock['PER'] < avg_per:
                signals.append("PER 업종평균 하회 (긍정)")
            if stock['PBR'] > 0 and stock['PBR'] < avg_pbr:
                signals.append("PBR 업종평균 하회 (긍정)")
            if div_val > avg_div:
                signals.append("배당률 업종평균 상회 (긍정)")
            if stock['EPS'] > 0 and stock['EPS'] > sector_peers['EPS'].median():
                signals.append("EPS 업종 중위수 상회 (긍정)")
            if mcap_ratio < 80:
                signals.append("시총 업종평균 하회 (저평가 신호)")

            if signals:
                signal_html = "<br>".join([f"✅ {s}" for s in signals])
                st.markdown(f"""
                <div class="analysis-box">
                <b>🎯 저평가 시그널 요약</b><br>
                {signal_html}
                </div>
                """, unsafe_allow_html=True)

            st.markdown("""
            <div style="background:linear-gradient(135deg,#FFF3E0,#FFE0B2);padding:1rem 1.5rem;border-radius:12px;margin-top:1rem;font-size:0.85rem;color:#E65100;box-shadow:0 2px 8px rgba(230,81,0,0.1);">
            ⚠️ <b>투자 유의사항</b>: 본 분석은 정량적 지표에 기반한 참고 자료이며, 투자 권유가 아닙니다.
            실제 투자 시 기업의 질적 요소, 산업 전망, 재무건전성 등을 종합적으로 검토하시기 바랍니다.
            </div>
            """, unsafe_allow_html=True)
