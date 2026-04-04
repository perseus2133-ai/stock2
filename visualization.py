"""
Plotly 3D 시각화 모듈
"""

import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np


def create_sector_bubble_3d(summary_df: pd.DataFrame) -> go.Figure:
    """업종별 3D 버블 차트 (PER x PBR x 시총순위, 크기=시총)"""
    df = summary_df.dropna(subset=['평균PER', '평균PBR']).copy()
    df = df[df['평균PER'] > 0]
    df = df[df['총시가총액'] > 0]

    if df.empty:
        return go.Figure()

    # DIV 없을 때 대체축: 시총 로그
    if '평균DIV' not in df.columns or df['평균DIV'].sum() == 0:
        df['평균DIV'] = np.log10(df['총시가총액'].clip(lower=1))

    # 버블 크기 정규화
    size_ref = df['총시가총액'].max() / 60

    fig = go.Figure(data=[go.Scatter3d(
        x=df['평균PER'],
        y=df['평균PBR'],
        z=df['평균DIV'],
        mode='markers+text',
        marker=dict(
            size=np.clip(df['총시가총액'] / size_ref, 8, 50),
            color=df['총시가총액'],
            colorscale='Viridis',
            opacity=0.85,
            line=dict(width=1, color='white'),
            colorbar=dict(title="시가총액", thickness=15),
        ),
        text=df['업종'],
        textposition='top center',
        textfont=dict(size=10, color='#333'),
        hovertemplate=(
            '<b>%{text}</b><br>'
            'PER: %{x:.1f}<br>'
            'PBR: %{y:.2f}<br>'
            'DIV: %{z:.2f}%<br>'
            '<extra></extra>'
        ),
    )])

    fig.update_layout(
        scene=dict(
            xaxis_title='평균 PER',
            yaxis_title='평균 PBR',
            zaxis_title='평균 배당률 (%)',
            bgcolor='#F8F9FA',
            xaxis=dict(backgroundcolor='#EEF2F7', gridcolor='#D0D5DD'),
            yaxis=dict(backgroundcolor='#EEF2F7', gridcolor='#D0D5DD'),
            zaxis=dict(backgroundcolor='#EEF2F7', gridcolor='#D0D5DD'),
        ),
        title=dict(text='업종별 밸류에이션 3D 맵', font=dict(size=18)),
        height=550,
        margin=dict(l=0, r=0, t=50, b=0),
        paper_bgcolor='white',
    )

    return fig


def create_sector_stocks_3d(sector_df: pd.DataFrame, sector_name: str) -> go.Figure:
    """업종 내 종목 3D 산점도 (PER x PBR x log시총, 색상=저평가점수)"""
    df = sector_df.copy()
    df = df[(df['PER'] > 0) & (df['PBR'] > 0) & (df['시가총액'] > 0)]

    if df.empty:
        return go.Figure()

    df['log_mcap'] = np.log10(df['시가총액'])
    df['시총_억'] = (df['시가총액'] / 1e8).round(0)
    if '저평가점수' not in df.columns:
        df['저평가점수'] = 0

    fig = go.Figure(data=[go.Scatter3d(
        x=df['PER'],
        y=df['PBR'],
        z=df['log_mcap'],
        mode='markers',
        marker=dict(
            size=7,
            color=df['저평가점수'],
            colorscale='RdYlGn',
            cmin=0,
            cmax=100,
            opacity=0.9,
            line=dict(width=0.5, color='white'),
            colorbar=dict(title="저평가점수", thickness=15),
        ),
        text=df['Name'],
        customdata=np.stack([df['시총_억'], df['저평가점수'], df['EPS']], axis=-1),
        hovertemplate=(
            '<b>%{text}</b><br>'
            'PER: %{x:.1f}<br>'
            'PBR: %{y:.2f}<br>'
            '시총: %{customdata[0]:,.0f}억<br>'
            '저평가점수: %{customdata[1]:.1f}<br>'
            'EPS: %{customdata[2]:,.0f}<br>'
            '<extra></extra>'
        ),
    )])

    fig.update_layout(
        scene=dict(
            xaxis_title='PER',
            yaxis_title='PBR',
            zaxis_title='Log(시가총액)',
            bgcolor='#F8F9FA',
            xaxis=dict(backgroundcolor='#EEF2F7', gridcolor='#D0D5DD'),
            yaxis=dict(backgroundcolor='#EEF2F7', gridcolor='#D0D5DD'),
            zaxis=dict(backgroundcolor='#EEF2F7', gridcolor='#D0D5DD'),
        ),
        title=dict(text=f'{sector_name} 종목 밸류에이션 3D 맵', font=dict(size=16)),
        height=550,
        margin=dict(l=0, r=0, t=50, b=0),
        paper_bgcolor='white',
    )

    return fig


def create_stock_vs_peers_3d(stock_row: pd.Series, peers_df: pd.DataFrame) -> go.Figure:
    """선택 종목 vs 동종 업종 3D 비교"""
    peers = peers_df[(peers_df['PER'] > 0) & (peers_df['PBR'] > 0)].copy()
    peers['log_mcap'] = np.log10(peers['시가총액'].clip(lower=1))

    fig = go.Figure()

    # 동종 업종 종목
    fig.add_trace(go.Scatter3d(
        x=peers['PER'],
        y=peers['PBR'],
        z=peers['log_mcap'],
        mode='markers',
        marker=dict(size=5, color='#B0BEC5', opacity=0.5, line=dict(width=0.5, color='white')),
        text=peers['Name'],
        hovertemplate='<b>%{text}</b><br>PER: %{x:.1f}<br>PBR: %{y:.2f}<extra></extra>',
        name='동종 업종',
    ))

    # 선택 종목 강조
    if stock_row['PER'] > 0 and stock_row['PBR'] > 0:
        fig.add_trace(go.Scatter3d(
            x=[stock_row['PER']],
            y=[stock_row['PBR']],
            z=[np.log10(max(stock_row['시가총액'], 1))],
            mode='markers+text',
            marker=dict(size=14, color='#FF1744', opacity=1.0,
                       line=dict(width=2, color='white'),
                       symbol='diamond'),
            text=[stock_row['Name']],
            textposition='top center',
            textfont=dict(size=12, color='#FF1744'),
            hovertemplate=f'<b>{stock_row["Name"]}</b><br>PER: {stock_row["PER"]:.1f}<br>PBR: {stock_row["PBR"]:.2f}<extra></extra>',
            name=stock_row['Name'],
        ))

    fig.update_layout(
        scene=dict(
            xaxis_title='PER',
            yaxis_title='PBR',
            zaxis_title='Log(시가총액)',
            bgcolor='#F8F9FA',
            xaxis=dict(backgroundcolor='#EEF2F7', gridcolor='#D0D5DD'),
            yaxis=dict(backgroundcolor='#EEF2F7', gridcolor='#D0D5DD'),
            zaxis=dict(backgroundcolor='#EEF2F7', gridcolor='#D0D5DD'),
        ),
        title=dict(text=f'{stock_row["Name"]} vs 동종업종 비교', font=dict(size=16)),
        height=500,
        margin=dict(l=0, r=0, t=50, b=0),
        paper_bgcolor='white',
        showlegend=True,
    )

    return fig


def create_radar_chart(stock_row: pd.Series, sector_avg: dict, sector_name: str) -> go.Figure:
    """종목 vs 업종 평균 레이더 차트"""
    categories = ['PER(역수)', 'PBR(역수)', 'EPS', '배당률', 'BPS']

    # 정규화 (업종 평균 대비 비율)
    stock_vals = []
    avg_vals = []

    def safe_inv(v):
        return 1 / v if v and v > 0 else 0

    metrics = [
        (safe_inv(stock_row.get('PER', 0)), safe_inv(sector_avg.get('평균PER', 1))),
        (safe_inv(stock_row.get('PBR', 0)), safe_inv(sector_avg.get('평균PBR', 1))),
        (stock_row.get('EPS', 0), sector_avg.get('평균EPS', 1)),
        (stock_row.get('DIV', 0), sector_avg.get('평균DIV', 1)),
        (stock_row.get('BPS', 0), sector_avg.get('평균BPS', 1)),
    ]

    for stock_v, avg_v in metrics:
        if avg_v and avg_v > 0:
            stock_vals.append(min((stock_v / avg_v) * 100, 200))
        else:
            stock_vals.append(50)
        avg_vals.append(100)

    fig = go.Figure()

    fig.add_trace(go.Scatterpolar(
        r=avg_vals + [avg_vals[0]],
        theta=categories + [categories[0]],
        fill='toself',
        name=f'{sector_name} 평균',
        line=dict(color='#90CAF9', width=2),
        fillcolor='rgba(144,202,249,0.2)',
    ))

    fig.add_trace(go.Scatterpolar(
        r=stock_vals + [stock_vals[0]],
        theta=categories + [categories[0]],
        fill='toself',
        name=stock_row.get('Name', '종목'),
        line=dict(color='#FF5722', width=2),
        fillcolor='rgba(255,87,34,0.2)',
    ))

    fig.update_layout(
        polar=dict(
            bgcolor='#F8F9FA',
            radialaxis=dict(visible=True, range=[0, 200], gridcolor='#D0D5DD'),
            angularaxis=dict(gridcolor='#D0D5DD'),
        ),
        title=dict(text='업종 평균 대비 분석', font=dict(size=16)),
        height=400,
        margin=dict(l=60, r=60, t=60, b=40),
        paper_bgcolor='white',
        showlegend=True,
        legend=dict(x=0.85, y=1.1),
    )

    return fig


def create_price_chart(price_df: pd.DataFrame, name: str) -> go.Figure:
    """주가 추이 차트"""
    if price_df.empty:
        return go.Figure()

    close_col = 'Close' if 'Close' in price_df.columns else '종가'
    if close_col not in price_df.columns:
        return go.Figure()

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=price_df.index,
        y=price_df[close_col],
        mode='lines',
        line=dict(color='#1E88E5', width=2),
        fill='tozeroy',
        fillcolor='rgba(30,136,229,0.1)',
        name='종가',
    ))

    # 이동평균선
    for period, color in [(20, '#FFA726'), (60, '#EF5350'), (120, '#AB47BC')]:
        if len(price_df) >= period:
            ma = price_df[close_col].rolling(window=period).mean()
            fig.add_trace(go.Scatter(
                x=price_df.index,
                y=ma,
                mode='lines',
                line=dict(color=color, width=1, dash='dot'),
                name=f'MA{period}',
            ))

    fig.update_layout(
        title=dict(text=f'{name} 주가 추이 (1년)', font=dict(size=16)),
        xaxis_title='날짜',
        yaxis_title='주가 (원)',
        height=400,
        margin=dict(l=50, r=20, t=50, b=40),
        paper_bgcolor='white',
        plot_bgcolor='#FAFAFA',
        xaxis=dict(gridcolor='#E0E0E0'),
        yaxis=dict(gridcolor='#E0E0E0'),
        hovermode='x unified',
    )

    return fig


def create_valuation_3d_surface(sector_df: pd.DataFrame, sector_name: str) -> go.Figure:
    """업종 밸류에이션 3D 서피스 (PER x PBR → 저평가점수)"""
    df = sector_df[(sector_df['PER'] > 0) & (sector_df['PBR'] > 0)].copy()

    if len(df) < 4:
        return go.Figure()

    # 격자 생성
    per_range = np.linspace(df['PER'].quantile(0.05), df['PER'].quantile(0.95), 20)
    pbr_range = np.linspace(df['PBR'].quantile(0.05), df['PBR'].quantile(0.95), 20)

    PER_grid, PBR_grid = np.meshgrid(per_range, pbr_range)

    # 단순 저평가 추정 surface (1/PER + 1/PBR 기반)
    Z = (1 / PER_grid * 0.5 + 1 / PBR_grid * 0.5) * 100

    fig = go.Figure(data=[go.Surface(
        x=PER_grid,
        y=PBR_grid,
        z=Z,
        colorscale='RdYlGn',
        opacity=0.7,
        contours_z=dict(show=True, usecolormap=True, highlightcolor="limegreen", project_z=True),
        colorbar=dict(title="저평가 지수"),
    )])

    # 실제 종목 점 추가
    fig.add_trace(go.Scatter3d(
        x=df['PER'],
        y=df['PBR'],
        z=(1 / df['PER'] * 0.5 + 1 / df['PBR'] * 0.5) * 100,
        mode='markers',
        marker=dict(size=5, color='#333', opacity=0.8),
        text=df['Name'],
        hovertemplate='<b>%{text}</b><br>PER: %{x:.1f}<br>PBR: %{y:.2f}<extra></extra>',
        name='종목',
    ))

    fig.update_layout(
        scene=dict(
            xaxis_title='PER',
            yaxis_title='PBR',
            zaxis_title='저평가 지수',
            bgcolor='#F8F9FA',
        ),
        title=dict(text=f'{sector_name} 밸류에이션 서피스', font=dict(size=16)),
        height=500,
        margin=dict(l=0, r=0, t=50, b=0),
        paper_bgcolor='white',
    )

    return fig


def create_sector_treemap_3d(summary_df: pd.DataFrame) -> go.Figure:
    """업종 시총 비중 3D 바 차트"""
    df = summary_df.head(10).copy()

    if df.empty:
        return go.Figure()

    colors = px.colors.qualitative.Set3[:len(df)]

    fig = go.Figure(data=[go.Bar(
        x=df['업종'],
        y=df['총시가총액_조'],
        marker=dict(
            color=colors,
            line=dict(width=1, color='white'),
        ),
        text=df['총시가총액_조'].apply(lambda x: f'{x:.0f}조'),
        textposition='outside',
        hovertemplate='<b>%{x}</b><br>시가총액: %{y:.1f}조<extra></extra>',
    )])

    fig.update_layout(
        title=dict(text='업종별 시가총액 (조 원)', font=dict(size=16)),
        xaxis_title='업종',
        yaxis_title='시가총액 (조 원)',
        height=400,
        margin=dict(l=50, r=20, t=50, b=80),
        paper_bgcolor='white',
        plot_bgcolor='#FAFAFA',
        yaxis=dict(gridcolor='#E0E0E0'),
    )

    return fig
