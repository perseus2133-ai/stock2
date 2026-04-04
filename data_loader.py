"""
KRX 주식 데이터 로딩 모듈
우선순위:
  1) pykrx  - Streamlit Cloud(Linux)에서 안정적
  2) 네이버 스크래핑 - 로컬 Windows 백업
"""

import streamlit as st
import pandas as pd
import numpy as np
import requests
import re
import time
from bs4 import BeautifulSoup
from datetime import datetime, timedelta
import FinanceDataReader as fdr
from sector_mapping import map_sector


# ─── 세션 ───────────────────────────────────────────
def _session():
    s = requests.Session()
    s.headers.update({
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
        'Accept-Language': 'ko-KR,ko;q=0.9',
        'Referer': 'https://finance.naver.com/',
    })
    return s


def _num(text):
    t = re.sub(r'[^\d.\-]', '', str(text).replace(',', ''))
    try:
        v = float(t)
        return 0 if (np.isnan(v) or np.isinf(v)) else v
    except:
        return 0


# ─── 공통: KRX-DESC 종목 리스트 (업종) ────────────────
@st.cache_data(ttl=86400, show_spinner="종목 리스트 불러오는 중...")
def load_stock_listing() -> pd.DataFrame:
    try:
        df = fdr.StockListing('KRX-DESC')
        df['Code'] = df['Code'].astype(str).str.zfill(6)
        return df
    except Exception as e:
        st.warning(f"종목 리스트 로딩 실패: {e}")
        return pd.DataFrame()


# ─── 방법 1: pykrx ───────────────────────────────────
def _get_latest_trading_date() -> str:
    today = datetime.now()
    # 주말 제외, 최근 7일 내 영업일
    for i in range(7):
        d = today - timedelta(days=i)
        if d.weekday() < 5:          # 월~금
            return d.strftime('%Y%m%d')
    return (today - timedelta(days=1)).strftime('%Y%m%d')


@st.cache_data(ttl=3600, show_spinner="시장 데이터 로딩 중 (pykrx)...")
def _load_pykrx() -> pd.DataFrame:
    """pykrx로 시총 + 펀더멘탈 로드 (Cloud Linux 환경에서 정상 작동)"""
    try:
        from pykrx import stock as pykrx_stock

        date_str = _get_latest_trading_date()

        # 시가총액
        caps = pykrx_stock.get_market_cap_by_ticker(date_str, market="ALL")
        if caps is None or caps.empty:
            return pd.DataFrame()

        # 펀더멘탈
        fund = pykrx_stock.get_market_fundamental_by_ticker(date_str, market="ALL")
        if fund is None or fund.empty:
            return pd.DataFrame()

        # 병합
        df = caps.join(fund, how='left')
        df.index.name = 'Code'
        df = df.reset_index()
        df['Code'] = df['Code'].astype(str).str.zfill(6)

        # 시가총액 컬럼 통일 (한글/영문 둘 다 대응)
        for col in df.columns:
            if '시가총액' in str(col) or col == 'MarCap':
                df = df.rename(columns={col: '시가총액'})
                break

        # 필수 컬럼 확인
        required = ['시가총액', 'PER', 'PBR', 'EPS']
        if not all(c in df.columns for c in required):
            return pd.DataFrame()

        # 종가 컬럼 (없으면 0)
        if '종가' not in df.columns:
            for c in ['Close', 'close']:
                if c in df.columns:
                    df = df.rename(columns={c: '종가'})
                    break
            else:
                df['종가'] = 0

        if 'DIV' not in df.columns:
            df['DIV'] = 0
        if 'BPS' not in df.columns:
            df['BPS'] = 0

        # Name은 listing에서 가져옴 (pykrx에는 이름 없음)
        df.attrs['date'] = date_str
        return df

    except Exception as e:
        # Windows 인코딩 오류 등 → 빈 DataFrame 반환 → naver fallback
        return pd.DataFrame()


# ─── 방법 2: 네이버 스크래핑 (Windows/로컬 백업) ────────
def _naver_fetch_page(session, sosok: int, page: int) -> list:
    url = f'https://finance.naver.com/sise/sise_market_sum.naver?sosok={sosok}&page={page}'
    resp = session.get(url, timeout=15)
    soup = BeautifulSoup(resp.text, 'html.parser')

    table = soup.select_one('table.type_2')
    if not table:
        return []

    # 헤더로 컬럼 위치 동적 탐색
    header_cells = table.select('thead th') or table.select('tr th')
    headers = [h.get_text(strip=True) for h in header_cells]

    idx = {'시가총액': None, 'PER': None, 'PBR': None,
           'EPS': None, 'BPS': None, 'DIV': None}
    for i, h in enumerate(headers):
        if '시가총액' in h:   idx['시가총액'] = i
        elif h == 'PER':     idx['PER'] = i
        elif h == 'PBR':     idx['PBR'] = i
        elif 'EPS' in h:     idx['EPS'] = i
        elif 'BPS' in h:     idx['BPS'] = i
        elif '배당' in h:    idx['DIV'] = i

    rows = []
    for row in table.select('tr'):
        tds = row.select('td')
        if len(tds) < 6:
            continue
        link = row.select_one('a.tltle')
        if not link:
            continue
        href = link.get('href', '')
        code = href.split('code=')[-1] if 'code=' in href else ''
        if not code or len(code) != 6:
            continue

        texts = [td.get_text(strip=True) for td in tds]
        n = len(texts)

        def gcol(key, fallback):
            i = idx.get(key)
            if i is not None and i < n:
                return _num(texts[i])
            if fallback is not None and fallback < n:
                return _num(texts[fallback])
            return 0

        rows.append({
            'Code':  code,
            'Name':  link.text.strip(),
            '현재가': _num(texts[2]) if n > 2 else 0,
            '시가총액': gcol('시가총액', 6) * 1_000_000,
            'EPS':   gcol('EPS', 7),
            'BPS':   gcol('BPS', 8),
            'PER':   gcol('PER', 9),
            'PBR':   gcol('PBR', 10),
            'DIV':   gcol('DIV', 11),
        })
    return rows


@st.cache_data(ttl=3600, show_spinner="네이버 증권에서 시세 수집 중...")
def _load_naver_bulk(sosok: int) -> pd.DataFrame:
    session = _session()
    # 필드 설정 시도 (로컬에서는 동작)
    try:
        session.post('https://finance.naver.com/sise/field_submit.naver', data={
            'menu': 'market_sum',
            'returnUrl': f'http://finance.naver.com/sise/sise_market_sum.naver?sosok={sosok}&page=1',
            'fieldIds': ['market_sum', 'per', 'pbr', 'eps', 'bps', 'dividend'],
        }, timeout=10)
    except:
        pass

    try:
        resp = session.get(
            f'https://finance.naver.com/sise/sise_market_sum.naver?sosok={sosok}&page=1',
            timeout=10)
        pages = re.findall(r'page=(\d+)', resp.text)
        max_page = max([int(p) for p in pages]) if pages else 1
    except:
        max_page = 50

    all_rows = []
    for page in range(1, max_page + 1):
        try:
            rows = _naver_fetch_page(session, sosok, page)
            all_rows.extend(rows)
            if page % 10 == 0:
                time.sleep(0.5)
        except:
            time.sleep(1)

    if not all_rows:
        return pd.DataFrame()

    df = pd.DataFrame(all_rows)
    # Market 컬럼
    df['Market'] = 'KOSPI' if sosok == 0 else 'KOSDAQ'
    # 종가 = 현재가
    df['종가'] = df['현재가']
    # 시총 0 제거
    df = df[df['시가총액'] > 0]
    return df


# ─── 마스터 DataFrame 구성 ──────────────────────────
@st.cache_data(ttl=3600, show_spinner="전체 데이터를 구성하는 중...")
def build_master_dataframe() -> pd.DataFrame:
    # 1. 업종 정보
    listing = load_stock_listing()

    # 2. 시세 데이터: pykrx 우선, 실패 시 naver
    pykrx_df = _load_pykrx()

    if not pykrx_df.empty:
        # pykrx 성공 → 이름은 listing에서 보완
        market_data = pykrx_df.copy()
        if not listing.empty and 'Name' in listing.columns:
            market_data = market_data.merge(
                listing[['Code', 'Name']], on='Code', how='left'
            )
        if 'Name' not in market_data.columns:
            market_data['Name'] = market_data['Code']
        market_data['Market'] = '전체'

    else:
        # naver 스크래핑 fallback
        k = _load_naver_bulk(0)
        q = _load_naver_bulk(1)
        if k.empty and q.empty:
            return pd.DataFrame()
        market_data = pd.concat([k, q], ignore_index=True)

    if market_data.empty:
        return pd.DataFrame()

    # 3. ETF/ETN/스팩/리츠 필터
    etf_kw = ['KODEX', 'TIGER', 'KBSTAR', 'ARIRANG', 'HANARO', 'SOL ',
              'ACE ', 'KOSEF', 'KINDEX', 'TIMEFOLIO', 'PLUS ',
              'ETN', '스팩', 'SPAC', '리츠', 'KOFR']
    if 'Name' in market_data.columns:
        market_data = market_data[
            ~market_data['Name'].apply(
                lambda n: any(k in str(n) for k in etf_kw)
            )
        ]

    # 4. 업종 병합 (inner join → 보통주만 유지)
    if not listing.empty and 'Sector' in listing.columns:
        cols = ['Code', 'Sector', 'Industry']
        if 'Name' in listing.columns:
            cols.append('Name')
        merge_cols = [c for c in cols if c not in market_data.columns or c == 'Code']

        market_data = market_data.merge(
            listing[cols].drop_duplicates('Code'),
            on='Code', how='inner', suffixes=('', '_lst')
        )
        # Name 컬럼 정리
        if 'Name_lst' in market_data.columns:
            market_data['Name'] = market_data['Name'].fillna(market_data['Name_lst'])
            market_data.drop(columns=['Name_lst'], inplace=True)

        market_data['업종'] = market_data['Sector'].apply(map_sector)
    else:
        market_data['업종'] = '기타'

    # 5. 정리
    market_data = market_data[market_data['시가총액'] > 0]
    for col in ['DIV', 'BPS', 'EPS', 'PER', 'PBR']:
        if col not in market_data.columns:
            market_data[col] = 0
        market_data[col] = pd.to_numeric(market_data[col], errors='coerce').fillna(0)

    market_data = market_data.drop_duplicates(subset='Code').reset_index(drop=True)
    market_data.attrs['date'] = datetime.now().strftime('%Y-%m-%d')

    return market_data


# ─── 개별 주가 히스토리 ─────────────────────────────
@st.cache_data(ttl=3600, show_spinner="주가 데이터 불러오는 중...")
def load_price_history(ticker: str, days: int = 365) -> pd.DataFrame:
    try:
        end = datetime.now()
        start = end - timedelta(days=days)
        df = fdr.DataReader(ticker, start.strftime('%Y-%m-%d'), end.strftime('%Y-%m-%d'))
        return df
    except:
        return pd.DataFrame()
