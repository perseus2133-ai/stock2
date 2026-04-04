"""
KRX 주식 데이터 로딩 모듈
전략:
  시가총액  : FDR StockListing (안정적)
  펀더멘탈  : pykrx → KRX API 직접 → 네이버 순으로 fallback
  업종 정보 : FDR KRX-DESC Industry (세분류) 우선
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


def _get_latest_trading_date() -> str:
    today = datetime.now()
    for i in range(7):
        d = today - timedelta(days=i)
        if d.weekday() < 5:
            return d.strftime('%Y%m%d')
    return (today - timedelta(days=1)).strftime('%Y%m%d')


# ─── 종목 리스트 (업종 정보) ────────────────────────
@st.cache_data(ttl=86400, show_spinner="종목 리스트 불러오는 중...")
def load_stock_listing() -> pd.DataFrame:
    try:
        df = fdr.StockListing('KRX-DESC')
        df['Code'] = df['Code'].astype(str).str.zfill(6)
        return df
    except Exception as e:
        st.warning(f"종목 리스트 로딩 실패: {e}")
        return pd.DataFrame()


# ─── FDR 시가총액 수집 ──────────────────────────────
def _load_fdr_caps() -> pd.DataFrame:
    """FDR StockListing으로 KOSPI+KOSDAQ 시가총액/종가 수집"""
    frames = []
    for mkt in ['KOSPI', 'KOSDAQ']:
        try:
            df_mkt = fdr.StockListing(mkt)
            if df_mkt is not None and not df_mkt.empty:
                df_mkt = df_mkt.copy()
                df_mkt['_mkt'] = mkt
                frames.append(df_mkt)
        except:
            pass

    if not frames:
        return pd.DataFrame()

    df = pd.concat(frames, ignore_index=True)

    # Code 컬럼 통일
    for col in ['Code', 'Symbol', 'ISU_SRT_CD']:
        if col in df.columns:
            df = df.rename(columns={col: 'Code'})
            break
    if 'Code' not in df.columns:
        return pd.DataFrame()

    df['Code'] = df['Code'].astype(str).str.extract(r'(\d{6})')[0].fillna('')
    df = df[df['Code'].str.len() == 6].copy()

    # 시가총액
    for col in df.columns:
        if str(col).lower() == 'marcap' or '시가총액' in str(col):
            df = df.rename(columns={col: '시가총액'})
            break
    if '시가총액' not in df.columns:
        df['시가총액'] = 0

    # 종가
    for col in ['Close', 'close', '종가', 'Price']:
        if col in df.columns:
            df = df.rename(columns={col: '종가'})
            break
    if '종가' not in df.columns:
        df['종가'] = 0

    # 이름
    for col in ['Name', 'name', '종목명']:
        if col in df.columns:
            df = df.rename(columns={col: 'Name'})
            break
    if 'Name' not in df.columns:
        df['Name'] = df['Code']

    df['시가총액'] = pd.to_numeric(df['시가총액'], errors='coerce').fillna(0)
    df['종가'] = pd.to_numeric(df['종가'], errors='coerce').fillna(0)

    return df[['Code', 'Name', '시가총액', '종가']].copy()


# ─── 펀더멘탈: pykrx ───────────────────────────────
def _load_pykrx_fund(date_str: str) -> pd.DataFrame:
    try:
        from pykrx import stock as pykrx_stock
        frames = []
        for market in ["KOSPI", "KOSDAQ"]:
            try:
                fund = pykrx_stock.get_market_fundamental_by_ticker(date_str, market=market)
                if fund is not None and not fund.empty:
                    frames.append(fund)
            except:
                pass
        if not frames:
            return pd.DataFrame()
        df = pd.concat(frames)
        df.index.name = 'Code'
        df = df.reset_index()
        df['Code'] = df['Code'].astype(str).str.zfill(6)
        return df
    except:
        return pd.DataFrame()


# ─── 펀더멘탈: KRX API 직접 ────────────────────────
def _load_krx_fund(date_str: str) -> pd.DataFrame:
    """KRX 데이터포털 API로 PER/PBR/EPS/BPS/DIV 직접 수집"""
    try:
        r = requests.post(
            "http://data.krx.co.kr/comm/bldAttendant/getJsonData.cmd",
            headers={
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)',
                'Referer': 'http://data.krx.co.kr/',
                'Content-Type': 'application/x-www-form-urlencoded; charset=UTF-8',
            },
            data={
                'bld': 'dbms/MDC/STAT/standard/MDCSTAT03402',
                'mktId': 'ALL',
                'trdDd': date_str,
                'csvxls_isNo': 'false',
            },
            timeout=30
        )
        data = r.json()

        # 응답에서 리스트 블록 찾기
        rows = None
        for key, val in data.items():
            if isinstance(val, list) and len(val) > 0:
                rows = val
                break
        if not rows:
            return pd.DataFrame()

        df = pd.DataFrame(rows)

        # 종목코드 컬럼
        code_col = next(
            (c for c in df.columns if 'SRT_CD' in c or ('ISU' in c.upper() and 'CD' in c.upper())),
            None
        )
        if code_col is None:
            return pd.DataFrame()
        df = df.rename(columns={code_col: 'Code'})
        df['Code'] = df['Code'].astype(str).str.zfill(6)

        # 지표 컬럼 매핑 (KRX API는 영문 약어를 그대로 쓰는 경우가 많음)
        col_map = {}
        for target, candidates in [
            ('BPS', ['BPS']),
            ('PER', ['PER']),
            ('PBR', ['PBR']),
            ('EPS', ['EPS']),
            ('DIV', ['DIV', 'DVD_YLD']),
        ]:
            for c in candidates:
                if c in df.columns:
                    col_map[c] = target
                    break
        df = df.rename(columns=col_map)

        for col in ['BPS', 'PER', 'PBR', 'EPS', 'DIV']:
            if col in df.columns:
                df[col] = pd.to_numeric(
                    df[col].astype(str).str.replace(',', ''), errors='coerce'
                ).fillna(0)
            else:
                df[col] = 0

        return df[['Code', 'BPS', 'PER', 'PBR', 'EPS', 'DIV']].copy()
    except:
        return pd.DataFrame()


# ─── 펀더멘탈: 네이버 스크래핑 fallback ─────────────
def _naver_fetch_page(session, sosok: int, page: int) -> list:
    url = f'https://finance.naver.com/sise/sise_market_sum.naver?sosok={sosok}&page={page}'
    resp = session.get(url, timeout=15)
    soup = BeautifulSoup(resp.text, 'html.parser')
    table = soup.select_one('table.type_2')
    if not table:
        return []

    header_cells = table.select('thead th') or table.select('tr th')
    headers = [h.get_text(strip=True) for h in header_cells]
    idx = {'시가총액': None, 'PER': None, 'PBR': None, 'EPS': None, 'BPS': None, 'DIV': None}
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
    df['Market'] = 'KOSPI' if sosok == 0 else 'KOSDAQ'
    df['종가'] = df['현재가']
    df = df[df['시가총액'] > 0]
    return df


# ─── 마스터 DataFrame 구성 ──────────────────────────
@st.cache_data(ttl=3600, show_spinner="전체 데이터를 구성하는 중...")
def build_master_dataframe() -> pd.DataFrame:
    # 1. 업종 정보 (Industry 세분류 포함)
    listing = load_stock_listing()

    # 2. 시가총액: FDR (안정적)
    caps_df = _load_fdr_caps()

    if caps_df.empty:
        # FDR 실패 시 naver로 시가총액도 가져옴
        k = _load_naver_bulk(0)
        q = _load_naver_bulk(1)
        if k.empty and q.empty:
            return pd.DataFrame()
        market_data = pd.concat([k, q], ignore_index=True)
        # naver에는 펀더멘탈도 있으므로 바로 업종 병합으로
    else:
        # 3. 펀더멘탈: pykrx → KRX API 순
        date_str = _get_latest_trading_date()
        fund_df = _load_pykrx_fund(date_str)
        if fund_df.empty:
            fund_df = _load_krx_fund(date_str)

        if not fund_df.empty:
            fund_cols = ['Code'] + [c for c in ['BPS', 'PER', 'PBR', 'EPS', 'DIV'] if c in fund_df.columns]
            market_data = caps_df.merge(fund_df[fund_cols], on='Code', how='left')
        else:
            market_data = caps_df.copy()

        for col in ['BPS', 'PER', 'PBR', 'EPS', 'DIV']:
            if col not in market_data.columns:
                market_data[col] = 0
            market_data[col] = pd.to_numeric(market_data[col], errors='coerce').fillna(0)

    if market_data.empty:
        return pd.DataFrame()

    # 4. ETF/ETN/스팩/리츠 필터
    etf_kw = ['KODEX', 'TIGER', 'KBSTAR', 'ARIRANG', 'HANARO', 'SOL ',
              'ACE ', 'KOSEF', 'KINDEX', 'TIMEFOLIO', 'PLUS ',
              'ETN', '스팩', 'SPAC', '리츠', 'KOFR']
    if 'Name' in market_data.columns:
        market_data = market_data[
            ~market_data['Name'].apply(
                lambda n: any(k in str(n) for k in etf_kw)
            )
        ]

    # 5. 업종 병합 (inner join → 보통주만 유지)
    if not listing.empty and ('Sector' in listing.columns or 'Industry' in listing.columns):
        avail_cols = ['Code']
        for c in ['Sector', 'Industry', 'Name']:
            if c in listing.columns:
                avail_cols.append(c)

        market_data = market_data.merge(
            listing[avail_cols].drop_duplicates('Code'),
            on='Code', how='inner', suffixes=('', '_lst')
        )
        if 'Name_lst' in market_data.columns:
            market_data['Name'] = market_data['Name'].fillna(market_data['Name_lst'])
            market_data.drop(columns=['Name_lst'], inplace=True)

        # ★ Industry(세분류) 우선으로 업종 매핑, 기타면 Sector로 재시도
        if 'Industry' in market_data.columns:
            market_data['업종'] = market_data['Industry'].apply(map_sector)
            # 여전히 기타인 것은 Sector로 재시도
            mask_other = market_data['업종'] == '기타'
            if mask_other.any() and 'Sector' in market_data.columns:
                market_data.loc[mask_other, '업종'] = (
                    market_data.loc[mask_other, 'Sector'].apply(map_sector)
                )
        elif 'Sector' in market_data.columns:
            market_data['업종'] = market_data['Sector'].apply(map_sector)
        else:
            market_data['업종'] = '기타'
    else:
        market_data['업종'] = '기타'

    # 6. 정리
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
