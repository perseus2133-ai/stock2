"""
KRX 주식 데이터 로딩 모듈
- 네이버 증권: 시가총액, PER, PBR, EPS, BPS 일괄 수집
- FinanceDataReader: 종목 업종 정보, 가격 데이터
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


def _create_session():
    session = requests.Session()
    session.headers.update({
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'Accept-Language': 'ko-KR,ko;q=0.9',
        'Referer': 'https://finance.naver.com/',
    })
    return session


def _parse_num(text):
    text = str(text).strip().replace(',', '').replace('+', '').replace('%', '')
    text = re.sub(r'[^\d.\-]', '', text)
    try:
        v = float(text)
        return v if not (np.isnan(v) or np.isinf(v)) else 0
    except:
        return 0


@st.cache_data(ttl=3600, show_spinner="종목 리스트 불러오는 중...")
def load_stock_listing() -> pd.DataFrame:
    """KRX 전 종목 리스트 (업종 포함)"""
    try:
        df = fdr.StockListing('KRX-DESC')
        if 'Code' in df.columns:
            df['Code'] = df['Code'].astype(str).str.zfill(6)
        return df
    except Exception as e:
        st.warning(f"종목 리스트 로딩 실패: {e}")
        return pd.DataFrame()


def _fetch_page(session, sosok: int, page: int) -> list:
    """한 페이지에서 종목 데이터 파싱 — 헤더 기반 동적 컬럼 탐색"""
    url = f'https://finance.naver.com/sise/sise_market_sum.naver?sosok={sosok}&page={page}'
    resp = session.get(url, timeout=10)
    soup = BeautifulSoup(resp.text, 'html.parser')

    table = soup.select_one('table.type_2')
    if not table:
        return []

    # 헤더에서 컬럼 인덱스 추출
    header_row = table.select_one('thead tr') or table.select('tr')[0]
    headers = [th.get_text(strip=True) for th in header_row.select('th')]

    # 알려진 헤더 이름 매핑 (한글 인코딩 이슈 대비 숫자 위치로도 처리)
    col_idx = {
        '시가총액': None, 'PER': None, 'PBR': None,
        'EPS': None, 'BPS': None, '배당수익률': None,
    }
    for i, h in enumerate(headers):
        if '시가총액' in h or '시총' in h:
            col_idx['시가총액'] = i
        elif h == 'PER':
            col_idx['PER'] = i
        elif h == 'PBR':
            col_idx['PBR'] = i
        elif 'EPS' in h:
            col_idx['EPS'] = i
        elif 'BPS' in h:
            col_idx['BPS'] = i
        elif '배당' in h:
            col_idx['배당수익률'] = i

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

        td_texts = [td.get_text(strip=True) for td in tds]
        n = len(td_texts)

        # 헤더 기반 추출 (인덱스 보정: td는 N컬럼 포함이므로 header idx와 일치)
        def get_col(key, fallback_idx):
            idx = col_idx.get(key)
            if idx is not None and idx < n:
                return _parse_num(td_texts[idx])
            if fallback_idx is not None and fallback_idx < n:
                return _parse_num(td_texts[fallback_idx])
            return 0

        # 기본 폴백 인덱스 (필드 설정 성공 시의 순서):
        # [0]=N [1]=종목명 [2]=현재가 [3]=전일비 [4]=등락률
        # [5]=액면가 [6]=시가총액 [7]=EPS [8]=BPS [9]=PER [10]=PBR [11]=DIV
        data = {
            'Code': code,
            'Name': link.text.strip(),
            '현재가': _parse_num(td_texts[2]) if n > 2 else 0,
            '시가총액': get_col('시가총액', 6) * 1_000_000,  # 백만원→원
            'EPS': get_col('EPS', 7),
            'BPS': get_col('BPS', 8),
            'PER': get_col('PER', 9),
            'PBR': get_col('PBR', 10),
            'DIV': get_col('배당수익률', 11),
        }

        # 시총이 비정상이면 현재가 * 간이추정 (PBR * BPS * 추정발행주 불가 → 스킵하지 않고 일단 포함)
        rows.append(data)

    return rows


@st.cache_data(ttl=3600, show_spinner="네이버 증권에서 시세 데이터 수집 중...")
def _fetch_naver_bulk(sosok: int) -> pd.DataFrame:
    """
    네이버 증권 시가총액 순위 페이지에서 전 종목 데이터 일괄 수집
    sosok: 0=KOSPI, 1=KOSDAQ
    """
    session = _create_session()

    # 필드 설정 시도 (Cloud 환경에서 쿠키 미유지 가능 → 동적 헤더 파싱으로 보완)
    try:
        field_data = {
            'menu': 'market_sum',
            'returnUrl': f'http://finance.naver.com/sise/sise_market_sum.naver?sosok={sosok}&page=1',
            'fieldIds': ['market_sum', 'per', 'pbr', 'eps', 'bps', 'dividend'],
        }
        session.post('https://finance.naver.com/sise/field_submit.naver',
                     data=field_data, timeout=10)
    except:
        pass

    # 총 페이지 수 확인
    try:
        resp = session.get(
            f'https://finance.naver.com/sise/sise_market_sum.naver?sosok={sosok}&page=1',
            timeout=10
        )
        pages = re.findall(r'page=(\d+)', resp.text)
        max_page = max([int(p) for p in pages]) if pages else 1
    except:
        max_page = 50

    all_rows = []

    for page in range(1, max_page + 1):
        try:
            rows = _fetch_page(session, sosok, page)
            all_rows.extend(rows)

            if page % 10 == 0:
                time.sleep(0.5)
        except Exception:
            time.sleep(1)
            continue

    if not all_rows:
        return pd.DataFrame()

    df = pd.DataFrame(all_rows)

    # 시총 0인 행 중 현재가와 PBR*BPS로 추정 가능한 경우 보완
    no_cap = df['시가총액'] == 0
    if no_cap.any():
        # PBR * BPS 비율로 현재가 대비 시총 근사 불가능하므로 제거
        df = df[~no_cap]

    return df


@st.cache_data(ttl=3600, show_spinner="전체 데이터를 구성하는 중...")
def build_master_dataframe() -> pd.DataFrame:
    """전체 데이터 병합: 네이버 시세 + FDR 업종"""
    # 1. 종목 리스트 (업종)
    listing = load_stock_listing()

    # 2. 네이버 시세 데이터 (KOSPI + KOSDAQ)
    kospi = _fetch_naver_bulk(0)
    kosdaq = _fetch_naver_bulk(1)

    if kospi.empty and kosdaq.empty:
        return pd.DataFrame()

    market_data = pd.concat([kospi, kosdaq], ignore_index=True)

    # Market 컬럼 추가
    if 'Market' not in market_data.columns:
        market_data['Market'] = '전체'

    # 3. ETF/ETN/스팩/리츠 필터링
    etf_keywords = ['KODEX', 'TIGER', 'KBSTAR', 'ARIRANG', 'HANARO', 'SOL ',
                    'ACE ', 'KOSEF', 'KINDEX', 'TIMEFOLIO', 'PLUS ',
                    'ETN', '스팩', 'SPAC', '리츠', 'KOFR']
    mask = market_data['Name'].apply(
        lambda n: not any(kw in str(n) for kw in etf_keywords)
    )
    market_data = market_data[mask]

    # 4. 업종 정보 병합
    if not listing.empty and 'Sector' in listing.columns:
        listing_cols = listing[['Code', 'Sector', 'Industry']].copy()
        market_data = market_data.merge(listing_cols, on='Code', how='inner')
        market_data['업종'] = market_data['Sector'].apply(map_sector)
    else:
        market_data['업종'] = '기타'
        market_data['Sector'] = ''
        market_data['Industry'] = ''

    # 5. 정리
    market_data = market_data[market_data['시가총액'] > 0]
    market_data['DIV'] = market_data['DIV'].fillna(0)
    market_data = market_data.drop_duplicates(subset='Code')
    market_data = market_data.reset_index(drop=True)

    market_data.attrs['date'] = datetime.now().strftime('%Y-%m-%d')

    return market_data


@st.cache_data(ttl=3600, show_spinner="주가 데이터 불러오는 중...")
def load_price_history(ticker: str, days: int = 365) -> pd.DataFrame:
    """개별 종목 가격 히스토리"""
    try:
        end = datetime.now()
        start = end - timedelta(days=days)
        df = fdr.DataReader(ticker, start.strftime('%Y-%m-%d'), end.strftime('%Y-%m-%d'))
        return df
    except:
        return pd.DataFrame()
