"""
KRX 주식 데이터 로딩 모듈
전략:
  시가총액  : FDR StockListing (안정적)
  펀더멘탈  : pykrx → KRX OTP-CSV → 네이버 스크래핑 순 fallback
  업종 정보 : FDR KRX-DESC Industry (세분류) 우선
"""

import io
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

_debug_log: list = []  # 디버그 메시지 저장


def _log(msg: str):
    _debug_log.append(msg)


def _session():
    s = requests.Session()
    s.headers.update({
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
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
        _log(f"KRX-DESC 실패: {e}")
        return pd.DataFrame()


# ─── FDR 시가총액 수집 ──────────────────────────────
def _load_fdr_caps() -> pd.DataFrame:
    """FDR StockListing으로 KOSPI+KOSDAQ 시가총액/종가 수집 (PER/PBR도 있으면 가져옴)"""
    frames = []
    for mkt in ['KOSPI', 'KOSDAQ']:
        try:
            df_mkt = fdr.StockListing(mkt)
            if df_mkt is not None and not df_mkt.empty:
                df_mkt = df_mkt.copy()
                df_mkt['_mkt'] = mkt
                frames.append(df_mkt)
        except Exception as e:
            _log(f"FDR {mkt} 실패: {e}")

    if not frames:
        return pd.DataFrame()

    df = pd.concat(frames, ignore_index=True)
    _log(f"FDR 컬럼: {list(df.columns)}")

    # Code 컬럼 통일
    for col in ['Code', 'Symbol', 'ISU_SRT_CD', 'Ticker']:
        if col in df.columns:
            df = df.rename(columns={col: 'Code'})
            break
    if 'Code' not in df.columns:
        return pd.DataFrame()

    df['Code'] = df['Code'].astype(str).str.extract(r'(\d{6})')[0].fillna('')
    df = df[df['Code'].str.len() == 6].copy()

    # 시가총액
    for col in df.columns:
        if str(col).lower() in ('marcap', 'mktcap') or '시가총액' in str(col):
            df = df.rename(columns={col: '시가총액'})
            break
    if '시가총액' not in df.columns:
        df['시가총액'] = 0

    # 종가
    for col in ['Close', 'close', '종가', 'Price', 'TDD_CLSPRC']:
        if col in df.columns:
            df = df.rename(columns={col: '종가'})
            break
    if '종가' not in df.columns:
        df['종가'] = 0

    # 이름
    for col in ['Name', 'name', '종목명', 'ISU_ABBRV']:
        if col in df.columns:
            df = df.rename(columns={col: 'Name'})
            break
    if 'Name' not in df.columns:
        df['Name'] = df['Code']

    # PER/PBR/EPS 있으면 가져옴
    for target, candidates in [
        ('PER', ['PER', 'per']),
        ('PBR', ['PBR', 'pbr']),
        ('EPS', ['EPS', 'eps']),
        ('BPS', ['BPS', 'bps']),
        ('DIV', ['DIV', 'div', 'DVD_YLD']),
    ]:
        if target not in df.columns:
            for c in candidates:
                if c in df.columns and c != target:
                    df = df.rename(columns={c: target})
                    break

    keep = ['Code', 'Name', '시가총액', '종가']
    for c in ['PER', 'PBR', 'EPS', 'BPS', 'DIV']:
        if c in df.columns:
            keep.append(c)

    result = df[[c for c in keep if c in df.columns]].copy()
    for c in ['시가총액', '종가', 'PER', 'PBR', 'EPS', 'BPS', 'DIV']:
        if c in result.columns:
            result[c] = pd.to_numeric(result[c], errors='coerce').fillna(0)

    return result


# ─── 펀더멘탈 1: pykrx ─────────────────────────────
def _load_pykrx_fund(date_str: str) -> pd.DataFrame:
    try:
        from pykrx import stock as pykrx_stock
        frames = []
        for market in ["KOSPI", "KOSDAQ"]:
            try:
                fund = pykrx_stock.get_market_fundamental_by_ticker(date_str, market=market)
                if fund is not None and not fund.empty:
                    frames.append(fund)
            except Exception as e:
                _log(f"pykrx {market} fund 실패: {e}")
        if not frames:
            _log("pykrx: 전체 실패")
            return pd.DataFrame()
        df = pd.concat(frames)
        df.index.name = 'Code'
        df = df.reset_index()
        df['Code'] = df['Code'].astype(str).str.zfill(6)
        _log(f"pykrx 성공: {len(df)}행, 컬럼={list(df.columns)}")
        return df
    except Exception as e:
        _log(f"pykrx import/실행 실패: {e}")
        return pd.DataFrame()


# ─── 펀더멘탈 2: KRX OTP-CSV 다운로드 ───────────────
def _load_krx_fund_csv(date_str: str) -> pd.DataFrame:
    """KRX 데이터포털 OTP-CSV 방식으로 PER/PBR/EPS 수집"""
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
            'Referer': 'http://data.krx.co.kr/contents/MDC/MDI/mdiIdx/MKD020210301.html',
        }

        # Step 1: OTP 생성
        otp_r = requests.post(
            "http://data.krx.co.kr/comm/fileDn/GenerateOTP/generate.cmd",
            headers=headers,
            data={
                'locale': 'ko_KR',
                'mktId': 'ALL',
                'trdDd': date_str,
                'share': '1',
                'money': '1',
                'csvxls_isNo': 'false',
                'name': 'fileDown',
                'url': 'dbms/MDC/STAT/standard/MDCSTAT03401',
            },
            timeout=15
        )
        otp = otp_r.text
        if not otp or len(otp) < 10:
            _log(f"KRX OTP 실패: len={len(otp) if otp else 0}")
            return pd.DataFrame()

        # Step 2: CSV 다운로드
        csv_r = requests.post(
            "http://data.krx.co.kr/comm/fileDn/download_csv/download.cmd",
            headers=headers,
            data={'code': otp},
            timeout=30
        )

        # euc-kr 또는 utf-8로 시도
        for enc in ['euc-kr', 'cp949', 'utf-8']:
            try:
                df = pd.read_csv(io.BytesIO(csv_r.content), encoding=enc)
                if not df.empty:
                    break
            except:
                continue
        else:
            _log("KRX CSV 파싱 실패")
            return pd.DataFrame()

        _log(f"KRX CSV 컬럼: {list(df.columns)}")

        # 종목코드 컬럼
        for c in df.columns:
            if '종목코드' in c or '코드' in c or 'ISU' in c:
                df = df.rename(columns={c: 'Code'})
                break
        if 'Code' not in df.columns:
            _log("KRX CSV: Code 컬럼 없음")
            return pd.DataFrame()

        df['Code'] = df['Code'].astype(str).str.zfill(6)

        # 지표 컬럼 매핑
        for target, keywords in [
            ('PER', ['PER', '주가수익비율']),
            ('PBR', ['PBR', '주가순자산비율']),
            ('EPS', ['EPS', '주당순이익']),
            ('BPS', ['BPS', '주당순자산']),
            ('DIV', ['DIV', '배당수익률']),
        ]:
            if target not in df.columns:
                for kw in keywords:
                    for c in df.columns:
                        if kw in str(c):
                            df = df.rename(columns={c: target})
                            break
                    if target in df.columns:
                        break

        for col in ['BPS', 'PER', 'PBR', 'EPS', 'DIV']:
            if col in df.columns:
                df[col] = pd.to_numeric(
                    df[col].astype(str).str.replace(',', ''), errors='coerce'
                ).fillna(0)
            else:
                df[col] = 0

        _log(f"KRX CSV 성공: {len(df)}행")
        return df[['Code', 'BPS', 'PER', 'PBR', 'EPS', 'DIV']].copy()

    except Exception as e:
        _log(f"KRX CSV 실패: {e}")
        return pd.DataFrame()


# ─── 펀더멘탈 3: KRX JSON API 직접 ──────────────────
def _load_krx_fund_json(date_str: str) -> pd.DataFrame:
    """KRX JSON API로 PER/PBR/EPS 수집"""
    try:
        for bld in [
            'dbms/MDC/STAT/standard/MDCSTAT03401',
            'dbms/MDC/STAT/standard/MDCSTAT03402',
        ]:
            try:
                r = requests.post(
                    "http://data.krx.co.kr/comm/bldAttendant/getJsonData.cmd",
                    headers={
                        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)',
                        'Referer': 'http://data.krx.co.kr/',
                    },
                    data={
                        'bld': bld,
                        'mktId': 'ALL',
                        'trdDd': date_str,
                        'share': '1',
                        'money': '1',
                        'csvxls_isNo': 'false',
                    },
                    timeout=30
                )
                data = r.json()
                rows = None
                for key, val in data.items():
                    if isinstance(val, list) and len(val) > 0:
                        rows = val
                        break
                if rows and len(rows) > 100:
                    _log(f"KRX JSON ({bld}) 성공: {len(rows)}행")
                    break
            except:
                continue
        else:
            _log("KRX JSON: 전체 실패")
            return pd.DataFrame()

        df = pd.DataFrame(rows)
        _log(f"KRX JSON 컬럼: {list(df.columns)}")

        code_col = next(
            (c for c in df.columns if 'SRT_CD' in c or 'ISU_CD' in c.upper()),
            None
        )
        if not code_col:
            return pd.DataFrame()
        df = df.rename(columns={code_col: 'Code'})
        df['Code'] = df['Code'].astype(str).str.zfill(6)

        col_map = {}
        for target, candidates in [
            ('BPS', ['BPS']), ('PER', ['PER']), ('PBR', ['PBR']),
            ('EPS', ['EPS']), ('DIV', ['DIV', 'DVD_YLD']),
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
    except Exception as e:
        _log(f"KRX JSON 실패: {e}")
        return pd.DataFrame()


# ─── 펀더멘탈 4: 네이버 스크래핑 ────────────────────
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
            'Code': code,
            'PER': gcol('PER', None),
            'PBR': gcol('PBR', None),
            'EPS': gcol('EPS', None),
            'BPS': gcol('BPS', None),
            'DIV': gcol('DIV', None),
        })
    return rows


def _load_naver_fund() -> pd.DataFrame:
    """네이버 스크래핑으로 PER/PBR/EPS만 가져오기 (시가총액은 FDR 사용)"""
    session = _session()

    all_rows = []
    for sosok in [0, 1]:
        try:
            # 필드 설정 쿠키 (튜플 리스트로 보냄)
            session.post(
                'https://finance.naver.com/sise/field_submit.naver',
                data=[
                    ('menu', 'market_sum'),
                    ('returnUrl', f'http://finance.naver.com/sise/sise_market_sum.naver?sosok={sosok}&page=1'),
                    ('fieldIds', 'per'),
                    ('fieldIds', 'pbr'),
                    ('fieldIds', 'eps'),
                    ('fieldIds', 'bps'),
                    ('fieldIds', 'dividend'),
                ],
                timeout=10,
                allow_redirects=True
            )
        except:
            pass

        try:
            resp = session.get(
                f'https://finance.naver.com/sise/sise_market_sum.naver?sosok={sosok}&page=1',
                timeout=10)
            pages = re.findall(r'page=(\d+)', resp.text)
            max_page = min(max([int(p) for p in pages]) if pages else 1, 50)
        except:
            max_page = 40

        for page in range(1, max_page + 1):
            try:
                rows = _naver_fetch_page(session, sosok, page)
                all_rows.extend(rows)
                if page % 15 == 0:
                    time.sleep(0.3)
            except:
                pass

    if not all_rows:
        _log("네이버 스크래핑 실패: 0행")
        return pd.DataFrame()

    df = pd.DataFrame(all_rows)
    # PER이 하나도 없으면 쿠키 실패로 판단
    if df['PER'].sum() == 0:
        _log("네이버 스크래핑: PER 전부 0 (쿠키 실패)")
        return pd.DataFrame()

    _log(f"네이버 스크래핑 성공: {len(df)}행, PER>0: {(df['PER']>0).sum()}")
    return df


# ─── 마스터 DataFrame 구성 ──────────────────────────
@st.cache_data(ttl=3600, show_spinner="전체 데이터를 구성하는 중...")
def build_master_dataframe() -> tuple:
    global _debug_log
    _debug_log = []

    # 1. 업종 정보 (Industry 세분류 포함)
    listing = load_stock_listing()

    # 2. 시가총액: FDR
    caps_df = _load_fdr_caps()

    if caps_df.empty:
        _log("FDR caps 실패 → 대체 없음")
        return pd.DataFrame(), []

    _log(f"FDR caps: {len(caps_df)}행")

    # 3. 펀더멘탈 (이미 FDR에 있으면 스킵)
    has_fund = 'PER' in caps_df.columns and caps_df['PER'].sum() > 0
    _log(f"FDR에 PER 있음: {has_fund}")

    if not has_fund:
        date_str = _get_latest_trading_date()
        _log(f"펀더멘탈 수집 시작 (기준일: {date_str})")

        # 시도 1: pykrx
        fund_df = _load_pykrx_fund(date_str)

        # 시도 2: KRX OTP-CSV
        if fund_df.empty or fund_df.get('PER', pd.Series([0])).sum() == 0:
            fund_df2 = _load_krx_fund_csv(date_str)
            if not fund_df2.empty and fund_df2['PER'].sum() > 0:
                fund_df = fund_df2

        # 시도 3: KRX JSON API
        if fund_df.empty or fund_df.get('PER', pd.Series([0])).sum() == 0:
            fund_df3 = _load_krx_fund_json(date_str)
            if not fund_df3.empty and fund_df3['PER'].sum() > 0:
                fund_df = fund_df3

        # 시도 4: 네이버 스크래핑
        if fund_df.empty or fund_df.get('PER', pd.Series([0])).sum() == 0:
            fund_df4 = _load_naver_fund()
            if not fund_df4.empty and fund_df4['PER'].sum() > 0:
                fund_df = fund_df4

        if not fund_df.empty:
            fund_cols = ['Code'] + [c for c in ['BPS', 'PER', 'PBR', 'EPS', 'DIV']
                                     if c in fund_df.columns]
            caps_df = caps_df.merge(fund_df[fund_cols], on='Code', how='left',
                                     suffixes=('_old', ''))
            # 새로 merge된 컬럼 우선
            for c in ['BPS', 'PER', 'PBR', 'EPS', 'DIV']:
                if f'{c}_old' in caps_df.columns:
                    caps_df[c] = caps_df[c].fillna(caps_df[f'{c}_old'])
                    caps_df = caps_df.drop(columns=[f'{c}_old'])
        else:
            _log("모든 펀더멘탈 소스 실패!")

    market_data = caps_df

    # 4. 기본 컬럼 보장
    for col in ['BPS', 'PER', 'PBR', 'EPS', 'DIV']:
        if col not in market_data.columns:
            market_data[col] = 0
        market_data[col] = pd.to_numeric(market_data[col], errors='coerce').fillna(0)

    # 5. ETF/ETN/스팩/리츠 필터
    etf_kw = ['KODEX', 'TIGER', 'KBSTAR', 'ARIRANG', 'HANARO', 'SOL ',
              'ACE ', 'KOSEF', 'KINDEX', 'TIMEFOLIO', 'PLUS ',
              'ETN', '스팩', 'SPAC', '리츠', 'KOFR']
    if 'Name' in market_data.columns:
        market_data = market_data[
            ~market_data['Name'].apply(
                lambda n: any(k in str(n) for k in etf_kw)
            )
        ]

    # 6. 업종 병합 (inner join → 보통주만 유지)
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

        # Industry(세분류) 우선, 기타면 Sector 재시도
        if 'Industry' in market_data.columns:
            market_data['업종'] = market_data['Industry'].apply(map_sector)
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

    # 7. 정리
    market_data = market_data[market_data['시가총액'] > 0]
    market_data = market_data.drop_duplicates(subset='Code').reset_index(drop=True)
    market_data.attrs['date'] = datetime.now().strftime('%Y-%m-%d')

    _log(f"최종: {len(market_data)}행, PER>0: {(market_data['PER']>0).sum()}")

    return market_data, list(_debug_log)


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
