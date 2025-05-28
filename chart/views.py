import pandas as pd
from django.shortcuts import render
import json
from datetime import timedelta, datetime, date as date_type
from django.http import JsonResponse
from django.core.cache import cache # Django 캐시 사용
import traceback
from django.db.models import Max, Min, F, Window
from django.db.models.functions import Lag, Round

# predict_info 앱의 모델 및 유틸리티를 가져옵니다.
try:
    from predict_info.models import StockPrice, MarketIndex
    from predict_info.utils import get_krx_stock_list 
except ImportError:
    print("[WARNING] chart/views.py: predict_info.models 또는 utils를 import할 수 없습니다.")
    StockPrice = None
    MarketIndex = None
    get_krx_stock_list = None

# 캐시 타임아웃 설정 (초 단위)
CACHE_TTL_STOCK_LIST = 60 * 60 * 24  # 종목 리스트: 24시간
CACHE_TTL_STOCK_INFO = 60 * 5        # 개별 종목 현재 정보: 5분
CACHE_TTL_STOCK_DATA = 60 * 30       # 개별 종목 기간별 데이터: 30분
CACHE_TTL_MARKET_SUMMARY = 60 * 10   # 시장 요약: 10분
CACHE_TTL_RANKINGS = 60 * 60         # 각종 순위 정보: 1시간
CACHE_TTL_52WK = 60 * 60 * 3         # 52주 최고/최저: 3시간


def get_krx_stock_list_for_chart_cached():
    cache_key = 'db_stock_list_chart_app_v3' # 캐시 키 버전 관리 용이하도록 변경
    # cache.get_or_set API를 사용하면 코드를 더 간결하게 만들 수 있습니다.
    # Django 3.2+ 에서 사용 가능. 이전 버전이면 기존 방식 유지.
    
    cached_list = cache.get(cache_key)
    if cached_list is not None:
        return cached_list

    if not StockPrice:
        print("[ERROR][chart_app] StockPrice model is not available for get_krx_stock_list_for_chart_cached.")
        # 오류 발생 시 빈 리스트를 짧은 시간 캐싱하여 반복적인 DB 접근 방지
        cache.set(cache_key, [], timeout=60) 
        return []

    try:
        print("[INFO][chart_app] Fetching unique stock list from StockPrice DB for autocomplete cache...")
        stocks_qs = StockPrice.objects.values('stock_code', 'stock_name', 'market_name').distinct().order_by('stock_name')
        stock_list_of_dicts = [
            {'name': str(s['stock_name']).strip(), 'code': str(s['stock_code']).strip(), 'market': str(s.get('market_name', '')).strip().upper()}
            for s in stocks_qs
        ]
        cache.set(cache_key, stock_list_of_dicts, timeout=CACHE_TTL_STOCK_LIST)
        print(f"[INFO][chart_app] Unique stock list from DB fetched and cached. Total: {len(stock_list_of_dicts)} stocks.")
        return stock_list_of_dicts
    except Exception as e:
        print(f"[ERROR][chart_app] Error fetching or caching unique stock list from DB: {e}")
        traceback.print_exc()
        cache.set(cache_key, [], timeout=60) # 오류 시 짧은 시간 캐시
        return []


def get_stock_info_from_db(stock_code_query):
    if not StockPrice or not stock_code_query:
        return None
    
    cache_key = f'stock_info_db_{stock_code_query}'
    cached_info = cache.get(cache_key)
    if cached_info is not None:
        return cached_info

    try:
        latest_stock_data = StockPrice.objects.filter(stock_code=stock_code_query).order_by('-date').first()
        if latest_stock_data:
            change_percent_val = latest_stock_data.change * 100 if latest_stock_data.change is not None else None
            stock_info = {
                'current_price': latest_stock_data.close_price,
                'previous_close': latest_stock_data.previous_day_close_price,
                'change': latest_stock_data.change_value,
                'change_percent': change_percent_val,
                'name': latest_stock_data.stock_name,
                'code': latest_stock_data.stock_code,
                'market_cap': latest_stock_data.market_cap,
                'market': latest_stock_data.market_name,
                'date': latest_stock_data.date.strftime('%Y-%m-%d')
            }
            cache.set(cache_key, stock_info, timeout=CACHE_TTL_STOCK_INFO)
            return stock_info
    except Exception as e:
        print(f"[ERROR][chart_app] Error fetching latest stock info from DB for {stock_code_query}: {e}")
        traceback.print_exc()
    return None # 오류 또는 데이터 없음


def get_stock_price_data_from_db(stock_code_query, period_str):
    if not StockPrice or not stock_code_query:
        return pd.DataFrame(), stock_code_query

    cache_key = f'stock_price_data_db_{stock_code_query}_{period_str}'
    cached_data = cache.get(cache_key)
    if cached_data is not None:
        # 캐시된 데이터가 DataFrame과 stock_name 튜플 형태라고 가정
        return cached_data['df'], cached_data['stock_name']

    end_date = datetime.now().date()
    # 기간 설정 (기존 로직 유지)
    if period_str == '1m': start_date = (pd.to_datetime(end_date) - pd.DateOffset(months=1)).date()
    elif period_str == '3m': start_date = (pd.to_datetime(end_date) - pd.DateOffset(months=3)).date()
    elif period_str == '6m': start_date = (pd.to_datetime(end_date) - pd.DateOffset(months=6)).date()
    elif period_str == '1y': start_date = (pd.to_datetime(end_date) - pd.DateOffset(years=1)).date()
    elif period_str == '3y': start_date = (pd.to_datetime(end_date) - pd.DateOffset(years=3)).date()
    elif period_str == '5y': start_date = (pd.to_datetime(end_date) - pd.DateOffset(years=5)).date()
    elif period_str == 'all': start_date = datetime(1990, 1, 1).date()
    else: start_date = (pd.to_datetime(end_date) - pd.DateOffset(months=6)).date()

    try:
        stock_data_qs = StockPrice.objects.filter(
            stock_code=stock_code_query,
            date__gte=start_date,
            date__lte=end_date
        ).order_by('date').values(
            'date', 'open_price', 'high_price', 'low_price', 'close_price', 'volume',
            'MA5', 'MA20'
        )
        
        stock_name_info = StockPrice.objects.filter(stock_code=stock_code_query).values('stock_name').first()
        stock_name_display = stock_name_info['stock_name'] if stock_name_info else stock_code_query

        if not stock_data_qs.exists():
            data_to_cache = {'df': pd.DataFrame(), 'stock_name': stock_name_display}
            cache.set(cache_key, data_to_cache, timeout=CACHE_TTL_STOCK_DATA)
            return pd.DataFrame(), stock_name_display

        df = pd.DataFrame(list(stock_data_qs))
        df = df.rename(columns={
            'date': 'datetime', 'open_price': 'open', 'high_price': 'high',
            'low_price': 'low', 'close_price': 'close', 'volume': 'volume',
            'MA5': 'ma5', 'MA20': 'ma20'
        })
        df['datetime'] = pd.to_datetime(df['datetime'])
        
        data_to_cache = {'df': df, 'stock_name': stock_name_display}
        cache.set(cache_key, data_to_cache, timeout=CACHE_TTL_STOCK_DATA)
        return df, stock_name_display
    except Exception as e:
        print(f"[ERROR][chart_app] Error fetching historical stock data from DB for {stock_code_query}: {e}")
        stock_info_err = StockPrice.objects.filter(stock_code=stock_code_query).values('stock_name').first()
        stock_name_display_err = stock_info_err['stock_name'] if stock_info_err else stock_code_query
        # 오류 시 빈 DataFrame과 종목명 캐싱 (짧게)
        cache.set(cache_key, {'df': pd.DataFrame(), 'stock_name': stock_name_display_err}, timeout=60)
        return pd.DataFrame(), stock_name_display_err


def chart_view(request):
    query_input = request.GET.get('query', '005930').strip()
    period = request.GET.get('period', '6m')

    resolved_stock_code = None
    resolved_stock_name = query_input 

    all_stocks_list = get_krx_stock_list_for_chart_cached()

    if query_input.isdigit() and len(query_input) == 6:
        resolved_stock_code = query_input
        stock_match = next((s for s in all_stocks_list if s['code'] == query_input), None)
        if stock_match: resolved_stock_name = stock_match['name']
    else:
        stock_match = next((s for s in all_stocks_list if s['name'].lower() == query_input.lower()), None)
        if stock_match:
            resolved_stock_code = stock_match['code']
            resolved_stock_name = stock_match['name']
        else:
            partial_match = next((s for s in all_stocks_list if query_input.lower() in s['name'].lower()), None)
            if partial_match:
                resolved_stock_code = partial_match['code']
                resolved_stock_name = partial_match['name']

    if not resolved_stock_code and all_stocks_list:
        samsung = next((s for s in all_stocks_list if s['code'] == '005930'), None)
        if samsung:
            resolved_stock_code = samsung['code']
            resolved_stock_name = samsung['name']
        else:
            resolved_stock_code = all_stocks_list[0]['code']
            resolved_stock_name = all_stocks_list[0]['name']
    elif not resolved_stock_code and not all_stocks_list: # 종목 목록도 없고, 코드도 못찾은 경우
        print(f"[WARNING][chart_app] No stock list and could not resolve '{query_input}'. Defaulting to Samsung Electronics code directly.")
        resolved_stock_code = '005930' # 삼성전자 코드로 강제 지정
        resolved_stock_name = '삼성전자'


    stock_info_db = get_stock_info_from_db(resolved_stock_code)
    df_db_data, stock_name_display_from_db = get_stock_price_data_from_db(resolved_stock_code, period)
    stock_name_display = stock_name_display_from_db if stock_name_display_from_db != resolved_stock_code else resolved_stock_name

    candle_dates, open_prices, high_prices, low_prices, close_prices, volume_data = [], [], [], [], [], []
    ma5_data, ma20_data = [], []

    if not df_db_data.empty:
        candle_dates = df_db_data['datetime'].dt.strftime('%Y-%m-%d').tolist()
        open_prices = df_db_data['open'].tolist()
        high_prices = df_db_data['high'].tolist()
        low_prices = df_db_data['low'].tolist()
        close_prices = df_db_data['close'].tolist()
        volume_data = df_db_data['volume'].tolist()
        if 'ma5' in df_db_data.columns:
            ma5_data = [round(x, 2) if pd.notnull(x) else None for x in df_db_data['ma5'].tolist()]
        if 'ma20' in df_db_data.columns:
            ma20_data = [round(x, 2) if pd.notnull(x) else None for x in df_db_data['ma20'].tolist()]

    # 52주 최고/최저가 (캐싱 적용)
    fifty_two_week_high, fifty_two_week_low = None, None
    if StockPrice and resolved_stock_code:
        cache_key_52wk = f'52wk_hl_{resolved_stock_code}'
        cached_52wk = cache.get(cache_key_52wk)
        if cached_52wk:
            fifty_two_week_high = cached_52wk['high']
            fifty_two_week_low = cached_52wk['low']
        else:
            one_year_ago_date = (datetime.now() - timedelta(weeks=52)).date()
            aggregate_52wk = StockPrice.objects.filter(
                stock_code=resolved_stock_code,
                date__gte=one_year_ago_date
            ).aggregate(max_high=Max('high_price'), min_low=Min('low_price'))
            fifty_two_week_high = aggregate_52wk.get('max_high')
            fifty_two_week_low = aggregate_52wk.get('min_low')
            cache.set(cache_key_52wk, {'high': fifty_two_week_high, 'low': fifty_two_week_low}, timeout=CACHE_TTL_52WK)
            
    # 시가총액 순위 (캐싱 적용)
    market_cap_rankings = []
    if StockPrice:
        # 가장 최근 거래일 찾기 (이것도 캐싱 가능하면 좋지만, 일단 순위 데이터 자체를 캐싱)
        latest_overall_date_obj = StockPrice.objects.aggregate(max_date=Max('date'))
        latest_overall_date = latest_overall_date_obj['max_date'] if latest_overall_date_obj else None

        if latest_overall_date:
            cache_key_mkt_cap = f'market_cap_rankings_{latest_overall_date.strftime("%Y%m%d")}'
            cached_rankings = cache.get(cache_key_mkt_cap)
            if cached_rankings:
                market_cap_rankings = cached_rankings
            else:
                try:
                    top_stocks_qs = StockPrice.objects.filter(date=latest_overall_date, market_cap__isnull=False) \
                                    .order_by('-market_cap')[:30] \
                                    .values('stock_name', 'stock_code', 'market_cap', 'market_name')
                    for stock_data in top_stocks_qs:
                        market_cap_rankings.append({
                            'name': stock_data['stock_name'],
                            'code': stock_data['stock_code'],
                            'market_cap': int(stock_data['market_cap']) if stock_data['market_cap'] else 0,
                            'market': stock_data['market_name']
                        })
                    cache.set(cache_key_mkt_cap, market_cap_rankings, timeout=CACHE_TTL_RANKINGS)
                    print(f"[INFO][chart_app] Successfully fetched and cached {len(market_cap_rankings)} market cap rankings from DB.")
                except Exception as e:
                    print(f"[ERROR][chart_app] Error fetching/caching market cap rankings from DB: {e}")
                    traceback.print_exc()
        else:
            print("[WARNING][chart_app] Could not determine latest overall date for market cap rankings.")


    # 급등/급락주 (캐싱 적용)
    top5_kospi_gainers_list, top5_kosdaq_gainers_list = [], []
    top5_kospi_losers_list, top5_kosdaq_losers_list = [], []
    if StockPrice:
        latest_stock_data_date_obj = StockPrice.objects.order_by('-date').first()
        if latest_stock_data_date_obj:
            latest_date = latest_stock_data_date_obj.date
            cache_key_top_bottom = f'top_bottom_stocks_{latest_date.strftime("%Y%m%d")}'
            cached_top_bottom = cache.get(cache_key_top_bottom)

            if cached_top_bottom:
                top5_kospi_gainers_list = cached_top_bottom.get('kp_gain', [])
                top5_kosdaq_gainers_list = cached_top_bottom.get('kd_gain', [])
                top5_kospi_losers_list = cached_top_bottom.get('kp_lose', [])
                top5_kosdaq_losers_list = cached_top_bottom.get('kd_lose', [])
            else:
                def format_stock_list(qs):
                    formatted_list = []
                    for stock in qs:
                        change_percent = stock.change * 100 if stock.change is not None else 0.0
                        status_color = "price-change-up" if change_percent > 0 else "price-change-down" if change_percent < 0 else "price-change-neutral"
                        formatted_list.append({
                            'name': stock.stock_name, 'code': stock.stock_code,
                            'change_display': f"{'+' if change_percent > 0 else ''}{change_percent:.2f}%",
                            'status': status_color, 'close': stock.close_price
                        })
                    return formatted_list

                try:
                    kospi_top5_gainers_qs = StockPrice.objects.filter(market_name='KOSPI', date=latest_date, change__isnull=False).order_by('-change')[:5]
                    kosdaq_top5_gainers_qs = StockPrice.objects.filter(market_name='KOSDAQ', date=latest_date, change__isnull=False).order_by('-change')[:5]
                    kospi_top5_losers_qs = StockPrice.objects.filter(market_name='KOSPI', date=latest_date, change__isnull=False).order_by('change')[:5]
                    kosdaq_top5_losers_qs = StockPrice.objects.filter(market_name='KOSDAQ', date=latest_date, change__isnull=False).order_by('change')[:5]

                    top5_kospi_gainers_list = format_stock_list(kospi_top5_gainers_qs)
                    top5_kosdaq_gainers_list = format_stock_list(kosdaq_top5_gainers_qs)
                    top5_kospi_losers_list = format_stock_list(kospi_top5_losers_qs)
                    top5_kosdaq_losers_list = format_stock_list(kosdaq_top5_losers_qs)
                    
                    data_to_cache_tb = {
                        'kp_gain': top5_kospi_gainers_list, 'kd_gain': top5_kosdaq_gainers_list,
                        'kp_lose': top5_kospi_losers_list, 'kd_lose': top5_kosdaq_losers_list
                    }
                    cache.set(cache_key_top_bottom, data_to_cache_tb, timeout=CACHE_TTL_RANKINGS)
                    print(f"[INFO][chart_app] Successfully fetched and cached top/bottom stocks for {latest_date}.")
                except Exception as e:
                    print(f"[ERROR][chart_app] Error fetching/caching top/bottom stocks: {e}")
                    traceback.print_exc()


    # 시장 요약 정보 (캐싱 적용)
    markets_summary = []
    if MarketIndex:
        cache_key_market_summary = 'markets_summary_data'
        cached_summary = cache.get(cache_key_market_summary)
        if cached_summary:
            markets_summary = cached_summary
        else:
            market_names_for_summary = ['KOSPI', 'KOSDAQ']
            for market_label in market_names_for_summary:
                latest_index_data = MarketIndex.objects.filter(market_name=market_label).order_by('-date').first()
                if latest_index_data:
                    change_display_val = ""
                    if latest_index_data.change_value is not None and latest_index_data.change_percent is not None:
                        prefix = "▲ " if latest_index_data.change_value > 0 else "▼ " if latest_index_data.change_value < 0 else ""
                        change_display_val = f"{prefix}{abs(latest_index_data.change_value):.2f} ({latest_index_data.change_percent:+.2f}%)"
                    status_class = "price-change-neutral"
                    if latest_index_data.change_value is not None:
                        if latest_index_data.change_value > 0: status_class = "price-change-up"
                        elif latest_index_data.change_value < 0: status_class = "price-change-down"
                    markets_summary.append({
                        'name': market_label, 'value': f"{latest_index_data.close_price:,.2f}",
                        'change': change_display_val, 'status': status_class
                    })
            # S&P, 나스닥은 현재 DB에 없으므로 하드코딩 유지 또는 제거
            markets_summary.extend([
                {'name': 'S&P 500', 'value': 'N/A', 'change': '', 'status': 'price-change-neutral'},
                {'name': '나스닥', 'value': 'N/A', 'change': '', 'status': 'price-change-neutral'},
            ])
            cache.set(cache_key_market_summary, markets_summary, timeout=CACHE_TTL_MARKET_SUMMARY)
            print(f"[INFO][chart_app] Successfully fetched and cached market summary.")


    context = {
        'stock_name_searched': query_input,
        'stock_name_displayed': stock_name_display,
        'period': period,
        'candle_dates': json.dumps(candle_dates),
        'open_prices': json.dumps(open_prices),
        'high_prices': json.dumps(high_prices),
        'low_prices': json.dumps(low_prices),
        'close_prices': json.dumps(close_prices),
        'ma5': json.dumps(ma5_data),
        'ma20': json.dumps(ma20_data),
        'volume': json.dumps(volume_data),
        'stock_info': stock_info_db,
        'fifty_two_week_high': fifty_two_week_high,
        'fifty_two_week_low': fifty_two_week_low,
        'top5_kospi_gainers': top5_kospi_gainers_list,
        'top5_kosdaq_gainers': top5_kosdaq_gainers_list,
        'top5_kospi_losers': top5_kospi_losers_list,
        'top5_kosdaq_losers': top5_kosdaq_losers_list,
        'markets': markets_summary,
        'market_cap_rankings': market_cap_rankings,
    }
    return render(request, 'chart.html', context)


def get_realtime_price(request): # 이름은 realtime이지만, DB의 최신 데이터를 반환 (캐싱된 get_stock_info_from_db 사용)
    symbol_query = request.GET.get('symbol', '').strip()
    if not symbol_query:
        return JsonResponse({'error': 'Symbol or name is required'}, status=400)

    resolved_stock_code = None
    all_stocks_list = get_krx_stock_list_for_chart_cached() 

    if symbol_query.isdigit() and len(symbol_query) == 6:
        resolved_stock_code = symbol_query
    else:
        stock_match = next((s for s in all_stocks_list if s['name'].lower() == symbol_query.lower()), None)
        if stock_match: resolved_stock_code = stock_match['code']

    if not resolved_stock_code:
         partial_match = next((s for s in all_stocks_list if symbol_query.lower() in s['name'].lower()), None)
         if partial_match: resolved_stock_code = partial_match['code']

    if not resolved_stock_code:
        return JsonResponse({'error': f"Symbol '{symbol_query}' not found in DB stock list."}, status=404)

    stock_info_from_db = get_stock_info_from_db(resolved_stock_code) # 캐싱된 함수 호출

    if stock_info_from_db and stock_info_from_db.get('current_price') is not None:
        return JsonResponse(stock_info_from_db)
    return JsonResponse({'error': f"Failed to fetch latest stock data from DB for {resolved_stock_code}"}, status=500)


def search_stocks_ajax_for_chart(request): # 이 함수는 이미 캐싱된 리스트를 사용하므로 변경 최소화
    term = request.GET.get('term', '').strip()
    limit = int(request.GET.get('limit', 7))

    if not term:
        return JsonResponse([], safe=False)

    all_stocks_list = get_krx_stock_list_for_chart_cached() 
    if not all_stocks_list: # 캐시 실패 또는 빈 목록
        # 이 경우, DB에서 직접 가져오는 fallback을 고려할 수 있으나,
        # get_krx_stock_list_for_chart_cached 내부에서 이미 DB 접근을 하므로,
        # 여기서 추가 DB 접근은 중복일 수 있음. 캐시 실패 시 에러 메시지가 더 적절.
        return JsonResponse({'error': '종목 목록을 불러올 수 없습니다. 잠시 후 다시 시도해주세요.'}, status=500)

    results = []
    term_lower = term.lower()

    for stock_item in all_stocks_list:
        stock_name_val = stock_item.get('name', '')
        stock_code_val = stock_item.get('code', '')
        match = False
        if term_lower in stock_name_val.lower(): match = True
        elif term in stock_code_val: match = True
        
        if match:
            results.append({
                'label': f"{stock_name_val} ({stock_code_val}) - {stock_item.get('market', '')}",
                'value': stock_name_val, 
            })
        if len(results) >= limit:
            break
    return JsonResponse(results, safe=False)

