#!/usr/bin/env python3
"""
创业板 混合策略 — 每日自动运行脚本
=====================================
功能: 自动获取最新数据, 判断买卖信号, 记录操作历史

用法:
  python daily_runner.py                    # 检查信号(模拟模式)
  python daily_runner.py --trade            # 实盘模式(记录到持仓日志)
  python daily_runner.py --history          # 查看历史操作记录
"""

import sys, io, os, json
from datetime import datetime, timedelta
if sys.stdout.encoding != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

import baostock as bs, pandas as pd, numpy as np
if not hasattr(pd.DataFrame, 'append'):
    pd.DataFrame.append = lambda s,o,**kw: pd.concat([s,o], ignore_index=kw.get('ignore_index', False))

# ============================================================
# 策略参数 (全部通过网格搜索优化)
# ============================================================
SYMBOL = 'sz.399006'
NAME   = '创业板'
INITIAL_CAPITAL = 1_000_000
AREA_THRESHOLD  = 2000      # ★ DIF水上面积止盈阈值
LONG_CONFIRM    = 14        # ★ 普通突破确认天数
SHORT_CONFIRM   = 1         # ★ 放量突破确认天数
VOL_PERIOD      = 150       # ★ 成交量均线周期
VOL_MULTIPLIER  = 1.8       # ★ 放量倍数阈值
GC_CONFIRM      = 0         # ★ 水下金叉确认天数
COMMISSION      = 0.0003    # 佣金
STAMP_DUTY      = 0.0005    # 印花税(卖出)
SLIPPAGE        = 0.001     # 滑点

SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
HISTORY_FILE = os.path.join(os.path.dirname(SCRIPT_DIR), 'trade_history.csv')

# ============================================================
# 数据获取
# ============================================================
def fetch_data(days=400) -> pd.DataFrame:
    """获取创业板最新日线数据"""
    lg = bs.login()
    end = datetime.now()
    start = end - timedelta(days=days)
    rs = bs.query_history_k_data_plus(
        SYMBOL, 'date,open,high,low,close,volume,amount',
        start_date=start.strftime('%Y-%m-%d'), end_date=end.strftime('%Y-%m-%d'),
        frequency='d', adjustflag='1')
    df = rs.get_data() if rs and rs.error_code == '0' else pd.DataFrame()
    bs.logout()
    if len(df) == 0: return df

    for c in ['open','high','low','close','volume','amount']:
        df[c] = pd.to_numeric(df[c], errors='coerce')
    df['date'] = pd.to_datetime(df['date'])
    df = df.dropna(subset=['close']).sort_values('date').reset_index(drop=True)
    return df

# ============================================================
# 指标计算
# ============================================================
def calc_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """计算所有技术指标"""
    df = df.copy()
    df['MA250']   = df['close'].rolling(250).mean()
    df['AVG_VOL'] = df['volume'].rolling(VOL_PERIOD).mean()
    e12 = df['close'].ewm(span=12, adjust=False).mean()
    e26 = df['close'].ewm(span=26, adjust=False).mean()
    df['DIF'] = e12 - e26
    df['DEA'] = df['DIF'].ewm(span=9, adjust=False).mean()
    df['BAR'] = 2 * (df['DIF'] - df['DEA'])
    return df

# ============================================================
# DIF面积计算 (需要全量数据)
# ============================================================
def calc_dif_area(dif: np.ndarray) -> np.ndarray:
    """计算DIF水上累计面积"""
    areas = np.zeros(len(dif))
    a = 0
    for i in range(len(dif)):
        if dif[i] > 0: a += dif[i]
        else: a = 0
        areas[i] = a
    return areas

# ============================================================
# 持仓状态追踪
# ============================================================
class PositionTracker:
    """跟踪当前持仓状态和待确认信号"""
    def __init__(self):
        self.in_market      = False     # 是否持仓
        self.entry_date     = None      # 入场日期
        self.entry_price    = 0         # 入场价
        self.shares         = 0         # 持仓份额
        self.peak_price     = 0         # 持仓期间最高价
        self.cash           = INITIAL_CAPITAL

        # 入场跟踪
        self.state          = 'ma_entry'  # ma_entry | golden
        self.pending_buy    = 0         # 连续站上年线天数
        self.pending_sell   = 0         # 连续跌破年线天数
        self.vol_boost      = False     # 当前突破是否放量
        self.pending_gc     = 0         # 水下金叉后等年线

        # DIF面积(运行时计算)
        self.dif_area       = 0

    def reset_entry_state(self, state='ma_entry'):
        self.state = state
        self.pending_buy = 0
        self.pending_sell = 0
        self.vol_boost = False
        self.pending_gc = 0
        self.dif_area = 0

# ============================================================
# 信号引擎
# ============================================================
def check_signals(df: pd.DataFrame, tracker: PositionTracker, trade_mode=False):
    """
    逐日扫描信号, 返回最近一个交易日的操作建议.

    返回: dict {
        'date': str, 'action': 'BUY'|'SELL'|'HOLD'|'WAIT',
        'price': float, 'reason': str, 'details': str
    }
    """
    if len(df) < 250:
        return {'date': str(datetime.now().date()), 'action': 'WAIT', 'reason': '数据不足'}

    close  = df['close'].values
    ma     = df['MA250'].values
    vol    = df['volume'].values
    avg_vol = df['AVG_VOL'].values
    dif    = df['DIF'].values
    dea    = df['DEA'].values
    dates  = df['date'].values
    areas  = calc_dif_area(dif)
    n      = len(df)

    # 从最后有有效MA250的地方开始检查
    start_idx = 0
    for i in range(n):
        if not np.isnan(ma[i]): start_idx = i; break

    # 确定扫描范围: 持仓时只看最后几天, 空仓时全扫
    if tracker.in_market:
        check_start = max(start_idx + 1, n - 30)  # 最近30天
    else:
        check_start = max(1, start_idx)
    check_range = range(check_start, n)

    last_signal = {'date': str(dates[-1])[:10], 'action': 'HOLD',
                   'reason': '持仓中' if tracker.in_market else '等待入场信号'}

    for i in check_range:
        price    = float(close[i])
        date_str = str(dates[i])[:10]
        d_cur    = dif[i]
        de_cur   = dea[i]
        d_prev   = dif[i-1]
        de_prev  = dea[i-1]
        above_ma = close[i] > ma[i]
        above_ma_prev = close[i-1] > ma[i-1]
        v        = vol[i]
        av       = avg_vol[i]

        # 更新DIF面积
        if d_cur > 0:
            tracker.dif_area += d_cur
        else:
            tracker.dif_area = 0

        if tracker.in_market:
            # ---- 持仓中: 检查出场 ----
            if price > tracker.peak_price:
                tracker.peak_price = price

            # 出场A: 面积止盈
            if tracker.dif_area >= AREA_THRESHOLD:
                sell_price = price * (1 - SLIPPAGE)
                sell_value = sell_price * tracker.shares
                comm = max(sell_value * COMMISSION, 5)
                stamp = sell_value * STAMP_DUTY
                tracker.cash += sell_value - comm - stamp

                ret_pct = (sell_price - tracker.entry_price) / tracker.entry_price * 100
                last_signal = {
                    'date': date_str, 'action': 'SELL',
                    'price': round(sell_price, 4), 'reason': '面积止盈',
                    'shares': tracker.shares, 'amount': round(sell_value, 2),
                    'return_pct': round(ret_pct, 2),
                    'details': f'DIF面积={tracker.dif_area:.0f}>={AREA_THRESHOLD}, 收益{ret_pct:+.1f}%'
                }
                tracker.in_market = False; tracker.shares = 0
                tracker.reset_entry_state('golden')

            # 出场B: MA250跌破
            tracker.pending_sell = 0 if above_ma else tracker.pending_sell + 1
            if tracker.pending_sell == 1 and above_ma_prev:
                sell_price = price * (1 - SLIPPAGE)
                sell_value = sell_price * tracker.shares
                comm = max(sell_value * COMMISSION, 5)
                stamp = sell_value * STAMP_DUTY
                tracker.cash += sell_value - comm - stamp

                ret_pct = (sell_price - tracker.entry_price) / tracker.entry_price * 100
                last_signal = {
                    'date': date_str, 'action': 'SELL',
                    'price': round(sell_price, 4), 'reason': 'MA250跌破',
                    'shares': tracker.shares, 'amount': round(sell_value, 2),
                    'return_pct': round(ret_pct, 2),
                    'details': f'收盘{price:.2f}<MA250={ma[i]:.2f}, 收益{ret_pct:+.1f}%'
                }
                tracker.in_market = False; tracker.shares = 0
                tracker.reset_entry_state('ma_entry')
        else:
            # ---- 空仓中: 检查入场 ----
            should_buy = False
            buy_reason = ''

            if tracker.state == 'ma_entry':
                # 主入场: MA250突破
                if not above_ma_prev and above_ma:
                    tracker.pending_buy = 1
                    tracker.vol_boost = (not np.isnan(av) and v > av * VOL_MULTIPLIER)

                if tracker.pending_buy > 0:
                    if above_ma:
                        tracker.pending_buy += 1
                    else:
                        tracker.pending_buy = 0
                        tracker.vol_boost = False

                    confirm_needed = SHORT_CONFIRM if tracker.vol_boost else LONG_CONFIRM
                    if tracker.pending_buy >= confirm_needed + 1:
                        should_buy = True
                        buy_reason = f'放量突破{SHORT_CONFIRM}天' if tracker.vol_boost else f'MA250突破{LONG_CONFIRM}天'
                        tracker.pending_buy = 0
                        tracker.vol_boost = False
                        tracker.state = 'ma_entry'

            elif tracker.state == 'golden':
                # 回补入场: 水下金叉
                golden = (d_cur > de_cur and d_prev <= de_prev and d_cur < 0)
                if golden and tracker.pending_gc == 0:
                    tracker.pending_gc = 1

                if tracker.pending_gc > 0:
                    if above_ma:
                        tracker.pending_gc += 1
                    else:
                        tracker.pending_gc = 0
                        tracker.state = 'ma_entry'

                    if tracker.pending_gc >= GC_CONFIRM + 2:
                        should_buy = True
                        buy_reason = '水下金叉回补'
                        tracker.pending_gc = 0
                        tracker.state = 'ma_entry'

            if should_buy:
                buy_price = price * (1 + SLIPPAGE)
                raw = int(tracker.cash * 0.998 / buy_price / 100) * 100
                if raw >= 100:
                    val = buy_price * raw
                    comm = max(val * COMMISSION, 5)
                    cost = val + comm
                    if cost <= tracker.cash:
                        tracker.cash -= cost
                        tracker.shares = raw
                        tracker.in_market = True
                        tracker.entry_date = date_str
                        tracker.entry_price = buy_price
                        tracker.peak_price = buy_price
                        tracker.dif_area = 0
                        tracker.pending_sell = 0

                        last_signal = {
                            'date': date_str, 'action': 'BUY',
                            'price': round(buy_price, 4), 'reason': buy_reason,
                            'shares': raw, 'amount': round(val, 2),
                            'details': f'价格{buy_price:.2f}, {raw}股, 现金剩余{tracker.cash:,.0f}'
                        }

    return last_signal

# ============================================================
# 持仓快照
# ============================================================
def get_status(df: pd.DataFrame, tracker: PositionTracker) -> dict:
    """获取当前状态摘要"""
    last = df.iloc[-1]
    return {
        'date': str(last['date'])[:10],
        'close': float(last['close']),
        'MA250': float(last['MA250']),
        'DIF': float(last['DIF']),
        'DEA': float(last['DEA']),
        'DIF_area': round(tracker.dif_area, 1),
        'above_ma': float(last['close']) > float(last['MA250']),
        'in_market': tracker.in_market,
        'state': tracker.state,
        'pending_buy_days': tracker.pending_buy,
        'pending_sell_days': tracker.pending_sell,
        'entry_price': tracker.entry_price if tracker.in_market else 0,
        'peak_price': tracker.peak_price if tracker.in_market else 0,
        'position_value': round(tracker.shares * float(last['close']), 2) if tracker.in_market else 0,
        'total_value': round(tracker.cash + (tracker.shares * float(last['close']) if tracker.in_market else 0), 2),
    }

# ============================================================
# 历史记录
# ============================================================
def log_trade(signal: dict):
    """记录交易到CSV"""
    file_exists = os.path.exists(HISTORY_FILE)
    row = {
        '时间': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        '信号日期': signal.get('date', ''),
        '操作': signal.get('action', ''),
        '价格': signal.get('price', 0),
        '原因': signal.get('reason', ''),
        '份额': signal.get('shares', 0),
        '金额': signal.get('amount', 0),
        '收益率': signal.get('return_pct', ''),
        '说明': signal.get('details', ''),
    }
    df = pd.DataFrame([row])
    df.to_csv(HISTORY_FILE, mode='a', header=not file_exists, index=False, encoding='utf-8-sig')

def show_history():
    """查看历史记录"""
    if not os.path.exists(HISTORY_FILE):
        print('暂无交易记录')
        return
    df = pd.read_csv(HISTORY_FILE)
    print(f'\n交易历史 ({len(df)} 条)')
    print(f'{"="*80}')
    print(df.tail(20).to_string(index=False))

# ============================================================
# Main
# ============================================================
def main():
    import argparse
    parser = argparse.ArgumentParser(description='创业板混合策略 - 每日运行')
    parser.add_argument('--trade', action='store_true', help='实盘模式(记录交易)')
    parser.add_argument('--history', action='store_true', help='查看历史记录')
    args = parser.parse_args()

    if args.history:
        show_history()
        return

    # 加载数据
    print(f'[{datetime.now().strftime("%H:%M:%S")}] 加载创业板数据...')
    df = fetch_data(400)
    if len(df) == 0:
        print('数据加载失败')
        return
    df = calc_indicators(df)

    # 检查信号
    tracker = PositionTracker()

    # 恢复持仓状态 (仅实盘模式)
    if args.trade and os.path.exists(HISTORY_FILE):
        hist = pd.read_csv(HISTORY_FILE)
        buys  = hist[hist['操作'] == 'BUY']
        sells = hist[hist['操作'] == 'SELL']
        if len(buys) > len(sells):
            last_buy = buys.iloc[-1]
            tracker.in_market = True
            tracker.entry_date = last_buy['信号日期']
            tracker.entry_price = last_buy['价格']
            tracker.shares = last_buy['份额']
            # 从历史买入后扫描最新数据，恢复峰值和DIF面积
            buy_date = pd.Timestamp(last_buy['信号日期'])
            mask = df['date'] >= buy_date
            for i in range(len(df)):
                if df.iloc[i]['date'] >= buy_date:
                    p = float(df.iloc[i]['close'])
                    if p > tracker.peak_price: tracker.peak_price = p
            tracker.state = 'ma_entry'

    # 只扫描最近信号 (空仓时全扫, 持仓时只检查最近一天)
    signal = check_signals(df, tracker, args.trade)

    # 打印状态
    status = get_status(df, tracker)
    print(f'\n{"="*60}')
    print(f'  创业板 混合策略 — 每日信号检查')
    print(f'  {datetime.now().strftime("%Y-%m-%d %H:%M")}')
    print(f'{"="*60}')
    print(f'  最新日期: {status["date"]}')
    print(f'  收盘价:   {status["close"]:.2f}')
    print(f'  MA250:    {status["MA250"]:.2f}')
    print(f'  年线上方: {"是" if status["above_ma"] else "否"}')
    print(f'  DIF:      {status["DIF"]:.1f}')
    print(f'  DEA:      {status["DEA"]:.1f}')
    print(f'  DIF面积:  {status["DIF_area"]:.0f} / {AREA_THRESHOLD}')
    print(f'  持仓状态: {"持仓中" if status["in_market"] else "空仓"}')
    if status['in_market']:
        dd = (status['close'] - status['peak_price']) / status['peak_price'] * 100 if status['peak_price'] > 0 else 0
        print(f'  入场价:   {status["entry_price"]:.2f}  峰值: {status["peak_price"]:.2f}  回撤: {dd:+.1f}%')
        print(f'  持仓市值: ¥{status["position_value"]:,.0f}')
    print(f'  总资产:   ¥{status["total_value"]:,.0f}')
    if status['in_market']:
        dd_pct = (status['close'] - status['peak_price']) / status['peak_price'] * 100 if status['peak_price'] > 0 else 0
        print(f'  当前回撤:  {dd_pct:+.1f}% (止损线 -8%)')
        print(f'  MA250状态: {"上方" if status["above_ma"] else "跌破! 注意出场"}')
    else:
        print(f'  待确认买入: 已站{status["pending_buy_days"]}天 / 需要{SHORT_CONFIRM if tracker.vol_boost else LONG_CONFIRM}天')
    print(f'  入场模式: {status["state"]}')
    print(f'{"="*60}')

    # 信号 (today only)
    today_str = datetime.now().strftime('%Y-%m-%d')
    if signal.get('action') in ('BUY','SELL') and signal.get('date','') != today_str:
        signal['action'] = 'HOLD'
        signal['reason'] = '持仓中' if status['in_market'] else '等待信号'
    print(f'\n  >>> 今日操作: {signal["action"]} | {signal["reason"]}')
    if signal.get('details') and signal.get('date','') == today_str:
        print(f'  >>> {signal["details"]}')
    print()

    # 记录
    if args.trade and signal['action'] in ('BUY', 'SELL'):
        log_trade(signal)
        print(f'  [已记录] {signal["action"]} @ {signal.get("price",0)}')

if __name__ == '__main__':
    main()
