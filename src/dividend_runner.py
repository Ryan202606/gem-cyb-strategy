#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
510880 红利ETF — 状态机MACD波段策略 每日运行
==============================================
状态机: HOLDING_BASE → WAITING_BUY → HOLDING_SWING → 循环
  ① Day1全仓买入 → HOLDING_BASE
  ② 死叉 → 卖100%底仓换现金 → WAITING_BUY
  ③ 水上金叉追入(优先) / 绿柱5条件入场 → HOLDING_SWING
  ④ 红柱4条件止盈 / 安全兜底 → WAITING_BUY → 循环
"""
import sys, io, os
from datetime import datetime, timedelta
from typing import Optional, List, Tuple
if sys.stdout.encoding != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
import pandas as pd
import numpy as np
if not hasattr(pd.DataFrame, 'append'):
    pd.DataFrame.append = lambda s, o, **kw: pd.concat([s, o], ignore_index=kw.get('ignore_index', False))

# ============================================================
# 配置
# ============================================================
SYMBOL = 'sh510880'
NAME = '红利ETF'
COMM = 0.00015
STAMP = 0.0
SLIP = 0.0
INITIAL_CAPITAL = 1_000_000
SWING_PCT = 1.0        # 死叉卖出100%底仓
SWING_BUY_PCT = 1.0     # 100%现金买入波段
IDLE_CASH_RATE = 0.025  # 闲置资金年化2.5%
STOP_LOSS = 0.10        # 安全兜底: 持仓>1年+回撤10%
CLUSTER_SKIP = 5        # 每簇跳过前5根
GREEN_MEAN_WINDOW = 90
GREEN_MEAN_MIN = 30

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
HF = os.path.join(os.path.dirname(SCRIPT_DIR), 'dividend_trade_history.csv')

# 状态机
STATE_HOLDING_BASE = 'HOLDING_BASE'
STATE_WAITING_BUY = 'WAITING_BUY'
STATE_HOLDING_SWING = 'HOLDING_SWING'

# ============================================================
# 数据加载 & 指标计算
# ============================================================
def load_data(days=800):
    """加载510880数据，计算所有技术指标"""
    df = None
    try:
        import efinance as ef
        df = ef.stock.get_quote_history('510880', beg='20100101', end=datetime.now().strftime('%Y%m%d'))
        df = df.rename(columns={'日期': 'date', '开盘': 'open', '最高': 'high', '最低': 'low', '收盘': 'close', '成交量': 'volume', '成交额': 'amount'})
    except:
        pass
    if df is None or len(df) < 500:
        csv_path = os.path.join(SCRIPT_DIR, '510880.csv')
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
    if df is None or len(df) == 0:
        return pd.DataFrame()
    df['date'] = pd.to_datetime(df['date'])
    for c in ['open', 'high', 'low', 'close', 'volume', 'amount']:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')
    df = df.dropna(subset=['close']).sort_values('date').reset_index(drop=True)

    # 均线
    df['MA250'] = df['close'].rolling(250).mean()

    # MACD
    ema12 = df['close'].ewm(span=12, adjust=False).mean()
    ema26 = df['close'].ewm(span=26, adjust=False).mean()
    df['DIF'] = ema12 - ema26
    df['DEA'] = df['DIF'].ewm(span=9, adjust=False).mean()
    df['MACD'] = (df['DIF'] - df['DEA']) * 2

    # 金叉/死叉
    df['golden_cross'] = (df['DIF'] > df['DEA']) & (df['DIF'].shift(1) <= df['DEA'].shift(1))
    df['death_cross'] = (df['DIF'] < df['DEA']) & (df['DIF'].shift(1) >= df['DEA'].shift(1))

    # 近90日绿柱均值
    green_abs = df['MACD'].where(df['MACD'] < 0, np.nan).abs()
    df['green_mean90'] = green_abs.rolling(GREEN_MEAN_WINDOW, min_periods=GREEN_MEAN_MIN).mean()

    df = df.dropna(subset=['DIF', 'DEA', 'MACD', 'MA250']).reset_index(drop=True)
    return df

# ============================================================
# 柱子簇识别
# ============================================================
def identify_clusters(df: pd.DataFrame) -> List[dict]:
    clusters = []
    if len(df) == 0:
        return clusters
    macd_vals = df['MACD'].values
    cluster_start = 0
    for i in range(1, len(macd_vals)):
        prev_color = 'green' if macd_vals[i-1] < 0 else 'red'
        curr_color = 'green' if macd_vals[i] < 0 else 'red'
        if curr_color != prev_color:
            clr = 'green' if macd_vals[i-1] < 0 else 'red'
            area = np.abs(macd_vals[cluster_start:i]).sum()
            clusters.append({'color': clr, 'start_idx': cluster_start, 'end_idx': i-1, 'count': i-cluster_start, 'area': area})
            cluster_start = i
    if cluster_start < len(macd_vals):
        clr = 'green' if macd_vals[cluster_start] < 0 else 'red'
        area = np.abs(macd_vals[cluster_start:]).sum()
        clusters.append({'color': clr, 'start_idx': cluster_start, 'end_idx': len(macd_vals)-1, 'count': len(macd_vals)-cluster_start, 'area': area})
    return clusters

# ============================================================
# K线形态过滤
# ============================================================
def kline_pattern_filter(df: pd.DataFrame, idx: int) -> Tuple[bool, str]:
    row = df.iloc[idx]
    o, h, l, c = row['open'], row['high'], row['low'], row['close']
    amplitude = (h - l) / o if o > 0 else 0
    if amplitude > 0.03:
        return True, f'豁免(振幅{amplitude*100:.1f}%)'
    body_pct = abs(c - o) / o if o > 0 else 0
    if body_pct < 0.003:
        return False, f'小实体({body_pct*100:.2f}%)'
    body_range_ratio = abs(c - o) / (h - l) if (h - l) > 0 else 1
    if body_range_ratio < 0.25:
        return False, f'纺锤线({body_range_ratio*100:.1f}%)'
    return True, f'通过'

# ============================================================
# 金叉/死叉检测
# ============================================================
def has_golden_cross_between(df: pd.DataFrame, start_idx: int, end_idx: int) -> bool:
    for i in range(start_idx + 1, min(end_idx + 1, len(df))):
        if df['golden_cross'].iloc[i]:
            return True
    return False

def has_death_cross_between(df: pd.DataFrame, start_idx: int, end_idx: int) -> bool:
    for i in range(start_idx + 1, min(end_idx + 1, len(df))):
        if df['death_cross'].iloc[i]:
            return True
    return False

# ============================================================
# 柱子大小过滤
# ============================================================
def check_bar_size_filter(df: pd.DataFrame, cluster: dict, idx_b: int,
                          green_clusters: List[dict]) -> Tuple[bool, str]:
    green_mean90 = df['green_mean90'].iloc[idx_b]
    if pd.isna(green_mean90) or green_mean90 == 0:
        return True, '豁免(无90日均值)'
    macd_b = abs(df['MACD'].iloc[idx_b])
    cluster_macd_max = np.abs(df['MACD'].iloc[cluster['start_idx']:cluster['end_idx']+1]).max()
    if cluster_macd_max > green_mean90 * 0.5:
        return True, f'豁免1(簇最大>{green_mean90*0.5:.4f})'
    green_only = [c for c in green_clusters if c['color'] == 'green' and c['end_idx'] < idx_b]
    if len(green_only) >= 3:
        last3 = green_only[-3:]
        if last3[0]['area'] > last3[1]['area'] > last3[2]['area']:
            return True, '豁免2(面积递减)'
    if macd_b < green_mean90 * 0.3:
        return False, f'过滤(|MACD_B|{macd_b:.4f}<{green_mean90*0.3:.4f})'
    return True, '通过'

# ============================================================
# 绿柱入场 — 5条件
# ============================================================
def check_long_entry(df: pd.DataFrame, idx_b: int,
                     clusters: List[dict],
                     green_clusters: List[dict]) -> Tuple[bool, str, float]:
    macd_b = df['MACD'].iloc[idx_b]
    if macd_b >= 0:
        return False, '非绿柱', 0
    current_cluster = None
    for c in clusters:
        if c['start_idx'] <= idx_b <= c['end_idx'] and c['color'] == 'green':
            current_cluster = c
            break
    if current_cluster is None:
        return False, '未找到绿柱簇', 0
    if idx_b - current_cluster['start_idx'] < CLUSTER_SKIP:
        return False, '前5根跳过', 0
    idx_a = idx_b - 1
    macd_a = df['MACD'].iloc[idx_a]
    # 条件①: 柱子缩小
    if abs(macd_a) <= abs(macd_b):
        return False, '柱未缩小', 0
    # 条件②: DIF深度够
    dif_b = df['DIF'].iloc[idx_b]
    dea_b = df['DEA'].iloc[idx_b]
    if dif_b >= 0:
        return False, 'DIF>=0', 0
    if abs(dif_b) <= abs(macd_b) * 0.50:
        return False, 'DIF不够深', 0
    if abs(dea_b) <= abs(macd_b) * 0.30:
        return False, 'DEA不够深', 0
    # 条件③: K线形态 (金叉跳过)
    if has_golden_cross_between(df, idx_a, idx_b):
        return True, f'簇内金叉入场(DIF{dif_b:.4f})', float(df['close'].iloc[idx_b])
    pass_a, reason_a = kline_pattern_filter(df, idx_a)
    pass_b, reason_b = kline_pattern_filter(df, idx_b)
    if not pass_a:
        return False, f'A柱K线({reason_a})', 0
    if not pass_b:
        return False, f'B柱K线({reason_b})', 0
    # 条件④: 柱子大小
    pass_size, reason_size = check_bar_size_filter(df, current_cluster, idx_b, green_clusters)
    if not pass_size:
        return False, f'柱子大小({reason_size})', 0
    # 条件⑤: 价格突破
    close_b = df['close'].iloc[idx_b]
    high_a = df['high'].iloc[idx_a]
    if close_b <= high_a * 1.003:
        return False, '价格未突破', 0
    return True, f'5条件通过', close_b

# ============================================================
# 红柱止盈 — 4条件
# ============================================================
def check_short_exit(df: pd.DataFrame, idx_b: int,
                     clusters: List[dict]) -> Tuple[bool, str, float]:
    macd_b = df['MACD'].iloc[idx_b]
    if macd_b <= 0:
        return False, '非红柱', 0
    current_cluster = None
    for c in clusters:
        if c['start_idx'] <= idx_b <= c['end_idx'] and c['color'] == 'red':
            current_cluster = c
            break
    if current_cluster is None:
        return False, '未找到红柱簇', 0
    idx_a = idx_b - 1
    macd_a = df['MACD'].iloc[idx_a]
    # 条件①: 柱子缩小
    if macd_a <= macd_b:
        return False, '柱未缩小', 0
    # 条件②: DIF够高
    dif_b = df['DIF'].iloc[idx_b]
    if dif_b <= 0:
        return False, 'DIF<=0', 0
    if dif_b <= abs(macd_b) * 0.75:
        return False, 'DIF不够高', 0
    # 条件③: K线形态 (死叉中止)
    if has_death_cross_between(df, idx_a, idx_b):
        return False, '死叉中止', 0
    pass_a, reason_a = kline_pattern_filter(df, idx_a)
    pass_b, reason_b = kline_pattern_filter(df, idx_b)
    if not pass_a:
        return False, f'A柱K线({reason_a})', 0
    if not pass_b:
        return False, f'B柱K线({reason_b})', 0
    # 条件④: 价格跌破
    close_b = df['close'].iloc[idx_b]
    low_a = df['low'].iloc[idx_a]
    if close_b >= low_a * 0.999:
        return False, '价格未跌破', 0
    return True, f'红柱止盈', close_b

# ============================================================
# 水上金叉追入
# ============================================================
def check_golden_chase_entry(df: pd.DataFrame, idx: int) -> Tuple[bool, str, float]:
    if not df['golden_cross'].iloc[idx]:
        return False, '', 0
    dif = df['DIF'].iloc[idx]
    dea = df['DEA'].iloc[idx]
    if dif <= 0:
        return False, f'DIF≤0({dif:.4f})', 0
    if dea <= 0:
        return False, f'DEA≤0({dea:.4f})', 0
    return True, f'水上金叉(DIF{dif:.4f},DEA{dea:.4f})', float(df['close'].iloc[idx])

# ============================================================
# 计算交易成本
# ============================================================
def calc_buy_cost(price: float, shares: int) -> Tuple[float, float]:
    value = price * shares
    commission = max(value * COMM, 5)
    return commission, value + commission

def calc_sell_proceeds(price: float, shares: int) -> Tuple[float, float, float]:
    value = price * shares
    commission = max(value * COMM, 5)
    stamp = value * STAMP
    return commission, stamp, value - commission - stamp

# ============================================================
# 日志交易记录
# ============================================================
def log_trade(sig):
    fe = os.path.exists(HF)
    row = {'time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'), 'sig_date': sig.get('date', ''),
           'action': sig.get('action', ''), 'price': sig.get('price', 0),
           'reason': sig.get('reason', ''), 'shares': sig.get('shares', 0),
           'amount': sig.get('amount', 0), 'ret': sig.get('ret', '')}
    pd.DataFrame([row]).to_csv(HF, mode='a', header=not fe, index=False, encoding='utf-8-sig')

# ============================================================
# 主逻辑: 状态机每日检查
# ============================================================
def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--trade', action='store_true')
    p.add_argument('--history', action='store_true')
    args = p.parse_args()

    if args.history:
        if os.path.exists(HF):
            df = pd.read_csv(HF)
            print(f'红利ETF交易记录({len(df)}条):')
            print(df.tail(20).to_string(index=False))
        else:
            print('无记录')
        return

    df = load_data()
    if len(df) == 0:
        print('数据加载失败')
        return

    n = len(df)
    close = df['close'].values
    clusters = identify_clusters(df)
    green_clusters = [c for c in clusters if c['color'] == 'green']

    # ── 状态变量 ──
    # 每日运行: 默认空仓持有现金(WAITING_BUY), 等绿柱/金叉入场信号。
    # 不做"Day1自动全仓买入"——那是回测专属起点, 每日脚本若自动建仓会在
    # 每次运行都重放最近60天, 产生虚假的"持仓"状态。
    cash = 0.0
    base_shares = 0
    swing_shares = 0
    swing_entry_price = 0.0
    swing_peak = 0.0
    swing_entry_idx = 0
    swing_cash = INITIAL_CAPITAL
    state = STATE_WAITING_BUY
    golden_chase_active = True
    golden_chase_used = False

    # ── 从历史恢复状态 ──
    last_trade_date = None  # track last trade date for forward scanning
    if args.trade and os.path.exists(HF):
        hist = pd.read_csv(HF)
        for _, r in hist.iterrows():
            action = str(r['action']).strip()
            reason = str(r.get('reason', '')).strip()
            d = str(r.get('sig_date', ''))[:10]
            if d:
                last_trade_date = d
            if action == '买入':
                if '金叉' in reason or '绿柱' in reason:
                    swing_shares = int(r['shares'])
                    swing_entry_price = float(r['price'])
                    swing_peak = swing_entry_price
                    swing_cash = 0.0
                    state = STATE_HOLDING_SWING
                    golden_chase_active = False
                    golden_chase_used = '金叉' in reason
            elif action == '卖出':
                if '清仓' in reason:
                    # 旧策略清仓 → 空仓(等入场); 不改变新策略基准资金
                    swing_shares = 0
                    swing_entry_price = 0
                    swing_peak = 0
                    swing_entry_idx = 0
                    state = STATE_WAITING_BUY
                    golden_chase_active = True
                    golden_chase_used = False
                elif '止盈' in reason or '兜底' in reason:
                    swing_shares = 0
                    swing_entry_price = 0
                    swing_peak = 0
                    swing_entry_idx = 0
                    swing_cash += float(r['amount'])
                    state = STATE_WAITING_BUY
                    golden_chase_active = True
                    golden_chase_used = False
            # 旧策略英文动作(BUY/SELL/HOLD) 或未知动作 → 忽略

    # ── 确定扫描起点 ──
    sig = {'action': '持有', 'reason': '监控中'}
    ch = min(60, n)
    today_str = datetime.now().strftime('%Y-%m-%d')

    # If we restored from history, only scan from after the last trade
    scan_start = max(1, n - ch)
    if last_trade_date:
        for i in range(n):
            if str(df['date'].iloc[i])[:10] > last_trade_date:
                scan_start = i
                break

    for i in range(scan_start, n):
        price = float(close[i])
        ds = str(df['date'].iloc[i])[:10]
        if ds > today_str:
            continue  # skip future dates

        # ── 安全兜底检查 (HOLDING_SWING) ──
        if state == STATE_HOLDING_SWING and swing_shares > 0:
            if price > swing_peak:
                swing_peak = price
            hold_days = i - swing_entry_idx
            if hold_days > 252 and swing_peak > 0 and (price - swing_peak) / swing_peak <= -STOP_LOSS:
                sp = price * (1 - SLIP) if SLIP > 0 else price
                sc, ss, proceeds = calc_sell_proceeds(sp, swing_shares)
                swing_cash += proceeds
                ret = (sp - swing_entry_price) / swing_entry_price * 100
                sig = {'date': ds, 'action': '卖出', 'price': round(sp, 3), 'reason': f'安全兜底(持仓{hold_days}天回撤{(price-swing_peak)/swing_peak*100:.1f}%)',
                       'shares': swing_shares, 'amount': round(proceeds, 2), 'ret': round(ret, 2)}
                swing_shares = 0; swing_entry_price = 0; swing_peak = 0; swing_entry_idx = 0
                golden_chase_active = True; golden_chase_used = False
                state = STATE_WAITING_BUY
                continue

        # ================================================================
        # WAITING_BUY: 等入场信号 (空仓持有现金)
        # ================================================================
        if state == STATE_WAITING_BUY:
            if swing_cash <= 0:
                continue

            # 水上金叉追入条件检查: 需要DIF>0且DEA>0
            if golden_chase_active and not golden_chase_used:
                if df['DIF'].iloc[i] < 0 or df['DEA'].iloc[i] < 0:
                    golden_chase_active = False

            bought = False

            # 优先1: 水上金叉追入
            if golden_chase_active and not golden_chase_used:
                triggered, reason, entry_p = check_golden_chase_entry(df, i)
                if triggered:
                    bp = entry_p * (1 + SLIP) if SLIP > 0 else entry_p
                    buy_shares = int(swing_cash * SWING_BUY_PCT / bp / 100) * 100
                    if buy_shares >= 100:
                        comm, cost = calc_buy_cost(bp, buy_shares)
                        while cost > swing_cash and buy_shares >= 200:
                            buy_shares -= 100
                            comm, cost = calc_buy_cost(bp, buy_shares)
                        if cost <= swing_cash:
                            swing_cash -= cost
                            swing_shares = buy_shares
                            swing_entry_price = bp
                            swing_peak = bp
                            swing_entry_idx = i
                            sig = {'date': ds, 'action': '买入', 'price': round(bp, 3),
                                   'reason': f'水上金叉追入:{reason}', 'shares': buy_shares,
                                   'amount': round(bp * buy_shares, 2)}
                            golden_chase_used = True
                            golden_chase_active = False
                            state = STATE_HOLDING_SWING
                            bought = True

            # 优先2: 绿柱5条件入场
            if not bought:
                triggered, reason, entry_p = check_long_entry(df, i, clusters, green_clusters)
                if triggered:
                    bp = entry_p * (1 + SLIP) if SLIP > 0 else entry_p
                    buy_shares = int(swing_cash * SWING_BUY_PCT / bp / 100) * 100
                    if buy_shares >= 100:
                        comm, cost = calc_buy_cost(bp, buy_shares)
                        while cost > swing_cash and buy_shares >= 200:
                            buy_shares -= 100
                            comm, cost = calc_buy_cost(bp, buy_shares)
                        if cost <= swing_cash:
                            swing_cash -= cost
                            swing_shares = buy_shares
                            swing_entry_price = bp
                            swing_peak = bp
                            swing_entry_idx = i
                            sig = {'date': ds, 'action': '买入', 'price': round(bp, 3),
                                   'reason': f'MACD绿柱入场:{reason}', 'shares': buy_shares,
                                   'amount': round(bp * buy_shares, 2)}
                            golden_chase_active = False
                            golden_chase_used = False
                            state = STATE_HOLDING_SWING
                            bought = True

        # ================================================================
        # HOLDING_SWING: 等红柱止盈
        # ================================================================
        elif state == STATE_HOLDING_SWING:
            if swing_shares <= 0:
                state = STATE_WAITING_BUY
                continue

            if price > swing_peak:
                swing_peak = price

            triggered, reason, exit_p = check_short_exit(df, i, clusters)
            if triggered:
                sp = exit_p * (1 - SLIP) if SLIP > 0 else exit_p
                sc, ss, proceeds = calc_sell_proceeds(sp, swing_shares)
                swing_cash += proceeds
                ret = (sp - swing_entry_price) / swing_entry_price * 100
                sig = {'date': ds, 'action': '卖出', 'price': round(sp, 3),
                       'reason': f'红柱止盈:{reason}', 'shares': swing_shares,
                       'amount': round(proceeds, 2), 'ret': round(ret, 2)}
                swing_shares = 0; swing_entry_price = 0; swing_peak = 0; swing_entry_idx = 0
                golden_chase_active = True; golden_chase_used = False
                state = STATE_WAITING_BUY
                continue

        # ── 闲置资金(swing_cash)按国债利率每日生息 ──
        if swing_cash > 0 and IDLE_CASH_RATE > 0:
            daily_rate = (1 + IDLE_CASH_RATE) ** (1/252) - 1
            swing_cash *= (1 + daily_rate)

    # ── 输出 ──
    last = df.iloc[-1]
    last_close = float(last['close'])
    last_ma250 = float(last['MA250'])
    last_dif = float(last['DIF'])
    last_dea = float(last['DEA'])
    last_macd = float(last['MACD'])

    # 当前权益
    holdings_val = (base_shares + swing_shares) * last_close
    total_equity = cash + swing_cash + holdings_val

    act = sig.get('action', '持有')
    if act in ('买入', '卖出') and sig.get('date', '') != today_str:
        act = '持有'
        sig['reason'] = '监控中'

    # Summary output
    status_map = {STATE_WAITING_BUY: '空仓(等入场)', STATE_HOLDING_SWING: '持仓(波段)'}
    status = status_map.get(state, '空仓')
    advice = act if act != '持有' else '无操作'
    reason = sig.get('reason', '-')

    print(f'红利[{NAME}]-----{status}-----{advice}-----{reason}')
    if base_shares > 0:
        print(f'  底仓: {base_shares:,}股')
    if swing_shares > 0:
        dd_val = (last_close - swing_peak) / swing_peak * 100 if swing_peak > 0 else 0
        hold_days = (datetime.now() - df['date'].iloc[swing_entry_idx]).days if swing_entry_idx > 0 else 0
        print(f'  波段: {swing_shares:,}股')
        print(f'  入场价: {swing_entry_price:.3f}')
        print(f'  回撤: {dd_val:+.2f}%')
        print(f'  持仓天数: {hold_days}天')
    if swing_cash > 0:
        print(f'  可用现金: {swing_cash:,.0f}')
    print(f'  总资产: {total_equity:,.0f}')
    print(f'  收盘: {last_close:.3f}')
    print(f'  年线: {last_ma250:.3f}')
    print(f'  DIF: {last_dif:.4f}')
    print(f'  DEA: {last_dea:.4f}')
    print(f'  MACD: {last_macd:.4f}')
    print(f'  死叉信号: {"是" if last["death_cross"] else "否"}')
    print(f'  金叉信号: {"是" if last["golden_cross"] else "否"}')

    # 金叉追入状态
    if state == STATE_WAITING_BUY and golden_chase_active and not golden_chase_used:
        print(f'  金叉追入: 待命中 (DIF>0 & DEA>0 时激活)')
    elif state == STATE_WAITING_BUY and golden_chase_active and golden_chase_used:
        print(f'  金叉追入: 已使用')

    if args.trade and act in ('买入', '卖出'):
        log_trade(sig)
        print(f'  [已记录]')


if __name__ == '__main__':
    main()
