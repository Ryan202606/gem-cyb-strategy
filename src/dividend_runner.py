#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
510880 红利ETF — 每日自动运行脚本 (MA250底仓 + MACD波段)
"""
import sys,io,os
from datetime import datetime,timedelta
if sys.stdout.encoding!='utf-8': sys.stdout=io.TextIOWrapper(sys.stdout.buffer,encoding='utf-8',errors='replace')
import pandas as pd,numpy as np
if not hasattr(pd.DataFrame,'append'): pd.DataFrame.append=lambda s,o,**kw:pd.concat([s,o],ignore_index=kw.get('ignore_index',False))

SYMBOL='sh510880';NAME='红利ETF';COMM=0.00015;STAMP=0;SLIP=0
MA_PERIOD=250;SWING_PCT=1.0;STOP_LOSS=0.05
SCRIPT_DIR=os.path.dirname(os.path.abspath(__file__))
HF=os.path.join(os.path.dirname(SCRIPT_DIR),'dividend_trade_history.csv')

def load_data():
    """加载510880数据(优先在线,失败则用本地CSV)"""
    df=None
    try:
        import efinance as ef
        df=ef.stock.get_quote_history('510880',beg='20100101',end=datetime.now().strftime('%Y%m%d'))
        df=df.rename(columns={'日期':'date','开盘':'open','最高':'high','最低':'low','收盘':'close','成交量':'volume','成交额':'amount'})
    except: pass
    if df is None or len(df)<500:
        csv_path=os.path.join(SCRIPT_DIR,'510880.csv')
        if os.path.exists(csv_path): df=pd.read_csv(csv_path)
    if df is None or len(df)==0: return pd.DataFrame()
    df['date']=pd.to_datetime(df['date'])
    for c in ['open','high','low','close','volume','amount']:
        if c in df.columns: df[c]=pd.to_numeric(df[c],errors='coerce')
    df=df.dropna(subset=['close']).sort_values('date').reset_index(drop=True)
    df['MA250']=df['close'].rolling(MA_PERIOD).mean()
    e12=df['close'].ewm(span=12,adjust=False).mean();e26=df['close'].ewm(span=26,adjust=False).mean()
    df['DIF']=e12-e26;df['DEA']=df['DIF'].ewm(span=9,adjust=False).mean();df['MACD']=(df['DIF']-df['DEA'])*2
    df['golden']=(df['DIF']>df['DEA'])&(df['DIF'].shift(1)<=df['DEA'].shift(1))
    df['death']=(df['DIF']<df['DEA'])&(df['DIF'].shift(1)>=df['DEA'].shift(1))
    green_abs=df['MACD'].where(df['MACD']<0,np.nan).abs()
    df['green90']=green_abs.rolling(90,min_periods=30).mean()
    df=df.dropna(subset=['MA250','DIF','DEA']).reset_index(drop=True)
    return df

def check_long(df,idx,clusters,green_clusters):
    """检查做多入场"""
    macd_b=df['MACD'].iloc[idx]
    if macd_b>=0: return False,'非绿柱',0
    cl=None
    for c in clusters:
        if c['start']<=idx<=c['end'] and c['color']=='green': cl=c;break
    if cl is None: return False,'无簇',0
    if idx-cl['start']<5: return False,'前5根',0
    idx_a=idx-1;macd_a=df['MACD'].iloc[idx_a]
    if abs(macd_a)<=abs(macd_b): return False,'柱未缩小',0
    dif_b=df['DIF'].iloc[idx];dea_b=df['DEA'].iloc[idx]
    if dif_b>=0: return False,'DIF>=0',0
    if abs(dif_b)<=abs(macd_b)*0.50: return False,'DIF不够深',0
    if abs(dea_b)<=abs(macd_b)*0.30: return False,'DEA不够深',0
    # Check golden cross in cluster
    has_gc=False
    for j in range(idx_a,idx+1):
        if df['golden'].iloc[j]: has_gc=True
    if has_gc: return True,'簇内金叉',float(df['close'].iloc[idx])
    # K-line filter (simplified)
    o=df['open'].iloc[idx];h=df['high'].iloc[idx];l=df['low'].iloc[idx];c2=df['close'].iloc[idx]
    amp=(h-l)/o if o>0 else 0
    if abs(c2-o)/o<0.003 and amp<0.03: return False,'小实体',0
    # bar size filter
    gm90=df['green90'].iloc[idx]
    if not pd.isna(gm90) and gm90>0:
        cluster_max=np.abs(df['MACD'].iloc[cl['start']:cl['end']+1]).max()
        if cluster_max<=gm90*0.5 and abs(macd_b)<gm90*0.3: return False,'柱太小',0
    # price breakthrough
    close_b=c2;high_a=df['high'].iloc[idx_a]
    if close_b<=high_a*1.003: return False,'价格未突破',0
    return True,'入场',close_b

def check_short(df,idx,clusters):
    """检查止盈出场"""
    macd_b=df['MACD'].iloc[idx]
    if macd_b<=0: return False,'非红柱',0
    cl=None
    for c in clusters:
        if c['start']<=idx<=c['end'] and c['color']=='red': cl=c;break
    if cl is None: return False,'无簇',0
    idx_a=idx-1;macd_a=df['MACD'].iloc[idx_a]
    if macd_a<=macd_b: return False,'柱未缩小',0
    dif_b=df['DIF'].iloc[idx]
    if dif_b<=0: return False,'DIF<=0',0
    if dif_b<=abs(macd_b)*0.75: return False,'DIF不够高',0
    has_dc=False
    for j in range(idx_a,idx+1):
        if df['death'].iloc[j]: has_dc=True
    if has_dc: return False,'死叉中止',0
    o=df['open'].iloc[idx];c2=df['close'].iloc[idx];h=df['high'].iloc[idx];l=df['low'].iloc[idx]
    amp=(h-l)/o if o>0 else 0
    if abs(c2-o)/o<0.003 and amp<0.03: return False,'小实体',0
    close_b=c2;low_a=df['low'].iloc[idx_a]
    if close_b>=low_a*0.999: return False,'价格未跌破',0
    return True,'止盈',close_b

def identify_clusters(macd_vals):
    clusters=[];cs=0
    for i in range(1,len(macd_vals)):
        pc='green' if macd_vals[i-1]<0 else 'red';cc='green' if macd_vals[i]<0 else 'red'
        if cc!=pc:
            clr='green' if macd_vals[i-1]<0 else 'red'
            area=np.abs(macd_vals[cs:i]).sum()
            clusters.append({'color':clr,'start':cs,'end':i-1,'count':i-cs,'area':area});cs=i
    if cs<len(macd_vals):
        clr='green' if macd_vals[cs]<0 else 'red'
        area=np.abs(macd_vals[cs:]).sum()
        clusters.append({'color':clr,'start':cs,'end':len(macd_vals)-1,'count':len(macd_vals)-cs,'area':area})
    return clusters

def gs2(df,pos):
    last=df.iloc[-1];base=pos.get('base',0);swing=pos.get('swing',0)
    tv=pos.get('cash',1_000_000)+(base+swing)*float(last['close'])
    return {'date':str(last['date'])[:10],'close':float(last['close']),'MA250':float(last['MA250']),'DIF':float(last['DIF']),'DEA':float(last['DEA']),'MACD':float(last['MACD']),'am':float(last['close'])>float(last['MA250']),'base':base,'swing':swing,'tv':round(tv,2)}

def lt2(sig):
    fe=os.path.exists(HF)
    row={'time':datetime.now().strftime('%Y-%m-%d %H:%M:%S'),'sig_date':sig.get('date',''),'action':sig.get('action',''),'price':sig.get('price',0),'reason':sig.get('reason',''),'shares':sig.get('shares',0),'amount':sig.get('amount',0)}
    pd.DataFrame([row]).to_csv(HF,mode='a',header=not fe,index=False,encoding='utf-8-sig')

def main():
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--trade',action='store_true');p.add_argument('--history',action='store_true');args=p.parse_args()
    if args.history:
        if os.path.exists(HF): df=pd.read_csv(HF);print('红利ETF历史交易 ('+str(len(df))+'条)');print(df.tail(20).to_string(index=False))
        else: print('无历史')
        return

    print(datetime.now().strftime('%Y-%m-%d %H:%M')+'  '+NAME+' MACD波段策略')
    print('数据加载中...')
    df=load_data()
    if len(df)==0: print('数据加载失败');return

    macd_vals=df['MACD'].values;clusters=identify_clusters(macd_vals)
    green_clusters=[c for c in clusters if c['color']=='green']
    close=df['close'].values;ma=df['MA250'].values;n=len(df)

    # Position state
    pos={'cash':1_000_000,'base':0,'swing':0,'in_base':False,'entry_price':0,'peak':0,
         'swing_mode':False,'gc_available':False,'swing_entry':0,'last_action':''}

    # Restore from history
    if args.trade and os.path.exists(HF):
        hist=pd.read_csv(HF)
        for _,r2 in hist.iterrows():
            if r2['action']=='BUY' and '底仓' in str(r2.get('reason','')): pos['in_base']=True;pos['entry_price']=r2['price'];pos['base']=r2['shares']
            elif r2['action']=='BUY' and '波段' in str(r2.get('reason','')): pos['swing_mode']=True;pos['swing']=r2['shares'];pos['swing_entry']=r2['price']
            elif r2['action']=='SELL': pos['in_base']=False;pos['base']=0;pos['swing']=0;pos['swing_mode']=False
            pos['last_action']=r2['action']

    sig={'action':'HOLD','reason':'监控中'}
    ch=30  # check recent 30 days

    for i in range(max(1,n-ch),n):
        p=float(close[i]);ds=str(df['date'].iloc[i])[:10];am=close[i]>ma[i];amp=i>0 and close[i-1]>ma[i-1]

        # Base position entry: MA250 breakthrough
        if not pos['in_base']:
            if not amp and am:
                pos['in_base']=True;bp=p*(1+SLIP)
                raw=int(pos['cash']/bp/100)*100
                # Only use 100k for base in daily mode
                base_cash=100000;raw=int(base_cash/bp/100)*100
                if raw>=100:
                    val=bp*raw;cost=val+max(val*COMM,5)
                    if cost<=pos['cash']:
                        pos['cash']-=cost;pos['base']=raw;pos['entry_price']=bp;pos['peak']=bp;pos['gc_available']=False
                        sig={'date':ds,'action':'BUY','price':round(bp,4),'reason':'MA250突破底仓','shares':raw,'amount':round(val,2)}

        # Swing entry: green cluster signal
        if pos['in_base'] and not pos['swing_mode']:
            ok,reason,entry_p=check_long(df,i,clusters,green_clusters)
            if ok and pos['base']>0:
                swing_shares=pos['base']//3  # 1/3 for swing
                swing_shares=(swing_shares//100)*100
                if swing_shares>=100:
                    bp=entry_p*(1+SLIP);val=bp*swing_shares;cost=val+max(val*COMM,5)
                    if cost<=pos['cash']:
                        pos['cash']-=cost;pos['swing']=swing_shares;pos['swing_mode']=True;pos['swing_entry']=bp
                        sig={'date':ds,'action':'BUY','price':round(bp,4),'reason':'MACD绿柱簇波段','shares':swing_shares,'amount':round(val,2)}

        # Swing exit: red cluster signal
        if pos['swing_mode']:
            ok2,reason2,exit_p=check_short(df,i,clusters)
            if ok2:
                sp=exit_p*(1-SLIP);val=sp*pos['swing'];cash_add=val-max(val*COMM,5)
                pos['cash']+=cash_add
                ret=(sp-pos['swing_entry'])/pos['swing_entry']*100
                sig={'date':ds,'action':'SELL','price':round(sp,4),'reason':'红柱簇止盈','shares':pos['swing'],'amount':round(val,2),'ret':round(ret,2)}
                pos['swing']=0;pos['swing_mode']=False;pos['gc_available']=True

        # MA250 exit: close below MA250
        if pos['in_base'] and i>0 and close[i-1]>ma[i-1] and not am:
            total_sh=pos['base']+pos['swing'];sp=p*(1-SLIP);val=sp*total_sh
            pos['cash']+=val-max(val*COMM,5)
            ret=(sp-pos['entry_price'])/pos['entry_price']*100
            sig={'date':ds,'action':'SELL','price':round(sp,4),'reason':'跌破年线清仓','shares':total_sh,'amount':round(val,2),'ret':round(ret,2)}
            pos['base']=0;pos['swing']=0;pos['in_base']=False;pos['swing_mode']=False

    st=gs2(df,pos)
    print('')
    print('='*50)
    print('  '+NAME+' MACD波段策略 - 每日信号')
    print('='*50)
    print('  日期: '+st['date']+'  收盘: '+format(st['close'],'.2f')+'  MA250: '+format(st['MA250'],'.2f'))
    print('  年线上方: '+('是' if st['am'] else '否')+'  DIF: '+format(st['DIF'],'.4f')+'  DEA: '+format(st['DEA'],'.4f')+'  MACD: '+format(st['MACD'],'.4f'))

    if st['base']>0:
        dd_val=(st['close']-pos.get('entry_price',st['close']))/pos.get('entry_price',st['close'])*100
        print('')
        print('  [当前持仓]')
        print('  底仓: '+format(int(st['base']),',')+'股  波段: '+format(int(st['swing']),',')+'股')
        print('  入场价: '+format(pos.get('entry_price',0),'.3f')+'  回撤: '+format(dd_val,'+.1f')+'%')
        print('  持仓市值: '+format(int((st['base']+st['swing'])*st['close']),',')+'  总资产: '+format(int(st['tv']),','))
    else:
        print('')
        print('  [空仓等待]')
        print('  总资产: '+format(int(st['tv']),','))

    today=datetime.now().strftime('%Y-%m-%d')
    act=sig.get('action','HOLD')
    if act in ('BUY','SELL') and sig.get('date','')!=today: act='HOLD';sig['reason']='监控中'
    rmap={'BUY':'买入','SELL':'卖出','HOLD':'持有'}
    print('')
    print('  >>> 操作: '+rmap.get(act,act)+' | '+sig.get('reason','-'))
    if act in ('BUY','SELL'):
        print('  >>> 价格: '+str(sig.get('price','?'))+'  份额: '+str(sig.get('shares','?'))+'  金额: '+str(sig.get('amount','?')))
        if sig.get('ret') is not None: print('  >>> 收益: '+format(sig['ret'],'+.2f')+'%')

    if args.trade and act in ('BUY','SELL'):
        lt2(sig);print('');print('  [已记录]')
    print('='*50)

if __name__=='__main__': main()
