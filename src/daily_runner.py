#!/usr/bin/env python3
"""
创业板 混合策略 -- 每日自动运行脚本
用法: python daily_runner.py / --trade / --history
"""
import sys,io,os
from datetime import datetime,timedelta
if sys.stdout.encoding!='utf-8': sys.stdout=io.TextIOWrapper(sys.stdout.buffer,encoding='utf-8',errors='replace')
import baostock as bs,pandas as pd,numpy as np
if not hasattr(pd.DataFrame,'append'): pd.DataFrame.append=lambda s,o,**kw:pd.concat([s,o],ignore_index=kw.get('ignore_index',False))

SYMBOL='sz.399006'
AREA=2000;LONG_C=14;SHORT_C=1;GC_C=0;VP=150;VM=1.8
COMM=0.0003;STAMP=0.0005;SLIP=0.001
SCRIPT_DIR=os.path.dirname(os.path.abspath(__file__))
HF=os.path.join(os.path.dirname(SCRIPT_DIR),'trade_history.csv')

def fd(days=400):
    lg=bs.login();end=datetime.now();start=end-timedelta(days=days)
    rs=bs.query_history_k_data_plus(SYMBOL,'date,open,high,low,close,volume,amount',start_date=start.strftime('%Y-%m-%d'),end_date=end.strftime('%Y-%m-%d'),frequency='d',adjustflag='1')
    df=rs.get_data() if rs and rs.error_code=='0' else pd.DataFrame();bs.logout()
    if len(df)==0: return df
    for c in ['open','high','low','close','volume','amount']: df[c]=pd.to_numeric(df[c],errors='coerce')
    df['date']=pd.to_datetime(df['date']);df=df.dropna(subset=['close']).sort_values('date').reset_index(drop=True)
    return df

def ci(df):
    df=df.copy();df['MA250']=df['close'].rolling(250).mean();df['AVG_VOL']=df['volume'].rolling(VP).mean()
    e12=df['close'].ewm(span=12,adjust=False).mean();e26=df['close'].ewm(span=26,adjust=False).mean()
    df['DIF']=e12-e26;df['DEA']=df['DIF'].ewm(span=9,adjust=False).mean();df['BAR']=2*(df['DIF']-df['DEA'])
    return df

def ca(dif):
    areas=np.zeros(len(dif));a=0
    for i in range(len(dif)):
        if dif[i]>0: a+=dif[i]
        else: a=0
        areas[i]=a
    return areas

class PT:
    def __init__(self):
        self.im=False;self.ed=None;self.ep=0;self.sh=0;self.pp=0;self.cash=1_000_000
        self.st='ma_entry';self.pb=0;self.ps=0;self.vb=False;self.pg=0;self.da=0
    def reset(self,st='ma_entry'): self.st=st;self.pb=0;self.ps=0;self.vb=False;self.pg=0;self.da=0

def cs(df,tracker):
    if len(df)<250: return {'date':str(datetime.now().date()),'action':'WAIT','reason':'data'}
    close=df['close'].values;ma=df['MA250'].values;vol=df['volume'].values;av=df['AVG_VOL'].values
    dif=df['DIF'].values;dea=df['DEA'].values;dates=df['date'].values;areas=ca(dif);n=len(df)
    si=0
    for i in range(n):
        if not np.isnan(ma[i]): si=i;break
    cs2=max(si+1,n-30) if tracker.im else max(1,si)
    ls={'date':str(dates[-1])[:10],'action':'HOLD','reason':'holding' if tracker.im else 'waiting'}
    for i in range(cs2,n):
        p=float(close[i]);ds=str(dates[i])[:10]
        d=dif[i];de=dea[i];dp=dif[i-1];dep=dea[i-1];am=close[i]>ma[i];amp=close[i-1]>ma[i-1];v=vol[i];av2=av[i]
        if d>0: tracker.da+=d
        else: tracker.da=0
        if tracker.im:
            if p>tracker.pp: tracker.pp=p
            if tracker.da>=AREA:
                sp=p*(1-SLIP);val=sp*tracker.sh;tracker.cash+=val-max(val*COMM,5)-val*STAMP
                ret=(sp-tracker.ep)/tracker.ep*100
                ls={'date':ds,'action':'SELL','price':round(sp,4),'reason':'area','shares':tracker.sh,'amount':round(val,2),'return_pct':round(ret,2),'details':'DIF面积='+str(int(tracker.da))+'>='+str(AREA)+', '+format(ret,'+.1f')+'%'}
                tracker.im=False;tracker.sh=0;tracker.reset('golden')
            tracker.ps=0 if am else tracker.ps+1
            if tracker.ps==1 and amp:
                sp=p*(1-SLIP);val=sp*tracker.sh;tracker.cash+=val-max(val*COMM,5)-val*STAMP
                ret=(sp-tracker.ep)/tracker.ep*100
                ls={'date':ds,'action':'SELL','price':round(sp,4),'reason':'ma','shares':tracker.sh,'amount':round(val,2),'return_pct':round(ret,2),'details':format(p,'.2f')+'<MA250='+format(ma[i],'.2f')+', '+format(ret,'+.1f')+'%'}
                tracker.im=False;tracker.sh=0;tracker.reset('ma_entry')
        else:
            sb=False;br=''
            if tracker.st=='ma_entry':
                if not amp and am: tracker.pb=1;tracker.vb=(not np.isnan(av2) and v>av2*VM)
                if tracker.pb>0:
                    if am: tracker.pb+=1
                    else: tracker.pb=0;tracker.vb=False
                    cn=SHORT_C if tracker.vb else LONG_C
                    if tracker.pb>=cn+1: sb=True;tracker.pb=0;br='fangliang'+str(SHORT_C)+'d' if tracker.vb else 'MA250'+str(LONG_C)+'d';tracker.vb=False;tracker.st='ma_entry'
            elif tracker.st=='golden':
                gc=(d>de and dp<=dep and d<0)
                if gc and tracker.pg==0: tracker.pg=1
                if tracker.pg>0:
                    if am: tracker.pg+=1
                    else: tracker.pg=0;tracker.st='ma_entry'
                    if tracker.pg>=GC_C+2: sb=True;tracker.pg=0;br='golden_cross';tracker.st='ma_entry'
            if sb:
                bp=p*(1+SLIP);raw=int(tracker.cash*0.998/bp/100)*100
                if raw>=100:
                    val=bp*raw;cost=val+max(val*COMM,5)
                    if cost<=tracker.cash:
                        tracker.cash-=cost;tracker.sh=raw;tracker.im=True;tracker.ep=bp;tracker.pp=bp;tracker.da=0;tracker.ps=0
                        ls={'date':ds,'action':'BUY','price':round(bp,4),'reason':br,'shares':raw,'amount':round(val,2),'details':format(bp,'.2f')+', '+str(raw)+'shares, cash='+format(int(tracker.cash),',')}
    return ls

def gs(df,tracker):
    last=df.iloc[-1]
    return {'date':str(last['date'])[:10],'close':float(last['close']),'MA250':float(last['MA250']),'DIF':float(last['DIF']),'DEA':float(last['DEA']),'da':round(tracker.da,1),'am':float(last['close'])>float(last['MA250']),'im':tracker.im,'st':tracker.st,'pb':tracker.pb,'ps':tracker.ps,'ep':tracker.ep if tracker.im else 0,'pp':tracker.pp if tracker.im else 0,'pv':round(tracker.sh*float(last['close']),2) if tracker.im else 0,'tv':round(tracker.cash+(tracker.sh*float(last['close']) if tracker.im else 0),2)}

def lt(sig):
    fe=os.path.exists(HF)
    row={'time':datetime.now().strftime('%Y-%m-%d %H:%M:%S'),'sig_date':sig.get('date',''),'action':sig.get('action',''),'price':sig.get('price',0),'reason':sig.get('reason',''),'shares':sig.get('shares',0),'amount':sig.get('amount',0),'ret':sig.get('return_pct',''),'detail':sig.get('details','')}
    pd.DataFrame([row]).to_csv(HF,mode='a',header=not fe,index=False,encoding='utf-8-sig')

def sh():
    if not os.path.exists(HF): print('No history');return
    df=pd.read_csv(HF);print('History: '+str(len(df)));print('='*60);print(df.tail(20).to_string(index=False))

def main():
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--trade',action='store_true');p.add_argument('--history',action='store_true');args=p.parse_args()
    if args.history: sh();return
    print('['+datetime.now().strftime('%H:%M:%S')+'] Loading...')
    df=fd(400)
    if len(df)==0: print('No data');return
    df=ci(df);tracker=PT()
    if args.trade and os.path.exists(HF):
        hist=pd.read_csv(HF);buys=hist[hist['action']=='BUY'];sells=hist[hist['action']=='SELL']
        if len(buys)>len(sells):
            lb=buys.iloc[-1];tracker.im=True;tracker.ed=lb['sig_date'];tracker.ep=lb['price'];tracker.sh=lb['shares']
            bd2=pd.Timestamp(lb['sig_date'])
            for i in range(len(df)):
                if df.iloc[i]['date']>=bd2:
                    pp2=float(df.iloc[i]['close'])
                    if pp2>tracker.pp: tracker.pp=pp2
            tracker.st='ma_entry'
    sig=cs(df,tracker);st=gs(df,tracker)
    print('='*50);print('  GEM CYB Strategy  '+datetime.now().strftime('%Y-%m-%d %H:%M'));print('='*50)
    print('  '+st['date']+' close='+format(st['close'],'.2f')+' MA250='+format(st['MA250'],'.2f')+' above='+('Y' if st['am'] else 'N'))
    print('  DIF='+format(st['DIF'],'.1f')+' DEA='+format(st['DEA'],'.1f')+' area='+str(int(st['da']))+'/'+str(AREA))
    print('  Position='+('IN' if st['im'] else 'OUT')+' state='+st['st'])
    if st['im']:
        dd=(st['close']-st['pp'])/st['pp']*100 if st['pp']>0 else 0
        print('  Entry='+format(st['ep'],'.2f')+' Peak='+format(st['pp'],'.2f')+' DD='+format(dd,'+.1f')+'% Value=Y'+format(int(st['pv']),','))
    print('  Total=Y'+format(int(st['tv']),','))
    today=datetime.now().strftime('%Y-%m-%d')
    if sig.get('action') in ('BUY','SELL') and sig.get('date','')!=today: sig['action']='HOLD';sig['reason']='holding' if st['im'] else 'waiting'
    print('\n  >>> '+sig['action']+' | '+sig['reason'])
    if sig.get('details') and sig.get('date','')==today: print('  >>> '+sig['details'])
    if args.trade and sig['action'] in ('BUY','SELL'): lt(sig);print('  [Logged]')

if __name__=='__main__': main()