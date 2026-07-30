#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import sys,io,os
from datetime import datetime,timedelta
if sys.stdout.encoding!='utf-8': sys.stdout=io.TextIOWrapper(sys.stdout.buffer,encoding='utf-8',errors='replace')
import baostock as bs,pandas as pd,numpy as np
if not hasattr(pd.DataFrame,'append'): pd.DataFrame.append=lambda s,o,**kw:pd.concat([s,o],ignore_index=kw.get('ignore_index',False))
AREA=2000;LONG_C=14;SHORT_C=1;GC_C=0;VP=150;VM=1.8
COMM=0.0003;STAMP=0.0005;SLIP=0.001
SYMBOL='sz.399006'
SCRIPT_DIR=os.path.dirname(os.path.abspath(__file__))
HF=os.path.join(os.path.dirname(SCRIPT_DIR),'trade_history.csv')

def fd(days=400):
    dn=os.devnull;old=sys.stdout;sys.stdout=open(dn,'w')
    lg=bs.login();sys.stdout=old
    end=datetime.now();start=end-timedelta(days=days)
    rs=bs.query_history_k_data_plus(SYMBOL,'date,open,high,low,close,volume,amount',start_date=start.strftime('%Y-%m-%d'),end_date=end.strftime('%Y-%m-%d'),frequency='d',adjustflag='1')
    df=rs.get_data() if rs and rs.error_code=='0' else pd.DataFrame()
    sys.stdout=open(dn,'w');bs.logout();sys.stdout=old
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
    def __init__(self): self.im=False;self.ep=0;self.sh=0;self.pp=0;self.cash=1_000_000;self.st='ma_entry';self.pb=0;self.ps=0;self.vb=False;self.pg=0;self.da=0
    def reset(self,st='ma_entry'): self.st=st;self.pb=0;self.ps=0;self.vb=False;self.pg=0;self.da=0

def cs(df,t):
    if len(df)<250: return {'date':str(datetime.now().date()),'action':'WAIT'}
    close=df['close'].values;ma=df['MA250'].values;vol=df['volume'].values;av=df['AVG_VOL'].values
    dif=df['DIF'].values;dea=df['DEA'].values;dates=df['date'].values;areas=ca(dif);n=len(df)
    si=0
    for i in range(n):
        if not np.isnan(ma[i]): si=i;break
    cs2=max(si+1,n-30) if t.im else max(1,si)
    ls={'date':str(dates[-1])[:10],'action':'持有','reason':'持仓中' if t.im else '等待信号'}
    for i in range(cs2,n):
        p=float(close[i]);ds=str(dates[i])[:10]
        d=dif[i];de=dea[i];dp=dif[i-1];dep=dea[i-1];am=close[i]>ma[i];amp=close[i-1]>ma[i-1];v=vol[i];av2=av[i]
        if d>0: t.da+=d
        else: t.da=0
        if t.im:
            if p>t.pp: t.pp=p
            if t.da>=AREA:
                sp=p*(1-SLIP);val=sp*t.sh;t.cash+=val-max(val*COMM,5)-val*STAMP
                ret=(sp-t.ep)/t.ep*100
                ls={'date':ds,'action':'卖出','price':round(sp,4),'reason':'面积止盈','shares':t.sh,'amount':round(val,2),'ret':round(ret,2)}
                t.im=False;t.sh=0;t.reset('golden')
            t.ps=0 if am else t.ps+1
            if t.ps==1 and amp:
                sp=p*(1-SLIP);val=sp*t.sh;t.cash+=val-max(val*COMM,5)-val*STAMP
                ret=(sp-t.ep)/t.ep*100
                ls={'date':ds,'action':'卖出','price':round(sp,4),'reason':'跌破年线','shares':t.sh,'amount':round(val,2),'ret':round(ret,2)}
                t.im=False;t.sh=0;t.reset('ma_entry')
        else:
            sb=False;br=''
            if t.st=='ma_entry':
                if not amp and am: t.pb=1;t.vb=(not np.isnan(av2) and v>av2*VM)
                if t.pb>0:
                    if am: t.pb+=1
                    else: t.pb=0;t.vb=False
                    cn=SHORT_C if t.vb else LONG_C
                    if t.pb>=cn+1: sb=True;t.pb=0;br='放量突破' if t.vb else '年线突破';t.vb=False;t.st='ma_entry'
            elif t.st=='golden':
                gc=(d>de and dp<=dep and d<0)
                if gc and t.pg==0: t.pg=1
                if t.pg>0:
                    if am: t.pg+=1
                    else: t.pg=0;t.st='ma_entry'
                    if t.pg>=GC_C+2: sb=True;t.pg=0;br='金叉回补';t.st='ma_entry'
            if sb:
                bp=p*(1+SLIP);raw=int(t.cash*0.998/bp/100)*100
                if raw>=100:
                    val=bp*raw;cost=val+max(val*COMM,5)
                    if cost<=t.cash: t.cash-=cost;t.sh=raw;t.im=True;t.ep=bp;t.pp=bp;t.da=0;t.ps=0
                    ls={'date':ds,'action':'买入','price':round(bp,4),'reason':br,'shares':raw,'amount':round(val,2)}
    return ls

def gs(df,t):
    last=df.iloc[-1]
    return {'date':str(last['date'])[:10],'close':float(last['close']),'MA250':float(last['MA250']),'DIF':float(last['DIF']),'am':float(last['close'])>float(last['MA250']),'im':t.im,'da':t.da,'ep':t.ep if t.im else 0,'pp':t.pp if t.im else 0,'pv':round(t.sh*float(last['close']),2) if t.im else 0,'tv':round(t.cash+(t.sh*float(last['close']) if t.im else 0),2)}

def lt(sig):
    fe=os.path.exists(HF)
    row={'time':datetime.now().strftime('%Y-%m-%d %H:%M:%S'),'sig_date':sig.get('date',''),'action':sig.get('action',''),'price':sig.get('price',0),'reason':sig.get('reason',''),'shares':sig.get('shares',0),'amount':sig.get('amount',0),'ret':sig.get('ret','')}
    pd.DataFrame([row]).to_csv(HF,mode='a',header=not fe,index=False,encoding='utf-8-sig')

def main():
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--trade',action='store_true');p.add_argument('--history',action='store_true');args=p.parse_args()
    if args.history:
        if os.path.exists(HF): df=pd.read_csv(HF);print('创业板交易记录('+str(len(df))+'条):');print(df.tail(20).to_string(index=False))
        else: print('无记录')
        return
    df=fd(400)
    if len(df)==0: return
    df=ci(df);t=PT()
    if args.trade and os.path.exists(HF):
        hist=pd.read_csv(HF);buys=hist[hist['action']=='买入'];sells=hist[hist['action']=='卖出']
        if len(buys)>len(sells):
            lb=buys.iloc[-1];t.im=True;t.ep=lb['price'];t.sh=lb['shares'];t.pp=lb['price']
            bd2=pd.Timestamp(lb['sig_date'])
            for i in range(len(df)):
                if df.iloc[i]['date']>=bd2:
                    pp2=float(df.iloc[i]['close'])
                    if pp2>t.pp: t.pp=pp2
            t.st='ma_entry'
    sig=cs(df,t);st=gs(df,t)
    today=datetime.now().strftime('%Y-%m-%d')
    act=sig.get('action','持有')
    if act in ('买入','卖出') and sig.get('date','')!=today: act='持有';sig['reason']='持仓中' if st['im'] else '等待信号'
    # Table format: 创业板 持仓中 卖出 跌破年线
    status='持仓中' if st['im'] else '空仓'
    advice=act if act!='持有' else '无操作'
    reason=sig.get('reason','-')
    print('创业板  '+status+'  '+advice+'  '+reason)
    # Details
    if st['im']:
        dd=(st['close']-st['pp'])/st['pp']*100 if st['pp']>0 else 0
        print('  入场'+format(st['ep'],'.2f')+' 现价'+format(st['close'],'.2f')+' 回撤'+format(dd,'+.1f')+'% 市值'+format(int(st['pv']),','))
    print('  资产'+format(int(st['tv']),',')+' 收盘='+format(st['close'],'.2f')+' 年线='+format(st['MA250'],'.2f')+' DIF='+format(st['DIF'],'.1f')+' 面积='+str(int(st['da']))+'/'+str(AREA))
    if args.trade and sig['action'] in ('买入','卖出'): lt(sig)

if __name__=='__main__': main()