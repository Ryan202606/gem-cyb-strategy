#!/usr/bin/env python3
"""创业板 混合策略 — MA250入场 + 面积止盈 + 水下金叉回补"""
import sys, io, os, json, csv
if sys.stdout.encoding != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
import baostock as bs, pandas as pd, numpy as np
from datetime import datetime, timedelta
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt; import matplotlib.ticker as mticker

if not hasattr(pd.DataFrame,'append'):
    pd.DataFrame.append=lambda s,o,**kw:pd.concat([s,o],ignore_index=kw.get('ignore_index',False))

# ======== 参数 ========
AREA = 2000; MA_CONFIRM_LONG = 14; MA_CONFIRM_SHORT = 1; GC_CONFIRM = 0
NAME = '创业板'
SYMBOL = 'sz.399006'
RUN_TIME = datetime.now().strftime('%Y%m%d_%H%M')

# ======== 数据 ========
lg=bs.login()
end=datetime.now(); start=end-timedelta(days=365*10+350)
rs=bs.query_history_k_data_plus(SYMBOL,'date,open,high,low,close,volume,amount',
    start_date=start.strftime('%Y-%m-%d'),end_date=end.strftime('%Y-%m-%d'),frequency='d',adjustflag='1')
df=rs.get_data(); bs.logout()
for c in ['open','high','low','close','volume','amount']: df[c]=pd.to_numeric(df[c],errors='coerce')
df['date']=pd.to_datetime(df['date']); df=df.dropna(subset=['close']).sort_values('date').reset_index(drop=True)
df['MA250']=df['close'].rolling(250).mean(); df['AVG_VOL']=df['volume'].rolling(250).mean()
e12=df['close'].ewm(span=12,adjust=False).mean(); e26=df['close'].ewm(span=26,adjust=False).mean()
df['DIF']=e12-e26; df['DEA']=df['DIF'].ewm(span=9,adjust=False).mean(); df['BAR']=2*(df['DIF']-df['DEA'])
bt_start=datetime.now()-timedelta(days=365*10)
mask=df['date']>=pd.Timestamp(bt_start.date())
df2=df[mask].reset_index(drop=True)
close=df2['close'].values; vol=df2['volume'].values; avg_vol=df2['AVG_VOL'].values; dates=df2['date'].values; n=len(close)
dif=df2['DIF'].values; dea=df2['DEA'].values; bar=df2['BAR'].values; ma=df2['MA250'].values

# ======== 回测 ========
cash=1_000_000; shares=0; im=False; da=0; pb=0; ps=0; tc=0
state='ma_entry'; pg=0; eq=[1_000_000]
trades=[]; buys=[]; sells=[]; pos_days=0

for i in range(1,n):
    p=float(close[i]); ds=str(dates[i])[:10]
    d=dif[i]; de=dea[i]; dp=dif[i-1]; dep=dea[i-1]; am=close[i]>ma[i]
    if d>0: da+=d
    else: da=0

    if im:
        pos_days+=1
        if da>=AREA:
            sp=p*0.999; val=sp*shares; comm=max(val*0.0003,5); stamp=val*0.0005
            cash+=val-comm-stamp
            t={'type':'SELL','date':ds,'price':round(sp,2),'shares':shares,'amount':round(val,2),
               'comm':round(comm,2),'stamp':round(stamp,2),'reason':'area'}
            trades.append(t); sells.append((ds,sp,'area')); shares=0; im=False; da=0; ps=0; pb=0; state='golden'; pg=0
        else:
            ps=0 if am else ps+1
            if ps==1 and close[i-1]>ma[i-1]:
                sp=p*0.999; val=sp*shares; comm=max(val*0.0003,5); stamp=val*0.0005
                cash+=val-comm-stamp
                t={'type':'SELL','date':ds,'price':round(sp,2),'shares':shares,'amount':round(val,2),
                   'comm':round(comm,2),'stamp':round(stamp,2),'reason':'ma_cross'}
                trades.append(t); sells.append((ds,sp,'ma_cross')); shares=0; im=False; da=0; ps=0; pb=0; state='ma_entry'; pg=0
    else:
        sb=False; entry_reason=''
        if state=='ma_entry':
            if close[i-1]<=ma[i-1] and am: pb=1
            if pb>0:
                if am: pb+=1
                else: pb=0
                if pb>=MA_CONFIRM+1: sb=True; pb=0; entry_reason='MA250突破'
        elif state=='golden':
            gc=(d>de and dp<=dep and d<0)
            if gc and pg==0: pg=1
            if pg>0:
                if am:
                    pg+=1
                    if pg>=GC_CONFIRM+2: sb=True; pg=0; entry_reason='水下金叉回补'; state='ma_entry'
                else:
                    pg=0; state='ma_entry'

        if sb:
            bp=p*1.001; raw=int(cash/bp/100)*100
            while raw>=100:
                val=bp*raw; cost=val+max(val*0.0003,5)
                if cost<=cash: break
                raw-=100
            if raw>=100:
                val=bp*raw; comm=max(val*0.0003,5); cost=val+comm
                if cost<=cash:
                    cash-=cost; shares=raw; im=True; tc+=1
                    t={'type':'BUY','date':ds,'price':round(bp,2),'shares':raw,'amount':round(val,2),
                       'comm':round(comm,2),'stamp':0,'reason':entry_reason}
                    trades.append(t); buys.append((ds,bp,entry_reason)); da=0; ps=0; state='ma_entry'
    eq.append(cash+shares*p)

if im and shares>0:
    val=close[-1]*shares; cash+=val-max(val*0.0003,5)-val*0.0005
    trades.append({'type':'SELL','date':str(dates[-1])[:10],'price':round(close[-1],2),'shares':shares,
                   'amount':round(val,2),'comm':0,'stamp':0,'reason':'end'})

# ======== 指标 ========
total_ret=(cash/1_000_000-1)*100; yrs=n/252; ann=((cash/1_000_000)**(1/yrs)-1)*100
bh_ret=(close[-1]/close[0]-1)*100
eq_arr=np.array(eq); peak=eq_arr[0]; max_dd=0
for v in eq_arr:
    if v>peak: peak=v
    dd=(v-peak)/peak*100
    if dd<max_dd: max_dd=dd
sharpe_val=float(np.diff(eq_arr).mean()/np.diff(eq_arr).std()*np.sqrt(252)) if np.diff(eq_arr).std()>0 else 0
btrades=[t for t in trades if t['type']=='BUY']; strades=[t for t in trades if t['type']=='SELL']
wr=sum(1 for b,s in zip(btrades,strades) if s['price']>b['price'])/len(strades)*100 if strades else 0

# ======== 输出目录 ========
out_dir=f'D:/Desktoop/量化交易/3_MA200突破选股/results/{NAME}_混合策略_{RUN_TIME}'
os.makedirs(out_dir,exist_ok=True)

# ======== 打印报告 ========
print(f"""
{"="*70}
  创业板 混合策略 — MA250入场 + 面积止盈 + 水下金叉回补
  运行时间: {RUN_TIME}
{"="*70}

[策略规则]
  主入场:  收盘上穿MA250 -> 连续{MA_CONFIRM}天站稳 -> 第{MA_CONFIRM+1}天全仓买入
  回补入场: 面积止盈后,首次水下金叉+站上年线{GC_CONFIRM}天 -> 买入
  出场A:   DIF水上累计面积 >= {AREA} (动能衰竭止盈)
  出场B:   收盘下穿MA250 (趋势结束)

[回测业绩]
  初始:  1,000,000
  最终:  {cash:,.0f}
  总收益: {total_ret:+.2f}%
  年化:   {ann:.2f}%
  回撤:   {max_dd:.2f}%
  夏普:   {sharpe_val:.2f}
  胜率:   {wr:.1f}%
  交易:   {tc}笔
  持仓:   {pos_days}天 ({pos_days/n*100:.0f}%)
  基准:   买入持有 {bh_ret:.2f}%

[逐笔交易]
{'#':<3} {'买入日':<12} {'买入价':>8} {'类型':<12} {'卖出日':<12} {'卖出价':>8} {'收益%':>8} {'持仓':>5} {'出场原因':<12}
{'-'*90}""")

pn=0
for j in range(len(trades)):
    t=trades[j]
    if t['type']=='BUY': b=t
    else:
        pn+=1; s=t
        ret=(s['price']-b['price'])/b['price']*100
        hold=(pd.Timestamp(s['date'])-pd.Timestamp(b['date'])).days
        rmap={'area':'面积止盈','ma_cross':'MA250跌破','end':'期末','dif_cross':'DIF跌破'}
        rr=rmap.get(s['reason'],s['reason'])
        print(f'{pn:<3} {b["date"]:<12} {b["price"]:>8.0f} {b["reason"]:<12} {s["date"]:<12} {s["price"]:>8.0f} {ret:>+7.2f}% {hold:>5} {rr:<12}')

# ======== 保存数据 ========
# JSON
report={'strategy':'创业板混合策略','params':{'area':AREA,'ma_confirm':MA_CONFIRM,'gc_confirm':GC_CONFIRM},
    'symbol':SYMBOL,'run_time':RUN_TIME,
    'summary':{'final_capital':round(cash,2),'total_return':round(total_ret,4),'annual_return':round(ann,4),
               'max_dd':round(max_dd,4),'sharpe':round(sharpe_val,2),'win_rate':round(wr,4),'trades':tc},
    'trades':trades,'equity':[{'date':str(dates[min(i,n-1)])[:10],'close':float(close[min(i,n-1)]),'total':float(eq[min(i,len(eq)-1)])} for i in range(0,n,20)]}
with open(f'{out_dir}/report.json','w',encoding='utf-8') as f: json.dump(report,f,ensure_ascii=False,indent=2,default=str)
# CSV equity
pd.DataFrame([{'date':str(dates[min(i,n-1)])[:10],'close':float(close[min(i,n-1)]),'equity':float(eq[min(i,len(eq)-1)])} for i in range(0,n,20)]).to_csv(f'{out_dir}/equity.csv',index=False,encoding='utf-8-sig')
# CSV trades
pd.DataFrame([{'日期':t['date'],'方向':t['type'],'价格':t['price'],'份额':t.get('shares',''),
               '金额':t.get('amount',''),'原因':t.get('reason','')} for t in trades]).to_csv(f'{out_dir}/trades.csv',index=False,encoding='utf-8-sig')
print(f'\n[数据] {out_dir}')

# ======== 总览图 ========
plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':['Microsoft YaHei','SimHei'],'axes.unicode_minus':False,'figure.dpi':150,'savefig.dpi':150,'savefig.bbox':'tight'})
BG='#FAFAFA'; GR='#E0E0E0'; LN='#CCCCCC'; TX='#374151'; MU='#9CA3AF'; BL='#2563EB'; RD='#DC2626'; GN='#16A34A'; OR='#EA580C'; PU='#7C3AED'

fig=plt.figure(figsize=(18,12),facecolor=BG)
gs=fig.add_gridspec(4,1,height_ratios=[3,1.5,1.5,2],hspace=0.05)
fig.suptitle(f'{NAME} 混合策略 (面积{AREA} MA确认{MA_CONFIRM}天 GC确认{GC_CONFIRM}天) — {RUN_TIME}',fontsize=15,fontweight='bold',color=TX,y=0.99)

# 图1
ax1=fig.add_subplot(gs[0]); ax1.set_facecolor(BG)
ax1.plot(dates,close,color=BL,lw=1.0,alpha=0.85,label='收盘价')
ax1.plot(dates,ma,color=OR,lw=1.0,alpha=0.5,ls='--',label='MA250')
for ds2,bp,reason in buys:
    c=GN if 'MA' in reason else PU
    ax1.scatter(pd.Timestamp(ds2),bp,color=c,s=50,marker='^',edgecolors='white',lw=0.5,zorder=5)
for ds2,sp,sr in sells:
    c=GN if sr=='area' else RD
    ax1.scatter(pd.Timestamp(ds2),sp,color=c,s=50,marker='v',edgecolors='white',lw=0.5,zorder=5)
# position shading
p2=[]; im2=False; da2=0; pb2=0; ps2=0; st2='ma_entry'; pg2=0
for i in range(1,n):
    d2=dif[i]; de2=dea[i]; dp2=dif[i-1]; dep2=dea[i-1]; am2=close[i]>ma[i]
    if d2>0: da2+=d2
    else: da2=0
    if im2:
        if da2>=AREA: im2=False; da2=0; ps2=0; pb2=0; st2='golden'; pg2=0
        else:
            ps2=0 if am2 else ps2+1
            if ps2==1 and close[i-1]>ma[i-1]: im2=False; da2=0; ps2=0; pb2=0; st2='ma_entry'; pg2=0
    else:
        sb2=False
        if st2=='ma_entry':
            if close[i-1]<=ma[i-1] and am2: pb2=1
            if pb2>0:
                if am2: pb2+=1
                else: pb2=0
                if pb2>=MA_CONFIRM+1: sb2=True; pb2=0
        elif st2=='golden':
            gc2=(d2>de2 and dp2<=dep2 and d2<0)
            if gc2 and pg2==0: pg2=1
            if pg2>0:
                if am2:
                    pg2+=1
                    if pg2>=GC_CONFIRM+2: sb2=True; pg2=0; st2='ma_entry'
                else:
                    pg2=0; st2='ma_entry'
        if sb2: im2=True; da2=0; ps2=0; st2='ma_entry'
    p2.append(1 if im2 else 0)
p2.append(0)
d=dates[1:]; starts=[]; ends=[]; i=0
while i<len(p2):
    if p2[i]==1:
        si=i
        while i<len(p2) and p2[i]==1: i+=1
        starts.append(d[si]); ends.append(d[min(i,len(d)-1)])
    else: i+=1
for s,e in zip(starts,ends): ax1.axvspan(s,e,alpha=0.07,color=GN,lw=0)

from matplotlib.lines import Line2D
ax1.legend(handles=[
    Line2D([0],[0],color=BL,lw=1.0,label='收盘价'),
    Line2D([0],[0],color=OR,lw=1.0,label='MA250'),
    Line2D([0],[0],marker='^',color='w',markerfacecolor=GN,markersize=8,label=f'MA250突破入场'),
    Line2D([0],[0],marker='^',color='w',markerfacecolor=PU,markersize=8,label='水下金叉回补'),
    Line2D([0],[0],marker='v',color='w',markerfacecolor=GN,markersize=8,label='面积止盈'),
    Line2D([0],[0],marker='v',color='w',markerfacecolor=RD,markersize=8,label='MA250跌破'),
],loc='upper left',fontsize=8,framealpha=0.9,edgecolor=LN)
ax1.set_ylabel('指数点位',fontsize=10,color=TX); ax1.tick_params(colors=MU,labelsize=8)
ax1.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x,_:f'{x:,.0f}'))
ax1.grid(True,color=GR,lw=0.5,alpha=0.7)

# 图2 DIF面积
ax2=fig.add_subplot(gs[1]); ax2.set_facecolor(BG)
areas=np.zeros(n); a=0
for i in range(n):
    if dif[i]>0: a+=dif[i]
    else: a=0
    areas[i]=a
ax2.fill_between(dates,0,areas,color=PU,alpha=0.15)
ax2.plot(dates,areas,color=PU,lw=1.5)
ax2.axhline(y=AREA,color=GN,lw=1.5,ls='--',label=f'止盈阈值={AREA}')
ax2.fill_between(dates,AREA,areas,where=(areas>=AREA),color=GN,alpha=0.12)
ax2.set_ylabel('DIF面积',fontsize=10,color=PU); ax2.legend(loc='upper left',fontsize=8,framealpha=0.9,edgecolor=LN)
ax2.tick_params(colors=MU,labelsize=8); ax2.grid(True,color=GR,lw=0.5,alpha=0.7)

# 图3 MACD
ax3=fig.add_subplot(gs[2]); ax3.set_facecolor(BG); ax3.axhline(y=0,color=LN,lw=0.8)
cb=['#DC2626' if b>=0 else '#16A34A' for b in bar]
ax3.bar(dates,bar,color=cb,width=0.8,alpha=0.5)
ax3.plot(dates,dif,color=BL,lw=1.0,label='DIF'); ax3.plot(dates,dea,color=OR,lw=1.0,label='DEA')
ax3.set_ylabel('MACD',fontsize=10,color=TX); ax3.legend(loc='upper left',fontsize=8,framealpha=0.9,edgecolor=LN)
ax3.tick_params(colors=MU,labelsize=8); ax3.grid(True,color=GR,lw=0.5,alpha=0.7)

# 图4 权益
ax4=fig.add_subplot(gs[3]); ax4.set_facecolor(BG)
ax4.plot(dates[1:],eq_arr[1:]/1e4,color=BL,lw=1.6,label='混合策略权益')
ax4.plot(dates,close/close[0]*1_000_000/1e4,color=MU,lw=1.0,ls='--',alpha=0.6,label='买入持有')
ax4.axhline(y=100,color=LN,lw=0.8,ls=':')
pi=np.argmax(eq_arr); di=pi+np.argmin(eq_arr[pi:])
ax4.plot([dates[min(pi,n-1)],dates[min(di,n-1)]],[eq_arr[min(pi,n-1)]/1e4,eq_arr[min(di,n-1)]/1e4],'o',color=RD,ms=5,mew=0.5,mec='white')
ax4.annotate(f'最大回撤{max_dd:.1f}%',(dates[min(di,n-1)],eq_arr[min(di,n-1)]/1e4),
    textcoords='offset points',xytext=(10,-15),fontsize=8,color=RD,arrowprops=dict(arrowstyle='->',color=RD,lw=0.8))
ax4.set_ylabel('权益(万元)',fontsize=10,color=TX); ax4.set_xlabel('日期',fontsize=10,color=TX)
ax4.legend(loc='upper left',fontsize=8.5,framealpha=0.9,edgecolor=LN)
ax4.tick_params(colors=MU,labelsize=8)
ax4.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x,_:f'{x:,.0f}'))

summary=(f'初始100万->{cash:,.0f}元 | 总收益{total_ret:+.2f}% 年化{ann:+.2f}% | '
         f'回撤{max_dd:.1f}% 夏普{sharpe_val:.2f} | {tc}笔 胜率{wr:.0f}% | 买入持有{bh_ret:+.1f}%')
fig.text(0.5,0.003,summary,ha='center',va='bottom',fontsize=9,color=MU,
    bbox=dict(boxstyle='round,pad=0.3',facecolor='white',edgecolor=LN,alpha=0.9),transform=fig.transFigure)
fig.savefig(f'{out_dir}/overview.png',facecolor=BG,edgecolor='none')
plt.close()
print(f'[总览图] {out_dir}/overview.png')

# ======== 交易明细图 ========
td_dir=f'{out_dir}/trades'; os.makedirs(td_dir,exist_ok=True)
pairs=[]; bt=None
for t in trades:
    if t['type']=='BUY': bt=t
    elif bt: pairs.append((bt,t)); bt=None

for idx,(b,s) in enumerate(pairs):
    bdt=pd.Timestamp(b['date']); sdt=pd.Timestamp(s['date'])
    ret=(s['price']-b['price'])/b['price']*100; hold=(sdt-bdt).days
    rc=GN if ret>0 else RD; rt='盈利' if ret>0 else '亏损'
    sd=bdt-pd.Timedelta(days=60); ed=sdt+pd.Timedelta(days=20)
    si=max(0,np.searchsorted(dates,np.datetime64(sd.to_datetime64()))-5)
    ei=min(n-1,np.searchsorted(dates,np.datetime64(ed.to_datetime64()))+5)
    pds=dates[si:ei+1]; pcl=close[si:ei+1]; pma=ma[si:ei+1]
    pdif=dif[si:ei+1]; pdea=dea[si:ei+1]; pbar=bar[si:ei+1]
    hm=(dates>=np.datetime64(bdt.to_datetime64()))&(dates<=np.datetime64(sdt.to_datetime64()))
    hd=dates[hm]; hcl=close[hm]

    fig2,(a1,a2)=plt.subplots(2,1,figsize=(16,9),gridspec_kw={'height_ratios':[2.5,1],'hspace':0.05},facecolor=BG)
    fig2.suptitle(f'#{idx+1} {b["date"]}->{s["date"]} | {ret:+.2f}%({rt}) | {hold}天 | {b["reason"]}',fontsize=13,fontweight='bold',color=TX,y=0.97)
    a1.set_facecolor(BG)
    a1.plot(pds,pcl,color=BL,lw=1.5,alpha=0.9); a1.plot(pds,pma,color=OR,lw=1.0,alpha=0.5,ls='--')
    if len(hd)>0: a1.axvspan(hd[0],hd[-1],alpha=0.08,color=GN,lw=0)
    a1.scatter(bdt,b['price'],color=GN,s=120,marker='^',edgecolors='white',lw=1.5,zorder=10)
    a1.scatter(sdt,s['price'],color=rc,s=120,marker='v',edgecolors='white',lw=1.5,zorder=10)
    a1.plot([bdt,sdt],[b['price'],s['price']],color=rc,lw=1.2,ls=':',alpha=0.5)
    a1.annotate(f'买{b["price"]:.0f}',(bdt,b['price']),textcoords='offset points',xytext=(15,15),
        fontsize=10,color=GN,fontweight='bold',bbox=dict(boxstyle='round',facecolor='white',edgecolor=GN,alpha=0.9))
    a1.annotate(f'卖{s["price"]:.0f}',(sdt,s['price']),textcoords='offset points',xytext=(-15,-25),
        fontsize=10,color=rc,fontweight='bold',bbox=dict(boxstyle='round',facecolor='white',edgecolor=rc,alpha=0.9))
    if len(hcl)>5:
        pv=hcl[0]; pi2=0; dds=[]
        for j in range(1,len(hcl)):
            if hcl[j]>pv: pv=hcl[j]; pi2=j
            dd=(hcl[j]-pv)/pv*100
            if dd<-5: dds.append((j,dd,pi2))
        dds.sort(key=lambda x:x[1]); ann_set=set()
        for j,dd,pi2 in dds:
            if j in ann_set: continue
            a1.plot([hd[pi2],hd[j]],[hcl[pi2],hcl[j]],color=RD,lw=2,alpha=0.5)
            a1.annotate(f'-{abs(dd):.1f}%',(hd[j],hcl[j]),textcoords='offset points',
                xytext=(5,-15),fontsize=8,color=RD,fontweight='bold',arrowprops=dict(arrowstyle='->',color=RD,lw=0.8))
            for k in range(max(0,j-10),min(len(hcl),j+10)): ann_set.add(k)
    a1.set_ylabel('指数',fontsize=10,color=TX); a1.tick_params(colors=MU,labelsize=8)
    a1.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x,_:f'{x:,.0f}'))
    a1.grid(True,color=GR,lw=0.5,alpha=0.7); a1.set_xlim(sd,ed)
    a2.set_facecolor(BG); a2.axhline(y=0,color=LN,lw=0.8)
    a2.bar(pds,pbar,color=['#DC2626' if x>=0 else '#16A34A' for x in pbar],width=0.8,alpha=0.5)
    a2.plot(pds,pdif,color=BL,lw=1.0); a2.plot(pds,pdea,color=OR,lw=1.0)
    a2.set_ylabel('MACD',fontsize=10,color=TX); a2.tick_params(colors=MU,labelsize=8)
    a2.grid(True,color=GR,lw=0.5,alpha=0.7); a2.set_xlim(sd,ed)
    fig2.savefig(f'{td_dir}/trade_{idx+1:02d}_{b["date"]}_{ret:+.1f}pct.png',facecolor=BG,edgecolor='none')
    plt.close(fig2)

print(f'[交易明细] {td_dir}/ ({len(pairs)}张)')
print(f'\n全部结果: {out_dir}')
