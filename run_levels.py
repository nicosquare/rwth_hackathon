"""Run from repo root: OMP_NUM_THREADS=4 python solution/run_levels.py.
Forecast tomorrow at noon Europe/Berlin today. Historical weather only.
Chronological train/calibration/test; all eligible test intervals evaluated.
"""
from pathlib import Path
import json, argparse, subprocess, gzip, shutil
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score,mean_pinball_loss
import joblib
T='kWh_received_Total'; TZ='Europe/Berlin'
F=['slot','dow','month','weekend','slot_sin','slot_cos','year_sin','year_cos','lag48','lag168','daymean','weekmean','hp48','temperature_past','humidity_past','has_pv','household_code']
def log(s): print(s,flush=True)
def pv(v):
    s=str(v).strip().lower()
    return 1 if s in ['true','1','1.0','yes','ja'] else 0 if s in ['false','0','0.0','no','nein'] else -1
def model(iters,quantile=None):
    kw=dict(max_iter=iters,max_leaf_nodes=23,learning_rate=.08,min_samples_leaf=40,l2_regularization=2,early_stopping=False,random_state=42)
    if quantile is not None: kw.update(loss='quantile',quantile=quantile)
    return HistGradientBoostingRegressor(**kw)
def cost(y,p,da=.1,buy=.18,sell=.06):
    return p*da+np.maximum(y-p,0)*buy-np.maximum(p-y,0)*sell
def metric(y,p,name,scope):
    return dict(model=name,scope=scope,n=len(y),MAE_kWh=float(mean_absolute_error(y,p)),RMSE_kWh=float(np.sqrt(mean_squared_error(y,p))),R2=float(r2_score(y,p)),WAPE=float(np.abs(np.asarray(y)-np.asarray(p)).sum()/np.asarray(y).sum()))
def prepare(data,out,cap):
    meta=pd.read_csv(data/'smart_meter_meta_data/households.csv',sep=';').set_index('Household_ID')
    ov=pd.read_csv(data/'smart_meter_meta_data/smart_meter_data_15min_overview.csv',sep=';')
    a=pd.to_datetime(ov.SMD_15min_TimeAvailable_EarliestTimestamp,utc=True).min().tz_convert(TZ).tz_localize(None).normalize()
    b=pd.to_datetime(ov.SMD_15min_TimeAvailable_LatestTimestamp,utc=True).max().tz_convert(TZ).tz_localize(None).normalize()
    days=pd.date_range(a,b); cs=days[int(.6*len(days))]; ts=days[int(.8*len(days))]; te=cs-pd.Timedelta(days=2); ce=ts-pd.Timedelta(days=2)
    log(f'Train < {te.date()}; calibration {cs.date()} to < {ce.date()}; test >= {ts.date()}')
    weather={}
    for f in sorted((data/'weather_data_hourly').glob('*.csv')):
        w=pd.read_csv(f,sep=';'); w['Timestamp']=pd.to_datetime(w.Timestamp,utc=True)
        weather[f.stem]=w.drop_duplicates('Timestamp').set_index('Timestamp').sort_index()
    cache=out/'.cache'; cache.mkdir(exist_ok=True); trparts=[]; evfiles=[]; audit=[]; files=sorted((data/'15min').glob('*.csv')); rng=np.random.default_rng(42); per=max(1,cap//len(files))
    for code,f in enumerate(files):
        d=pd.read_csv(f,sep=';',usecols=['Household_ID','Timestamp',T,'kWh_received_HeatPump'])
        d['Timestamp']=pd.to_datetime(d.Timestamp,utc=True,errors='coerce'); d[T]=pd.to_numeric(d[T],errors='coerce')
        raw=len(d); bad=d.Timestamp.isna()|d[T].isna()|(d[T]<0); nb=int(bad.sum()); d=d.loc[~bad].sort_values('Timestamp')
        nd=int(d.duplicated('Timestamp').sum()); d=d.drop_duplicates('Timestamp').set_index('Timestamp')
        if not len(d):
            audit.append(dict(Household_ID=int(f.stem),raw_rows=raw,invalid_rows=nb,duplicate_rows=nd,eligible_rows=0,training_sample=0,calibration_rows=0,test_rows=0,has_pv=pv(meta.loc[int(f.stem),'Installation_HasPVSystem']) if int(f.stem) in meta.index else -1)); continue
        hid=int(d.Household_ID.iloc[0]); local=d.index.tz_convert(TZ); day=local.tz_localize(None).normalize()
        origins=pd.DatetimeIndex(day-pd.Timedelta(days=1)+pd.Timedelta(hours=12)).tz_localize(TZ).tz_convert('UTC')
        assert ((d.index-pd.Timedelta(hours=48))<origins).all()
        assert ((d.index-pd.Timedelta(hours=168))<origins).all()
        x=pd.DataFrame(index=d.index); x['Household_ID']=hid; x['day']=day; x[T]=d[T].to_numpy(dtype='float32')
        x['slot']=local.hour*4+local.minute//15; x['dow']=local.dayofweek; x['month']=local.month; x['weekend']=(local.dayofweek>=5).astype(int)
        x['slot_sin']=np.sin(2*np.pi*x.slot/96); x['slot_cos']=np.cos(2*np.pi*x.slot/96)
        x['year_sin']=np.sin(2*np.pi*local.dayofyear/365.25); x['year_cos']=np.cos(2*np.pi*local.dayofyear/365.25)
        x['lag48']=d[T].reindex(d.index-pd.Timedelta(hours=48)).to_numpy(); x['lag168']=d[T].reindex(d.index-pd.Timedelta(hours=168)).to_numpy()
        x['hp48']=pd.to_numeric(d.kWh_received_HeatPump,errors='coerce').reindex(d.index-pd.Timedelta(hours=48)).to_numpy()
        daily=d[T].groupby(day).mean().reindex(pd.date_range(day.min(),day.max()))
        x['daymean']=daily.reindex(day-pd.Timedelta(days=2)).to_numpy(); x['weekmean']=daily.rolling(7,min_periods=3).mean().reindex(day-pd.Timedelta(days=2)).to_numpy()
        flag=pv(meta.loc[hid,'Installation_HasPVSystem']) if hid in meta.index else -1
        x['has_pv']=flag; x['household_code']=code
        wid=str(meta.loc[hid,'Weather_ID']) if hid in meta.index else ''; w=weather.get(wid); wt=origins-pd.Timedelta(hours=2)
        for col,dest in [('Temperature_avg_hourly','temperature_past'),('Humidity_avg_hourly','humidity_past')]:
            x[dest]=w[col].reindex(wt,method='ffill',tolerance=pd.Timedelta(hours=24)).to_numpy() if w is not None else np.nan
        ok=x.lag48.notna()&x.lag168.notna(); counts=ok.groupby(day).sum(); totals=x.groupby('day').size(); ud=pd.DatetimeIndex(counts.index)
        expected=((ud+pd.Timedelta(days=1)).tz_localize(TZ).tz_convert('UTC')-ud.tz_localize(TZ).tz_convert('UTC'))/pd.Timedelta(minutes=15)
        eligible=counts.index[(counts.to_numpy()==expected)&(totals.reindex(counts.index).to_numpy()==expected)]
        x=x[x.day.isin(eligible)].copy()
        x['split']=np.select([x.day<te,(x.day>=cs)&(x.day<ce),x.day>=ts],['train','calibration','test'],default='gap')
        tr=x[x.split=='train'].copy(); ev=x[x.split.isin(['calibration','test'])].copy()
        if len(tr)>per: tr=tr.iloc[np.sort(rng.choice(len(tr),per,replace=False))].copy()
        tr[F]=tr[F].astype('float32'); ev[F]=ev[F].astype('float32'); trparts.append(tr); cachefile=cache/f'{hid}.pkl'; ev.to_pickle(cachefile); evfiles.append(cachefile)
        audit.append(dict(Household_ID=hid,raw_rows=raw,invalid_rows=nb,duplicate_rows=nd,eligible_rows=len(x),training_sample=len(tr),calibration_rows=int((ev.split=='calibration').sum()),test_rows=int((ev.split=='test').sum()),has_pv=flag))
        if (code+1)%50==0: log(f'Prepared {code+1}/{len(files)} households')
    pd.DataFrame(audit).to_csv(out/'data_audit.csv',index=False)
    info=dict(files=len(files),train_end_exclusive=str(te.date()),calibration_start=str(cs.date()),calibration_end_exclusive=str(ce.date()),test_start=str(ts.date()),data_last_day=str(b.date()),seed=42,features=F)
    return pd.concat(trparts),evfiles,info

class Accumulator:
    def __init__(self): self.n=0; self.sy=0.; self.sy2=0.; self.ae=0.; self.se=0.
    def add(self,y,p):
        y=np.asarray(y,dtype=float); p=np.asarray(p,dtype=float); self.n+=len(y); self.sy+=y.sum(); self.sy2+=(y*y).sum(); self.ae+=np.abs(y-p).sum(); self.se+=((y-p)**2).sum()
    def result(self,name,scope):
        return dict(model=name,scope=scope,n=self.n,MAE_kWh=self.ae/self.n,RMSE_kWh=np.sqrt(self.se/self.n),R2=1-self.se/(self.sy2-self.sy**2/self.n),WAPE=self.ae/self.sy)

def run(args):
    data=Path(args.data); out=Path(args.out); out.mkdir(parents=True,exist_ok=True)
    train,evfiles,info=prepare(data,out,args.train_cap); info['training_sample_rows']=len(train)
    log(f'Training sample {len(train):,}; evaluation cached across {len(evfiles)} household files')
    X=train[F]; y=train[T]; models={}; models['global']=model(args.iters).fit(X,y); log('Level 0 fitted')
    for flag in [0,1]:
        mask=train.has_pv==flag
        if mask.sum()>=500: models[f'pv_{flag}']=model(args.iters).fit(X.loc[mask],y.loc[mask]); log(f'Level 1 PV={flag} fitted')
    for q,n in [(.1,'p10'),(.5,'p50'),(.9,'p90')]: models[n]=model(args.iters,q).fit(X,y); log(f'Level 3 {n} fitted')
    names=['baseline48','baseline_week','level0','level1']; meters={n:Accumulator() for n in names}; groups={}; port=None
    total=0; calrows=0; testhh=set(); cover_sum=0.; width_sum=0.; score_sum=0.; crossing=0; pin={n:0. for n in ['p10','p50','p90']}; uncertainty_parts=[]; sample=None
    with gzip.open(out/'household_predictions.csv.gz','wt') as output:
        header=True
        for i,f in enumerate(evfiles):
            ev=pd.read_pickle(f)
            if not len(ev): continue
            ev['level0']=np.maximum(0,models['global'].predict(ev[F])); ev['level1']=ev.level0
            for flag in [0,1]:
                dest=ev.has_pv==flag
                if f'pv_{flag}' in models and dest.any(): ev.loc[dest,'level1']=np.maximum(0,models[f'pv_{flag}'].predict(ev.loc[dest,F]))
            ev['baseline48']=ev.lag48; ev['baseline_week']=ev.lag168
            for n in ['p10','p50','p90']: ev[n]=np.maximum(0,models[n].predict(ev[F]))
            ts=ev.split=='test'; crossing+=int((((ev.p10>ev.p50)|(ev.p50>ev.p90))&ts).sum()); ev[['p10','p50','p90']]=np.sort(ev[['p10','p50','p90']].to_numpy(),axis=1)
            tst=ev[ts]; calrows+=int((ev.split=='calibration').sum())
            if len(tst):
                hid=int(tst.Household_ID.iloc[0]); flag=int(tst.has_pv.iloc[0]); testhh.add(hid); total+=len(tst)
                for n in names:
                    meters[n].add(tst[T],tst[n]); groups.setdefault((flag,n),Accumulator()).add(tst[T],tst[n])
                cover=(tst[T]>=tst.p10)&(tst[T]<=tst.p90); width=tst.p90-tst.p10; cover_sum+=cover.sum(); width_sum+=width.sum(); score_sum+=(width+10*np.maximum(tst.p10-tst[T],0)+10*np.maximum(tst[T]-tst.p90,0)).sum()
                for q,n in [(.1,'p10'),(.5,'p50'),(.9,'p90')]: pin[n]+=mean_pinball_loss(tst[T],tst[n],alpha=q)*len(tst)
                uncertainty_parts.append(tst.assign(width=width,covered=cover).groupby(['slot','has_pv']).agg(width_sum=('width','sum'),cover_sum=('covered','sum'),n=('width','size')))
                tst.reset_index()[['Timestamp','Household_ID','day',T,'has_pv']+names+['p10','p50','p90']].to_csv(output,index=False,header=header); header=False
                if sample is None and len(tst)>=288: sample=tst.iloc[:288].copy()
            ep=ev.groupby(['split',ev.index])[[T]+names].sum().rename(columns={T:'actual'}); ep['households']=ev.groupby(['split',ev.index]).Household_ID.nunique()
            port=ep if port is None else port.add(ep,fill_value=0)
            if (i+1)%50==0: log(f'Evaluated {i+1}/{len(evfiles)} households')
    stats=[meters[n].result(n,'household') for n in names]
    pd.DataFrame([dict(PV=flag,**v.result(n,'household')) for (flag,n),v in groups.items() if n in ['level0','level1']]).to_csv(out/'level1_pv_comparison.csv',index=False)
    port=port.reset_index(); cp=port[port.split=='calibration']; tp=port[port.split=='test'].copy()
    selected=min(['level0','level1'],key=lambda n: cost(cp.actual.to_numpy(),cp[n].to_numpy()).sum()); info['selected_on_calibration']=selected
    residual=(cp.actual-cp[selected])/cp.households; lo,hi=np.quantile(residual,[.1,.9]); q=(.18-.1)/(.18-.06); adj=float(np.quantile(residual,q))
    tp['portfolio_p10']=np.maximum(0,tp[selected]+lo*tp.households); tp['portfolio_p90']=np.maximum(tp.portfolio_p10,np.maximum(0,tp[selected]+hi*tp.households)); tp['risk_adjusted']=np.maximum(0,tp[selected]+adj*tp.households)
    business=[]; perfect=float((tp.actual*.1).sum()); base=float(cost(tp.actual.to_numpy(),tp.baseline_week.to_numpy()).sum())
    for n in names+['risk_adjusted']:
        c=float(cost(tp.actual.to_numpy(),tp[n].to_numpy()).sum()); stats.append(metric(tp.actual,tp[n],n,'portfolio'))
        business.append(dict(model=n,total_cost_EUR=c,perfect_cost_EUR=perfect,extra_cost_EUR=c-perfect,saving_vs_weekly_EUR=base-c,shortage_kWh=float(np.maximum(tp.actual-tp[n],0).sum()),surplus_kWh=float(np.maximum(tp[n]-tp.actual,0).sum())))
    pd.DataFrame(stats).to_csv(out/'forecast_metrics.csv',index=False); pd.DataFrame(business).to_csv(out/'level2_procurement.csv',index=False)
    sens=[]
    for buy,sell in [(.14,.08),(.18,.06),(.25,.04)]:
        quant=(buy-.1)/(buy-sell); ad=np.quantile(residual,quant); decision=np.maximum(0,tp[selected].to_numpy()+ad*tp.households.to_numpy())
        for n,p in [(selected,tp[selected].to_numpy()),('risk_adjusted',decision)]: sens.append(dict(buy=buy,sell=sell,optimal_quantile=quant,model=n,cost_EUR=float(cost(tp.actual.to_numpy(),p,buy=buy,sell=sell).sum())))
    pd.DataFrame(sens).to_csv(out/'price_sensitivity.csv',index=False)
    uncertainty=dict(nominal_coverage=.8,household_coverage=float(cover_sum/total),household_mean_width_kWh=float(width_sum/total),mean_interval_score=float(score_sum/total),portfolio_coverage=float(((tp.actual>=tp.portfolio_p10)&(tp.actual<=tp.portfolio_p90)).mean()),portfolio_mean_width_kWh=float((tp.portfolio_p90-tp.portfolio_p10).mean()),procurement_quantile=q,calibration_adjustment_per_household_kWh=adj,pre_sort_crossing_rate=crossing/total)
    uncertainty.update({'pinball_'+n:v/total for n,v in pin.items()})
    summary=dict(**info,test_rows=total,test_households=len(testhh),calibration_rows=calrows,test_start_actual=str(tp.Timestamp.min()),test_end_actual=str(tp.Timestamp.max()),uncertainty=uncertainty,prices_EUR_per_kWh=dict(day_ahead=.1,intraday_buy=.18,intraday_sell=.06))
    summary['git_commit']=subprocess.check_output(['git','rev-parse','HEAD'],cwd=data.resolve().parent,text=True).strip(); (out/'summary.json').write_text(json.dumps(summary,indent=2))
    tp.to_csv(out/'portfolio_predictions.csv',index=False)
    ug=pd.concat(uncertainty_parts).groupby(level=[0,1]).sum(); ug['width']=ug.width_sum/ug.n; ug['coverage']=ug.cover_sum/ug.n; ug.to_csv(out/'uncertainty_by_time_pv.csv')
    joblib.dump(dict(models=models,features=F,summary=summary,household_codes={int(f.stem):i for i,f in enumerate(sorted((data/'15min').glob('*.csv')))}),out/'models.joblib',compress=3)
    plt.style.use('seaborn-v0_8-whitegrid'); ms=pd.DataFrame(stats); hh=ms[ms.scope=='household']
    fig,ax=plt.subplots(figsize=(9,5)); ax.bar(hh.model,hh.MAE_kWh); ax.set_ylabel('MAE [kWh / 15 minutes]'); ax.set_title('Levels 0–1: held-out household forecasts'); fig.tight_layout(); fig.savefig(out/'01_model_comparison.png',dpi=150); plt.close(fig)
    fig,ax=plt.subplots(figsize=(12,5)); ax.plot(sample.index,sample[T],label='Actual',lw=1); ax.plot(sample.index,sample.p50,label='Median',lw=1); ax.fill_between(sample.index,sample.p10,sample.p90,alpha=.25,label='P10–P90'); ax.legend(); ax.set_ylabel('kWh / 15 minutes'); ax.set_title(f'Level 3: household {int(sample.Household_ID.iloc[0])}'); fig.autofmt_xdate(); fig.tight_layout(); fig.savefig(out/'02_household_uncertainty.png',dpi=150); plt.close(fig)
    pp=tp.iloc[:672]; fig,ax=plt.subplots(figsize=(12,5)); ax.plot(pp.Timestamp,pp.actual,label='Actual',lw=1); ax.plot(pp.Timestamp,pp[selected],label=selected,lw=1); ax.fill_between(pp.Timestamp,pp.portfolio_p10,pp.portfolio_p90,alpha=.25,label='Calibration-based interval'); ax.legend(); ax.set_ylabel('Portfolio kWh / 15 minutes'); fig.autofmt_xdate(); fig.tight_layout(); fig.savefig(out/'03_portfolio_forecast.png',dpi=150); plt.close(fig)
    br=pd.DataFrame(business); fig,ax=plt.subplots(figsize=(9,5)); ax.bar(br.model,br.extra_cost_EUR); ax.set_ylabel('Extra cost versus perfect forecast [EUR]'); ax.set_title('Illustrative procurement-price scenario'); fig.tight_layout(); fig.savefig(out/'04_procurement_cost.png',dpi=150); plt.close(fig)
    shutil.rmtree(out/'.cache'); log(ms.to_string(index=False)); log(br.to_string(index=False)); log(json.dumps(uncertainty,indent=2))
if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--data',default='data'); p.add_argument('--out',default='solution/results'); p.add_argument('--train-cap',type=int,default=300000); p.add_argument('--iters',type=int,default=130); run(p.parse_args())
