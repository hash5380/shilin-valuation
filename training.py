"""Train & export three residential valuation models from Taiwan MOI transactions.

Run: python training.py --cleaned /path/to/cleaned.csv
 OR: python training.py --input-dir /path/to/zip-archives
Admin on Render uses the latter. OOF target encoding only uses each inner train fold.
"""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
from datetime import date
import numpy as np
import pandas as pd
import statsmodels.api as sm
import joblib
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score
from study_core import (load_sources, clean_data, X_basic, get_models,
                        FEATURES, encode_road, make_training_te, road_zone, ZONES)

ROOT=Path(__file__).resolve().parent
MODEL_DIR=ROOT/'model'
NAMES=('Hedonic OLS','Random Forest','Gradient Boosting')

def evaluate(df):
    """5-fold out-of-fold validation; returns numbers computed in unit price scale."""
    df=df.reset_index(drop=True)
    X=X_basic(df)
    predictions={name:np.zeros(len(df)) for name in NAMES}
    kfold=KFold(n_splits=5,shuffle=True,random_state=42)
    for fi,(tr,te) in enumerate(kfold.split(df),start=1):
        train,test=df.iloc[tr].reset_index(drop=True),df.iloc[te].reset_index(drop=True)
        xtr,xte=X.iloc[tr].reset_index(drop=True),X.iloc[te].reset_index(drop=True)
        y=train['ln_單價'].to_numpy()
        base=sm.OLS(y,sm.add_constant(xtr,has_constant='add')).fit()
        predictions['Hedonic OLS'][te]=np.exp(np.clip(np.asarray(base.predict(sm.add_constant(xte,has_constant='add'))),-10,15))
        extra_tr=make_training_te(train,seed=42+fi)
        extra_te=encode_road(train,test,k=5)
        xtr2=xtr.copy(); xtr2['路段平滑平均ln單價']=extra_tr
        xte2=xte.copy(); xte2['路段平滑平均ln單價']=extra_te
        for name in NAMES[1:]:
            reg=get_models()[name];reg.fit(xtr2,y)
            predictions[name][te]=np.exp(np.clip(reg.predict(xte2),-10,15))
    actual=df['單價萬元坪'].to_numpy()
    result={}
    for name,pred in predictions.items():
        ape=np.abs((actual-pred)/actual)
        result[name]={
            'mape_pct':round(float(100*ape.mean()),3),
            'medape_pct':round(float(100*np.median(ape)),3),
            'r2':round(float(r2_score(actual,pred)),4),
            'within_10_pct':round(float(100*(ape<=.1).mean()),2),
            'mae_10k_ping':round(float(np.abs(actual-pred).mean()),3)
        }
    return result

def train_and_export(df, path=MODEL_DIR, use_metrics=None):
    df=df.reset_index(drop=True)
    if len(df)<100:
        raise ValueError(f'有效住宅樣本只有 {len(df)} 筆，低於 100 筆的訓練門檻。')
    path=Path(path);path.mkdir(parents=True,exist_ok=True)
    X=X_basic(df)
    yy=df['ln_單價'].to_numpy()
    ols=sm.OLS(yy,sm.add_constant(X,has_constant='add')).fit(cov_type='HC1')
    model_features=list(X.columns)
    ols_file={'const':float(ols.params['const']), 'features':model_features,
              'coefficients':{f:float(ols.params[f]) for f in model_features}}
    # training target encoding is out-of-fold; refit inference priors from all observed data
    xx=X.copy();xx['路段平滑平均ln單價']=make_training_te(df,seed=2026)
    rf=get_models()['Random Forest'];rf.fit(xx,yy)
    gb=get_models()['Gradient Boosting'];gb.fit(xx,yy)
    joblib.dump(rf,path/'random_forest.joblib',compress=3)
    joblib.dump(gb,path/'gradient_boosting.joblib',compress=3)
    (path/'hedonic_ols.json').write_text(json.dumps(ols_file,ensure_ascii=False,indent=2),encoding='utf8')
    # Only aggregate road / zone statistics: no personal addresses are shipped with website
    zone_stats=df.groupby('生活圈')['ln_單價'].agg(['sum','count']).to_dict('index')
    road_stats=df.groupby('路名')['ln_單價'].agg(['sum','count']).to_dict('index')
    (path/'road_priors.json').write_text(json.dumps({
        'global_ln_mean':float(df['ln_單價'].mean()),
        'zone':{k:{'sum':float(v['sum']),'count':int(v['count'])} for k,v in zone_stats.items()},
        'road':{k:{'sum':float(v['sum']),'count':int(v['count'])} for k,v in road_stats.items()},
        'k':5
    },ensure_ascii=False),encoding='utf8')
    if use_metrics is None:
        cv=evaluate(df)
    else:
        cv=use_metrics
    parking=df.loc[(df['含車位']==1)&(df['車位總價元_num']>0),'車位總價元_num']/10000
    months=pd.to_datetime((df['交易民國年']+1911).astype(int).astype(str)+'-'+df['交易月'].astype(int).astype(str)+'-01')
    latest_ym=str(months.max().strftime('%Y-%m'))
    first_ym=str(months.min().strftime('%Y-%m'))
    observed_trend=float(ols.params['交易月份'])
    projection_trend=float(np.clip(observed_trend,-0.01,0.01))
    meta={
        'title':'士林區 2025–2026 Q3 住宅估價（教學研究）',
        'records':len(df),'market':'臺北市士林區',
        'price_basis':'建物移轉總面積扣除車位移轉面積，單位萬元/坪',
        'sample_dates':{'from':str(months.min().date()),'to':max(f"{int(y+1911):04d}-{int(m):02d}-{int(d):02d}" for y,m,d in df[['交易民國年','交易月','交易日']].itertuples(index=False,name=None))},
        'supported_months':first_ym+' 至 '+latest_ym+'（已觀察交易）；最新成交月份後最多 3 個月僅能做短期趨勢推估',
        'latest_month':latest_ym,'earliest_month':'2024-07',
        'latest_month_observation_count':int(sum(months.dt.strftime('%Y-%m')==latest_ym)),
        'projection':{'trend_ln_month_raw':observed_trend,'trend_ln_month_applied':projection_trend,'max_horizon_months':3,'method':'對最新觀察月份的模型估價乘上 exp(截尾月趨勢 × 月數)，月趨勢由 Hedonic OLS 訓練樣本估計，上下限 ±1%/月；屬情境估算、未經獨立校驗，非即時行情。'},
        'model_names':list(NAMES),'metrics':cv,
        'trade_year_counts':{str(int(k)+1911):int(v) for k,v in df['交易民國年'].value_counts().items()},
        'source_quarter_counts':df['季度'].value_counts().astype(int).to_dict(),
        'road_counts':df['路名'].value_counts().astype(int).to_dict(),
        'zone_counts':df['生活圈'].value_counts().astype(int).to_dict(),
        'type_counts':df['建物三類'].value_counts().astype(int).to_dict(),
        'parking':{'n_separate':int(len(parking)),'median_10k':float(parking.median()) if len(parking) else None},
        'contains_elevator_feature':False,
        'elevator_source_col_present':'電梯' in df.columns,
        'elevator_type_crosstab':{t:df.loc[df['建物三類']==t,'電梯'].value_counts().astype(int).to_dict() for t in df['建物三類'].unique()} if '電梯' in df.columns else {},
        'variables':model_features,
        'limitations':[
            '已納入 2025 年第 1–4 季與 2026 年第 1–3 季登錄檔；季度檔包含前期揭露的實際成交紀錄。',
            '只有士林區住宅樣本，不支援其他行政區及未知道路。',
            '原始資料有電梯欄位，但有效住宅樣本中其有無與建物型態完全重疊，無法識別獨立的電梯溢價；裝潢、採光、景觀、精確座標亦未納入。',
            '五折隨機交叉驗證非嚴格時間外推；歷史測試誤差不是個別房屋的保證誤差。',
            '2026 年 10 月的估價為截至最新觀察月份基礎上最多 3 個月的短期趨勢推估，沒有 10 月實價登錄支撐；不等於即時成交價。'
        ]
    }
    (path/'metadata.json').write_text(json.dumps(meta,ensure_ascii=False,indent=2),encoding='utf8')
    return meta

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--cleaned',help='已依原研究篩選得到的有效樣本 CSV')
    parser.add_argument('--input-dir',help='含內政部官方 ZIP 的目錄')
    parser.add_argument('--output-dir',default=str(MODEL_DIR))
    parser.add_argument('--reuse-study-metrics',help='已有的 5-fold CV 比較 CSV；僅適用同批樣本')
    args=parser.parse_args()
    if bool(args.cleaned)==bool(args.input_dir):parser.error('必須選擇 --cleaned 或 --input-dir')
    if args.cleaned:
        df=pd.read_csv(args.cleaned)
    else:
        raw,logs=load_sources(args.input_dir)
        df,counts,meta=clean_data(raw)
        print(counts.to_string(index=False))
        print('來源檔案：',logs)
    cv=None
    if args.reuse_study_metrics:
        summary=pd.read_csv(args.reuse_study_metrics)
        cv={row['模型']:{'mape_pct':round(float(row['MAPE_%']),3),
            'medape_pct':round(float(row['MedAPE_%']),3),
            'r2':round(float(row['R2_價格尺度']),4),
            'within_10_pct':round(float(row['10%內比例_%']),2),
            'mae_10k_ping':round(float(row['MAE_萬元坪']),3)} for _,row in summary.iterrows()}
    res=train_and_export(df,args.output_dir,cv)
    print(json.dumps({'records':res['records'],'metrics':res['metrics'],'parking':res['parking']},ensure_ascii=False,indent=2))
if __name__=='__main__':main()
