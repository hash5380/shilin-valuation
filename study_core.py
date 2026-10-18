#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Taipei Shilin residential hedonic versus ML: reproducible 2025–2026 analysis.
Usage: python run_analysis.py --input-dir /path/to/zip-directory --output-dir /path/to/results
Requires pandas numpy scipy statsmodels scikit-learn matplotlib. The three 114S2-114S4 archive names may vary.
Legacy XLS is intentionally excluded if it actually contains year 115 (2026).
"""
import argparse, json, re, zipfile, math, warnings
from pathlib import Path
from collections import defaultdict
import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_percentage_error, r2_score

ZONES={
 '天母/芝山':('中山北路六段 中山北路七段 天母東路 天母西路 天母北路 天玉街 忠誠路一段 忠誠路二段 德行東路 德行西路 磺溪街 克強路 中庸一路 中庸二路 士東路 雨聲街 雨農路 芝玉路一段 芝玉路二段 福華路 福志路 華齡街').split(),
 '士林站/夜市':('中正路 文林路 大東路 大南路 大北路 基河路 承德路四段 劍潭路 小北街 小西街 前街 後街 大西路 福德路 文昌路 福林路 中山北路五段 福國路 忠義街 後港街 重慶北路四段').split(),
 '社子/延平':('延平北路五段 延平北路六段 延平北路八段 社子街 社中街 社正路 中社路一段 中社路二段 通河街 通河東街一段 通河東街二段 通河西街二段 福港街 環河北路三段 葫蘆街 葫東街 士商路 永平街 倫等街').split(),
 '外雙溪/陽明山':('至誠路一段 至誠路二段 格致路 菁山路 仰德大道三段 東山路 翠山街 雙溪街 明溪街 光華路 故宮路').split(),
}
ZONE_ORDER=['天母/芝山','士林站/夜市','社子/延平','外雙溪/陽明山','未分類路段']
TYPE_ORDER=['公寓','華廈','住宅大樓']
ROADS=sorted(((road,zone) for zone,roads in ZONES.items() for road in roads),key=lambda x:-len(x[0]))
REMARK_EXCLUDE=['親友','特殊關係','共有人','預售屋','分件登記','公共設施保留地','毛胚','政府機關','地清','未辦繼承','未登記建物','都更','重建','夾層','頂樓加蓋','瑕疵','凶宅','法拍','身故']
CNUM={'零':0,'〇':0,'○':0,'一':1,'二':2,'兩':2,'三':3,'四':4,'五':5,'六':6,'七':7,'八':8,'九':9}
FEATURE_BASE=['ln_坪數','屋齡','屋齡平方除100','樓層','一樓','頂樓','房數','管委會','含車位','車位價未拆分','陽台外推或增建','交易月份']
DUMMY_TYPES=['型態_華廈','型態_住宅大樓']
DUMMY_ZONES=[f'生活圈_{z}' for z in ZONE_ORDER if z not in ('天母/芝山','未分類路段')]
FEATURES=FEATURE_BASE+DUMMY_TYPES+DUMMY_ZONES


def from_roc(v):
 if pd.isna(v):return (np.nan,np.nan,np.nan)
 digits=re.sub(r'\D','',str(v))
 if not digits:return (np.nan,np.nan,np.nan)
 digits=digits.zfill(7)
 try:
  year=int(digits[:-4]);mon=int(digits[-4:-2]);day=int(digits[-2:]);
  if year<1 or not 1<=mon<=12 or not 1<=day<=31:return (np.nan,np.nan,np.nan)
  return year,mon,day
 except (ValueError,TypeError):return (np.nan,np.nan,np.nan)

def chinum(v):
 if not v:return None
 if v.isdecimal():return int(v)
 if '百' in v:
  a,b=v.split('百',1);h=CNUM.get(a,1) if a else 1
  return h*100+(chinum(b) or 0)
 if '十' in v:
  a,b=v.split('十',1);t=CNUM.get(a,1) if a else 1
  return t*10+(chinum(b) or 0)
 out=0
 for c in v:
  if c not in CNUM:return None
  out=out*10+CNUM[c]
 return out

def parse_floors(x):
 if pd.isna(x):return np.nan
 s=str(x).replace('層次','層').replace('樓','層')
 m=re.findall(r'([零一二兩三四五六七八九十百〇○\d]+)層',s)
 a=[chinum(v) for v in m]
 a=[v for v in a if v is not None]
 return float(min(a)) if a else np.nan

def road_zone(addr):
 if pd.isna(addr):return ('','未分類路段')
 a=str(addr).replace('臺','台')
 for road,zone in ROADS:
  if road in a:return road,zone
 # Even for unmatched streets retain a road identity for smoothing to area mean
 m=re.search(r'([^區鄉鎮市路街巷弄號\s\d]{1,9}(?:路|街|大道)(?:[一二三四五六七八九十\d]+段)?)',a)
 return (m.group(1) if m else '', '未分類路段')

def load_sources(input_dir):
 """Load official main tables, including nested DATA.zip -> quarterly ZIP archives.

 Quarter identity is verified using the official folder marker (e.g. 114S1, 115S1)
 when available. Duplicate record IDs (編號) are kept only once.
 """
 import io
 from collections import Counter
 inp=Path(input_dir)
 archives=sorted(inp.glob('*.zip')) if inp.is_dir() else [inp]
 dfs=[]; log=[]; loaded=set()

 def read_inner(archive_bytes, display_name):
  with zipfile.ZipFile(io.BytesIO(archive_bytes)) as z:
   names=z.namelist()
   tab=next((name for name in names if name.rsplit('/',1)[-1].lower()=='a_lvr_land_a.csv'),None)
   if tab is None:
    return False
   quarter=next((n.strip('/') for n in names if re.fullmatch(r'11[45]S[1-4]/?',n)),None)
   fingerprint=(quarter, len(archive_bytes))
   if fingerprint in loaded:return True
   with z.open(tab) as f:
    df=pd.read_csv(f,dtype=str,skiprows=[1],encoding='utf-8-sig',low_memory=False)
   if '鄉鎮市區' not in df or '交易年月日' not in df or '編號' not in df:
    raise ValueError(f'主表欄位缺失: {display_name}')
   df['來源檔案']=display_name
   df['季度']=quarter or 'unknown'
   dfs.append(df);loaded.add(fingerprint)
   log.append({'file':display_name,'quarter':quarter,'status':'included CSV',
       'rows':int(len(df)),'shilin_rows':int((df['鄉鎮市區']=='士林區').sum())})
   return True

 for arch in archives:
  if not zipfile.is_zipfile(arch):continue
  contents=arch.read_bytes()
  if read_inner(contents,arch.name):continue
  with zipfile.ZipFile(io.BytesIO(contents)) as outer:
   found=0
   for n in outer.namelist():
    if n.lower().endswith('.zip'):
     data=outer.read(n)
     if zipfile.is_zipfile(io.BytesIO(data)) and read_inner(data,arch.name+'/'+n):found+=1
   if found==0:log.append({'file':arch.name,'status':'no valid A_lvr_land_A.csv; skipped'})
 if not dfs:raise ValueError('找不到官方 A_lvr_land_A.csv')
 whole=pd.concat(dfs,ignore_index=True)
 raw_n=len(whole)
 # 編號 is the registration-record ID. Only true ID duplicates are removed;
 # similar prices/addresses may correspond to separate genuine units.
 whole=whole.drop_duplicates(subset=['編號'],keep='last').reset_index(drop=True)
 log.append({'step':'dedup_on_編號','before':raw_n,'after':len(whole),'removed':raw_n-len(whole)})
 return whole,log

def clean_data(df):
 history=[]
 def record(name,x):history.append({'步驟':name,'剩餘筆數':int(len(x)),'本步排除筆數': int((history[-1]['剩餘筆數'] if history else len(df))-len(x))})
 record('合併可讀主表',df)
 df=df[df['鄉鎮市區']=='士林區'].copy();record('0. 僅保留士林區',df)
 df=df[df['交易標的'].isin(['房地(土地+建物)','房地(土地+建物)+車位'])].copy();record('1. 房地或房地+車位',df)
 df=df[df['主要用途'].isin(['住家用','住商用'])].copy();record('2. 住家用或住商用',df)
 def classify(s):
  if pd.isna(s):return None
  if str(s).startswith('公寓'):return '公寓'
  if str(s).startswith('華廈'):return '華廈'
  if str(s).startswith('住宅大樓'):return '住宅大樓'
  return None
 df['建物三類']=df['建物型態'].map(classify)
 df=df[df['建物三類'].notna()].copy();record('3. 公寓/華廈/住宅大樓',df)
 df=df[~df['備註'].fillna('').astype(str).str.contains('|'.join(map(re.escape,REMARK_EXCLUDE)),regex=True)].copy();record('4. 排除指定備註關鍵字',df)
 tx_dates=df['交易年月日'].apply(from_roc)
 df[['交易民國年','交易月','交易日']]=pd.DataFrame(tx_dates.tolist(),index=df.index)
 df=df[((df['交易民國年']>113)|((df['交易民國年']==113)&(df['交易月']>=7)))].copy();record('5. 交易日期>=113/7',df)
 df[['完工民國年','完工月','完工日']]=pd.DataFrame(df['建築完成年月'].apply(from_roc).tolist(),index=df.index)
 df['屋齡']= (df['交易民國年']-df['完工民國年'])+(df['交易月']-df['完工月'])/12
 df['屋齡']=df['屋齡'].clip(lower=0)
 for name in ['總價元','車位總價元','建物移轉總面積平方公尺','車位移轉總面積平方公尺','建物現況格局-房']:
  df[name+'_num']=pd.to_numeric(df[name],errors='coerce').fillna(0 if name in ('車位總價元','車位移轉總面積平方公尺') else np.nan)
 df['淨坪數']=(df['建物移轉總面積平方公尺_num']-df['車位移轉總面積平方公尺_num'])*0.3025
 df['單價萬元坪']=(df['總價元_num']-df['車位總價元_num'])/df['淨坪數']/10000
 df=df[df['淨坪數'].between(8,200)&df['屋齡'].notna()&(df['單價萬元坪']>0)&np.isfinite(df['單價萬元坪'])].copy();record('6. 坪數8–200、屋齡可算、有效單價',df)
 lo,hi=df['單價萬元坪'].quantile([.01,.99]);df=df[df['單價萬元坪'].between(lo,hi,inclusive='both')].copy();record('7. 每坪單價去除兩端各1%',df)
 df['樓層']=df['移轉層次'].apply(parse_floors)
 df['總樓層']=df['總樓層數'].apply(parse_floors)
 df['房數']=df['建物現況格局-房_num'].clip(lower=0,upper=6)
 # require model variables same for all 3 models, and positive normal above-ground floor
 df=df[df['樓層'].notna()&df['總樓層'].notna()&df['房數'].notna() & (df['樓層']>=1)&(df['總樓層']>=1)&(df['樓層']<=df['總樓層'])].copy()
 record('8. 模型必需樓層/房數有效（額外）',df)
 df['一樓']=(df['樓層']==1).astype(int)
 df['頂樓']=((df['樓層']==df['總樓層'])&(df['總樓層']>1)).astype(int)
 df['管委會']=(df['有無管理組織']=='有').astype(int)
 df['含車位']=(df['交易標的']=='房地(土地+建物)+車位').astype(int)
 df['車位價未拆分']=((df['含車位']==1)&(df['車位總價元_num']==0)).astype(int)
 df['陽台外推或增建']=df['備註'].fillna('').str.contains('陽台外推|增建').astype(int)
 df['交易月份']=(df['交易民國年']-113)*12+(df['交易月']-7)
 df['ln_坪數']=np.log(df['淨坪數'])
 df['屋齡平方除100']=df['屋齡']**2/100
 df[['路名','生活圈']]=pd.DataFrame(df['土地位置建物門牌'].apply(road_zone).tolist(),index=df.index)
 df['ln_單價']=np.log(df['單價萬元坪'])
 return df,pd.DataFrame(history),{'winsor_cut_lower':float(lo),'winsor_cut_upper':float(hi)}

def X_basic(df):
 base=df[FEATURE_BASE].copy().astype(float)
 base['型態_華廈']=(df['建物三類']=='華廈').astype(float)
 base['型態_住宅大樓']=(df['建物三類']=='住宅大樓').astype(float)
 for name in ZONE_ORDER:
  if name not in ('天母/芝山','未分類路段'):base[f'生活圈_{name}']=(df['生活圈']==name).astype(float)
 assert base.columns.tolist()==FEATURES
 return base.reset_index(drop=True)

def encode_road(train,test,k=5):
 """Encode roads with train-only fitted means, shrink to train-only zone priors."""
 tz=train.groupby('生活圈')['ln_單價'].agg(['sum','count'])
 rz=train.groupby('路名')['ln_單價'].agg(['sum','count'])
 global_mu=float(train['ln_單價'].mean())
 def encode(fr):
  vals=[]
  for _,r in fr.iterrows():
   zone=r['生活圈'];road=r['路名'];prior=float(tz.loc[zone,'sum']/tz.loc[zone,'count']) if zone in tz.index else global_mu
   if road and road in rz.index:
    sz,n=rz.loc[road,['sum','count']]
    vals.append((float(sz)+k*prior)/(float(n)+k))
   else:vals.append(prior)
  return np.asarray(vals)
 return encode(test)

def make_training_te(df,seed=42):
 if len(df)<10:raise ValueError('Too few training records for inner-fold target encoding')
 res=np.empty(len(df))
 inner=KFold(n_splits=min(5,len(df)),shuffle=True,random_state=seed)
 for ti,vi in inner.split(df):
  res[vi]=encode_road(df.iloc[ti],df.iloc[vi])
 return res

def get_models():
 return {
  'Hedonic OLS':None,
  'Random Forest':RandomForestRegressor(n_estimators=500,min_samples_leaf=3,max_features=.8,n_jobs=-1,random_state=42),
  'Gradient Boosting':GradientBoostingRegressor(n_estimators=300,learning_rate=.04,max_depth=3,min_samples_leaf=5,subsample=.85,random_state=42),
 }

def model_fit_predict(name,train,test,Xtr,Xte,seed=42):
 y=train['ln_單價'].to_numpy()
 if name=='Hedonic OLS':
  reg=sm.OLS(y,sm.add_constant(Xtr,has_constant='add')).fit()
  pred=reg.predict(sm.add_constant(Xte,has_constant='add'))
  return reg,np.asarray(pred),list(Xtr.columns),None
 te_tr=make_training_te(train,seed=seed);te_te=encode_road(train,test)
 mX=Xtr.copy();mX['路段平滑平均ln單價']=te_tr
 vX=Xte.copy();vX['路段平滑平均ln單價']=te_te
 reg=get_models()[name];reg.fit(mX,y)
 pred=reg.predict(vX)
 return reg,pred,list(mX.columns),vX

def metrics(y,pred):
 ape=np.abs((y-pred)/y)
 return {'MAPE_%':float(ape.mean()*100),'MedAPE_%':float(np.median(ape)*100),'R2_價格尺度':float(r2_score(y,pred)),'10%內比例_%':float((ape<=.1).mean()*100),'MAE_萬元坪':float(np.abs(y-pred).mean())}

