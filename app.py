"""FastAPI interactive Shilin residential valuation site.
User only needs a browser. The server runs the pretrained OLS/RF/GB models.
"""
from __future__ import annotations
import io
import os
import json
import hmac
import tempfile
import threading
import zipfile
from datetime import datetime
from zoneinfo import ZoneInfo
from pathlib import Path
from typing import Optional
from fastapi import FastAPI, Header, HTTPException, UploadFile, File
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field, model_validator
import joblib
import numpy as np
import pandas as pd
from study_core import X_basic, road_zone, ZONES
from training import train_and_export, load_sources, clean_data

ROOT=Path(__file__).resolve().parent
MODEL=ROOT/'model'
APP=FastAPI(title='士林 AI 住宅估價網站',docs_url=None,redoc_url=None)
app=APP
MODELS=('Hedonic OLS','Random Forest','Gradient Boosting')
ADMIN_TOKEN=os.getenv('ADMIN_TOKEN','')
_STATE={'running':False,'message':'尚未啟動重新訓練','last_result':None}
_LOCK=threading.Lock()
UPLOADS=Path(tempfile.gettempdir())/'shilin_valuation_uploads';UPLOADS.mkdir(exist_ok=True)

class Property(BaseModel):
    address:str=Field(min_length=3,max_length=120)
    district:str='士林區'
    building_type:str='華廈'
    area_ping:float=Field(ge=8,le=200)
    age_years:float=Field(ge=0,le=100)
    floor:int=Field(ge=1,le=100)
    total_floors:int=Field(ge=1,le=100)
    rooms:int=Field(ge=0,le=6)
    managed:bool=False
    elevator:bool=False  # UI only. This parameter was NOT part of trained research features.
    has_parking:bool=False
    parking_price_10k:Optional[float]=Field(default=None,ge=0,le=5000)
    extra_construction:bool=False
    price_month:str='2026-10'

    @model_validator(mode='after')
    def validate(self):
        if self.floor>self.total_floors:raise ValueError('所在樓層不可高於總樓層。')
        if self.building_type not in ('公寓','華廈','住宅大樓'):
            raise ValueError('本模型僅支援公寓、華廈及住宅大樓。')
        if self.district!='士林區':raise ValueError('本模型僅使用士林區資料訓練，目前不支援其他行政區。')
        try:y,m=[int(i) for i in self.price_month.split('-')]
        except Exception:raise ValueError('估價基準月份應為 YYYY-MM。')
        if (y,m)<(2024,7) or m<1 or m>12:
            raise ValueError('估價月份不得早於 2024/07。')
        return self

def read_bundle():
    # small bundles; prefer disk refresh after admin retraining
    try:
        meta=json.loads((MODEL/'metadata.json').read_text(encoding='utf8'))
        ols=json.loads((MODEL/'hedonic_ols.json').read_text(encoding='utf8'))
        priors=json.loads((MODEL/'road_priors.json').read_text(encoding='utf8'))
        rf=joblib.load(MODEL/'random_forest.joblib')
        gb=joblib.load(MODEL/'gradient_boosting.joblib')
        return meta,ols,priors,rf,gb
    except FileNotFoundError:
        raise HTTPException(status_code=503,detail='目前沒有可用的預訓練模型。')

def road_prior(priors,road,zone):
    mu=priors['global_ln_mean']
    z=priors['zone'].get(zone)
    if z and z['count']:mu=z['sum']/z['count']
    r=priors['road'].get(road)
    if r and r['count']:mu=(r['sum']+priors['k']*mu)/(r['count']+priors['k'])
    return float(mu)

@app.get('/')
def index():return FileResponse(ROOT/'static'/'index.html')

@app.get('/admin')
def admin_html():return FileResponse(ROOT/'static'/'admin.html')

@app.get('/api/health')
def health():return {'ok':True,'model_ready':(MODEL/'metadata.json').is_file()}

@app.get('/api/status')
def status():
    if not (MODEL/'metadata.json').is_file():return {'ready':False}
    meta=json.loads((MODEL/'metadata.json').read_text(encoding='utf8'))
    current_month=datetime.now(ZoneInfo('Asia/Taipei')).strftime('%Y-%m')
    max_projection=meta.get('projection',{}).get('max_horizon_months',3)
    ly,lm=map(int,meta['latest_month'].split('-'));cy,cm=map(int,current_month.split('-'))
    months_gap=(cy-ly)*12+cm-lm
    enabled_current=months_gap<=max_projection and months_gap>=0
    return {'ready':True,'title':meta['title'],'market':meta['market'],'records':meta['records'],
      'dates':meta['sample_dates'],'cv':meta['metrics'],'zone_counts':meta['zone_counts'],
      'road_counts':meta['road_counts'],'parking':meta['parking'],
      'contains_elevator_feature':meta['contains_elevator_feature'],
      'elevator_type_crosstab':meta.get('elevator_type_crosstab',{}),
      'limitations':meta['limitations'],'supported_zones':ZONES,
      'trade_year_counts':meta.get('trade_year_counts',{}),
      'latest_month':meta.get('latest_month','2025-12'),'earliest_month':meta.get('earliest_month','2024-07'),
      'current_month':current_month,'current_projection_available':enabled_current,
      'max_selectable_month':current_month if enabled_current else meta['latest_month'],
      'projection':meta.get('projection',{}),
      'supported_months':meta.get('supported_months','')}

@app.post('/api/estimate')
def estimate(item:Property):
    meta,ols,priors,rf,gb=read_bundle()
    if '區' in item.address and '士林區' not in item.address:
        raise HTTPException(status_code=422,detail='地址不是士林區。此模型目前只支援臺北市士林區住宅。')
    road,zone=road_zone(item.address)
    if zone=='未分類路段':
        raise HTTPException(status_code=422,detail='此道路不在本研究指定的士林區四個生活圈路段內，請改輸入已收錄的路名（例如：中山北路六段）。')
    year,month=map(int,item.price_month.split('-'))
    month_index=(year-2024)*12+(month-7)
    latest=meta.get('latest_month','2025-12')
    ly,lm=map(int,latest.split('-'))
    month_gap=(year-ly)*12+month-lm
    current_month=datetime.now(ZoneInfo('Asia/Taipei')).strftime('%Y-%m')
    max_future=meta.get('projection',{}).get('max_horizon_months',3)
    if month_index<0 or item.price_month>current_month or month_gap>max_future:
        raise HTTPException(422,f'目前最多只提供至 {latest} 的歷史估價，或其後 3 個月內且不晚於當月的短期趨勢推估。')
    observed_based=month_gap>0
    effective_month_index=(ly-2024)*12+(lm-7) if observed_based else month_index
    row={'淨坪數':item.area_ping,'屋齡':item.age_years,'樓層':item.floor,
         '總樓層':item.total_floors,'房數':item.rooms,'一樓':int(item.floor==1),
         '頂樓':int(item.floor==item.total_floors and item.total_floors>1),
         '管委會':int(item.managed),'含車位':int(item.has_parking),
         '車位價未拆分':0,'陽台外推或增建':int(item.extra_construction),
         '交易月份':effective_month_index,'建物三類':item.building_type,
         '生活圈':zone,'路名':road,'ln_坪數':np.log(item.area_ping),
         '屋齡平方除100':item.age_years**2/100}
    X=X_basic(pd.DataFrame([row]))
    lnp=ols['const']+sum(float(X.loc[0,f])*ols['coefficients'][f] for f in ols['features'])
    xm=X.copy();xm['路段平滑平均ln單價']=road_prior(priors,road,zone)
    preds={
      'Hedonic OLS':float(np.exp(lnp)),
      'Random Forest':float(np.exp(rf.predict(xm)[0])),
      'Gradient Boosting':float(np.exp(gb.predict(xm)[0]))}
    growth_adjustment=1.0
    if observed_based:
        trend=meta['projection']['trend_ln_month_applied']
        growth_adjustment=float(np.exp(trend*month_gap))
        preds={k:v*growth_adjustment for k,v in preds.items()}
    park_cost=0.0
    park_source='none'
    warnings=[]
    if observed_based:
        warnings.append(f'此為截至 {latest} 最近成交月份估價，再將 Hedonic OLS 估計月趨勢（每月 {100*meta["projection"]["trend_ln_month_applied"]:+.2f}% 的對數變化）外推 {month_gap} 個月至 {item.price_month}；不是當月真實成交資料或已驗證的精準預測。')
    if item.has_parking:
        if item.parking_price_10k is not None:
            park_cost=item.parking_price_10k;park_source='user_input'
        else:
            med=meta['parking']['median_10k']
            if med is None:
                warnings.append('無可分拆車位成交價格；總價僅包含住宅，尚未加入車位。')
            else:
                park_cost=med;park_source='overall_median'
                warnings.append(f'車位暫按樣本中 {meta["parking"]["n_separate"]} 筆分拆車位交易的中位數 {med:.0f} 萬估算，未依車位種類或坪數調整。')
    count=int(meta['road_counts'].get(road,0))
    if max(preds.values())/min(preds.values())>1.15:
        warnings.append('三個模型的單價估計相差超過 15%，顯示模型不確定性較高；請搭配最新的相近成交案例審慎判斷。')
    if count<10:
        warnings.append(f'指定路段有效住宅樣本只有 {count} 筆；估價大量借用生活圈其他交易，可信度有限。')
    combos=meta.get('elevator_type_crosstab',{})
    elevator_value='有' if item.elevator else '無'
    if combos and combos.get(item.building_type,{}).get(elevator_value,0)==0:
        warnings.append(f'這批樣本內沒有「{item.building_type}、電梯{elevator_value}」的有效交易，此組合超出既有訓練資料，價格未能識別此特殊條件。')
    warnings.append('原始登錄有電梯欄位，但在本樣本中電梯與建物型態完全重合；三模型無法估算獨立電梯溢價，切換電梯不會直接調整價格。')
    if month_gap>=0:
        warnings.append('模型依實際成交資料訓練；2026 年較新月份的資料筆數可能很少，請特別留意樣本代表性。')
    if item.age_years>60:
        warnings.append('屋齡偏高，可能超出樣本主要涵蓋範圍；結果有外推風險。')
    warnings.append(f'估價基準 {item.price_month}；截至 {latest} 的已揭露住宅成交資料，非即時行情或正式鑑價。')
    final={}
    for key,unit in preds.items():
        # CV MAPE is an aggregate diagnostic, NOT a property-level PI
        err=meta['metrics'][key]['mape_pct']/100
        final[key]={'unit_10k_ping':round(unit,2),'house_total_10k':round(unit*item.area_ping,2),
          'parking_total_10k':round(park_cost,2) if item.has_parking else 0,
          'total_10k':round(unit*item.area_ping+park_cost,2),
          'illustrative_low_10k_ping':round(unit*(1-err),2),
          'illustrative_high_10k_ping':round(unit*(1+err),2),
          'cv_mape_pct':meta['metrics'][key]['mape_pct']}
    return {'road':road,'zone':zone,'road_sample_count':count,
      'basis_month':item.price_month,'area_ping':item.area_ping,
      'is_extrapolated':observed_based,'months_extrapolated':max(0,month_gap),
      'trend_adjustment_pct':round((growth_adjustment-1)*100,3),
      'estimates':final,'best_model':min(MODELS,key=lambda n:meta['metrics'][n]['mape_pct']),
      'parking_source':park_source,'warnings':warnings,
      'range_note':'上下幅度僅以各模型整體 CV MAPE 作 ±% 敏感度示例，不是統計預測區間。'}

def authenticate(token):
    if not ADMIN_TOKEN:raise HTTPException(403,'尚未於 Render 環境變數設定 ADMIN_TOKEN，無法使用管理功能。')
    if not token or not hmac.compare_digest(ADMIN_TOKEN,token):
        raise HTTPException(401,'管理密碼錯誤。')

@app.get('/api/admin/status')
def admin_status(x_admin_token:Optional[str]=Header(None)):
    authenticate(x_admin_token)
    return {**_STATE,'uploaded_files':[p.name for p in UPLOADS.glob('*.zip')]}

@app.post('/api/admin/upload')
async def admin_upload(file:UploadFile=File(...),x_admin_token:Optional[str]=Header(None)):
    authenticate(x_admin_token)
    if not (file.filename or '').lower().endswith('.zip'):
        raise HTTPException(422,'只允許官方實價登錄 ZIP 壓縮檔。')
    b=await file.read(16_000_001)
    if len(b)>16_000_000:raise HTTPException(413,'檔案不可大於 16 MB。')
    try:
        with zipfile.ZipFile(io.BytesIO(b)) as z:
            members=z.infolist()
            if len(members)>200 or sum(x.file_size for x in members)>100_000_000:raise ValueError('解壓檔案太大')
            if not any(x.filename.endswith('A_lvr_land_A.csv') for x in members):
                raise ValueError('ZIP 中找不到 A_lvr_land_A.csv')
    except (zipfile.BadZipFile,ValueError) as err:
        raise HTTPException(422,str(err))
    import uuid
    path=UPLOADS/('official_'+uuid.uuid4().hex[:10]+'.zip')
    path.write_bytes(b)
    return {'ok':True,'filename':path.name,'bytes':len(b)}

def background_train():
    try:
        _STATE['message']='正在讀取與清理官方資料…'
        raw,logs=load_sources(UPLOADS)
        available={x.get('quarter') for x in logs if x.get('status')=='included CSV'}
        required={'114S1','114S2','114S3','114S4','115S1','115S2','115S3'}
        missing=sorted(required-available)
        if missing:
            raise ValueError('缺少原始七季資料：'+', '.join(missing)+'。為避免覆蓋完整模型，請先補齊所有季度 ZIP 再重新訓練。')
        clean,history,options=clean_data(raw)
        _STATE['message']=f'資料清理完成（{len(clean)} 筆），正在訓練並執行五折測試…'
        with tempfile.TemporaryDirectory() as tmp:
            meta=train_and_export(clean,Path(tmp))
            # Files are generated before being promoted; model loads may briefly need retry.
            for f in Path(tmp).iterdir():
                os.replace(f,MODEL/f.name)
        _STATE['last_result']={'records':meta['records'],'metrics':meta['metrics'],'filters':history.to_dict('records'),'sources':logs}
        _STATE['message']='模型重新訓練完成；請重新整理估價頁面。'
    except Exception as exc:
        _STATE['message']='訓練失敗：'+str(exc)[:280]
    finally:
        _STATE['running']=False
        _LOCK.release()

@app.post('/api/admin/retrain')
def retrain(x_admin_token:Optional[str]=Header(None)):
    authenticate(x_admin_token)
    if not list(UPLOADS.glob('*.zip')):raise HTTPException(400,'請先上傳官方 ZIP。')
    if not _LOCK.acquire(blocking=False):raise HTTPException(409,'模型正在重新訓練。')
    _STATE['running']=True
    th=threading.Thread(target=background_train,daemon=True)
    th.start()
    return {'ok':True,'message':'已開始背景訓練，請至管理頁查詢狀態。'}
