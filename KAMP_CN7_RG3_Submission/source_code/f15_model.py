"""Supplied F1.5 notebook model scoring functions, unmodified.
"""
from __future__ import annotations
from pathlib import Path
import math
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, average_precision_score, f1_score, fbeta_score, precision_score, recall_score, confusion_matrix

DOMAIN = 'competition_supplied_scale'


def load_decision_bundle(path):
    bundle = joblib.load(path)
    for key in ('product_gate','cn7_quality','rg3_quality','decision_policy'):
        if key not in bundle:
            raise KeyError(f'Model bundle missing required key: {key}')
    return bundle


def _deterministic_rank(df, score_col, id_col='_row_id'):
    tmp = df[[score_col,id_col]].copy()
    tmp['_idx'] = df.index
    tmp = tmp.sort_values([score_col,id_col],ascending=[False,True],kind='stable')
    tmp['_rank'] = np.arange(1,len(tmp)+1)
    return pd.Series(tmp['_rank'].to_numpy(),index=tmp['_idx'],dtype=int).reindex(df.index)


def _cn7_queue(scores, frozen_rhos=(.2,.3,.2), capacities=(.05,.10,.20)):
    d=scores.copy()
    d['C_rank']=_deterministic_rank(d,'XGB_SVF_SSL')
    d['A1_rank']=_deterministic_rank(d,'LOG_SV_SSL')
    d['A2_rank']=_deterministic_rank(d,'TEAM_LX')
    n=len(d)
    d['C_frac']=d['C_rank']/n; d['A1_frac']=d['A1_rank']/n; d['A2_frac']=d['A2_rank']/n
    selected=[]; selected_set=set(); source={}; support_of={}
    for stage,(cap,rho) in enumerate(zip(capacities,frozen_rhos),start=1):
        target_k=max(1,int(np.floor(n*cap)))
        new_slots=target_k-len(selected)
        if new_slots<=0: continue
        rem=d.loc[~d.index.isin(selected_set)].copy()
        rescue=rem[(rem.C_frac>cap)&((rem.A1_frac<=cap)|(rem.A2_frac<=cap))].copy()
        rescue['support']=(rescue.A1_frac<=cap).astype(int)+(rescue.A2_frac<=cap).astype(int)
        rescue['best_aux']=rescue[['A1_frac','A2_frac']].min(axis=1)
        rescue['gap']=rescue.C_frac-rescue.best_aux
        rescue=rescue.sort_values(['support','gap','best_aux','C_frac','_row_id'],ascending=[False,False,True,True,True],kind='stable')
        n_rescue=min(int(np.floor(new_slots*rho)),len(rescue))
        rescue_idx=list(rescue.head(n_rescue).index)
        core_pool=d.loc[~d.index.isin(selected_set.union(rescue_idx))].sort_values(['C_frac','_row_id'],ascending=[True,True],kind='stable')
        core_idx=list(core_pool.head(new_slots-n_rescue).index)
        for idx in core_idx:
            source[idx]='CORE_XGB'; support_of[idx]=0
        for idx,row in rescue.loc[rescue.index.isin(rescue_idx)].iterrows():
            source[idx]='RESCUE_LOG_TEAM'; support_of[idx]=int(row.support)
        selected.extend(core_idx+rescue_idx); selected_set.update(core_idx+rescue_idx)
    rest=d.loc[~d.index.isin(selected_set)].sort_values(['C_frac','_row_id'],ascending=[True,True],kind='stable')
    for idx in rest.index:
        source[idx]='CORE_XGB_AFTER20'; support_of[idx]=0
    selected.extend(list(rest.index))
    rank_map=pd.Series(np.arange(1,n+1),index=selected,dtype=int)
    d['PriorityRank']=rank_map.reindex(d.index)
    d['QueueSource']=pd.Series(source).reindex(d.index)
    d['RescueSupport']=pd.Series(support_of).reindex(d.index).astype(int)
    d['PriorityScore']=1-(d.PriorityRank-1)/max(n-1,1)
    for cap in capacities:
        pct=int(round(cap*100)); k=max(1,int(np.floor(n*cap)))
        d[f'Inspect_Top{pct}']=(d.PriorityRank<=k)
    d['InspectionTier']=np.select(
        [d.Inspect_Top5,d.Inspect_Top10,d.Inspect_Top20],
        ['A_TOP5','B_5_TO_10','C_10_TO_20'],
        default='D_AFTER_20')
    return d


def score_cn7(rows,bundle):
    m=bundle['cn7_quality']; base=list(m['base_features'])
    missing=sorted(set(base)-set(rows.columns))
    if missing: raise KeyError(f'CN7 missing features: {missing}')
    d=rows.reset_index(drop=True).copy(); d['_row_id']=np.arange(len(d))
    inter=m.get('interaction_definition',{'S':'Max_Switch_Over_Pressure','V':'Max_Injection_Speed','F':'Filling_Time'})
    S,V,F=inter['S'],inter['V'],inter['F']
    d['INT_SV']=d[S]*d[V]; d['INT_SF']=d[S]*d[F]; d['INT_FV']=d[F]*d[V]; d['INT_SVF']=d[S]*d[V]*d[F]
    logb=m['log_aux_model']; X=d[logb['features']].astype(float)
    log_input=X if hasattr(logb['scaler'],'feature_names_in_') else X.to_numpy(float)
    log_score=logb['model'].predict_proba(logb['scaler'].transform(log_input))[:,1]
    teamb=m['team_aux_model']; Xt=d[teamb['features']].to_numpy(float)
    team_score=.5*teamb['log'].predict_proba(Xt)[:,1]+.5*teamb['xgb'].predict_proba(Xt)[:,1]
    seed=[]
    for xb in m['xgb_core_models']:
        seed.append(xb['model'].predict_proba(d[xb['features']].to_numpy(float))[:,1])
    xgb_score=np.mean(np.column_stack(seed),axis=1)
    q=_cn7_queue(pd.DataFrame({'_row_id':np.arange(len(d)),'XGB_SVF_SSL':xgb_score,'LOG_SV_SSL':log_score,'TEAM_LX':team_score}),tuple(m['frozen_rhos']),tuple(m['capacities']))
    # queue is input-indexed; preserve row order
    q=q.sort_values('_row_id').reset_index(drop=True)
    out=pd.DataFrame({
        'Native_Score_Name':'CN7_XGB_Core_Score',
        'Native_Risk_Score':q['XGB_SVF_SSL'].to_numpy(float),
        'RiskScore_0_1':q['PriorityScore'].to_numpy(float),
        'PriorityScore_0_1':q['PriorityScore'].to_numpy(float),
        'PriorityRank':q['PriorityRank'].to_numpy(int),
        'InspectionTier':q['InspectionTier'].to_numpy(),
        'Inspect_Top5':q['Inspect_Top5'].to_numpy(bool),
        'Inspect_Top10':q['Inspect_Top10'].to_numpy(bool),
        'Inspect_Top20':q['Inspect_Top20'].to_numpy(bool),
        'CN7_XGB_Core_Score':q['XGB_SVF_SSL'].to_numpy(float),
        'CN7_LOG_Aux_Score':q['LOG_SV_SSL'].to_numpy(float),
        'CN7_TEAM_Aux_Score':q['TEAM_LX'].to_numpy(float),
        'CN7_QueueSource':q['QueueSource'].to_numpy(),
        'CN7_RescueSupport':q['RescueSupport'].to_numpy(int),
        'RG3_Core_Risk_Score':np.nan,
        'RG3_Pair_Adjustment':np.nan,
        'RG3_Pair_Eligible':pd.Series(pd.NA,index=range(len(q)),dtype='Int64'),
        'RG3_Pair_Position':np.nan,
    })
    return out


def _rg3_pair_position(rows,features,id_col):
    n=len(rows); sig=np.zeros(n,float); eligible=np.zeros(n,int); pos=np.full(n,np.nan)
    ids=pd.to_numeric(rows[id_col],errors='raise').to_numpy(float)
    group=pd.factorize(pd.MultiIndex.from_frame(rows[features]),sort=False)[0]
    for _, offsets in pd.Series(np.arange(n)).groupby(group,sort=False):
        idx=offsets.to_numpy()
        if len(idx)!=2: continue
        first,second=idx[np.argsort(ids[idx])]
        if not np.isclose(ids[second]-ids[first],1.0): continue
        eligible[[first,second]]=1; pos[first]=0.; pos[second]=1.; sig[first]=1.; sig[second]=-1.
    return eligible,pos,sig


def score_rg3(rows,bundle):
    m=bundle['rg3_quality']; features=list(m['feature_columns']); idcol=m['id_column']
    missing=sorted(set(features+[idcol])-set(rows.columns))
    if missing: raise KeyError(f'RG3 missing columns: {missing}')
    d=rows.reset_index(drop=True).copy(); n=len(d)
    eligible,pos,sig=_rg3_pair_position(d,features,idcol)
    rg3_input=d[features].astype(float)
    scaler_input=rg3_input if hasattr(m['core_scaler'],'feature_names_in_') else rg3_input.to_numpy(float)
    z=m['core_scaler'].transform(scaler_input)*np.sqrt(m['feature_weights'])
    dist=m['core_nn'].kneighbors(z,return_distance=True)[0].ravel(); core=1/(1+dist)
    adj=float(m['pair_lambda'])*sig*int(m['pair_direction']); raw=core+adj
    order=np.argsort(-raw,kind='stable'); rank=np.empty(n,int); rank[order]=np.arange(1,n+1)
    ps=1-(rank-1)/max(n-1,1)
    k5=max(1,int(np.floor(n*.05))); k10=max(1,int(np.floor(n*.10))); k20=max(1,int(np.floor(n*.20)))
    top5=rank<=k5;top10=rank<=k10;top20=rank<=k20
    tier=np.select([top5,top10,top20],['A_TOP5','B_5_TO_10','C_10_TO_20'],default='D_AFTER_20')
    out=pd.DataFrame({
        'Native_Score_Name':'RG3_WeightedSoftPair_RiskScore',
        'Native_Risk_Score':raw,
        'RiskScore_0_1':ps,
        'PriorityScore_0_1':ps,
        'PriorityRank':rank,
        'InspectionTier':tier,
        'Inspect_Top5':top5,'Inspect_Top10':top10,'Inspect_Top20':top20,
        'CN7_XGB_Core_Score':np.nan,'CN7_LOG_Aux_Score':np.nan,'CN7_TEAM_Aux_Score':np.nan,
        'CN7_QueueSource':pd.Series(pd.NA,index=range(n),dtype='string'),'CN7_RescueSupport':pd.Series(pd.NA,index=range(n),dtype='Int64'),
        'RG3_Core_Risk_Score':core,'RG3_Pair_Adjustment':adj,'RG3_Pair_Eligible':eligible,'RG3_Pair_Position':pos,
    })
    return out


def _auto_route(frame,bundle):
    g=bundle['product_gate']; features=list(g['features'])
    missing=sorted(set(features)-set(frame.columns))
    if missing: raise KeyError(f'Product gate missing features: {missing}')
    x=pd.DataFrame(frame[features].to_numpy(float),columns=features)
    prob=g['model'].predict_proba(x)[:,1]
    low=pd.Series(g['training_min'])[features].to_numpy(float); high=pd.Series(g['training_max'])[features].to_numpy(float)
    tol=1e-12*np.maximum(1.,np.maximum(np.abs(low),np.abs(high)))
    outside=((x.to_numpy()<(low-tol))|(x.to_numpy()>(high+tol))).any(axis=1)
    review=(np.maximum(prob,1-prob)<.8)|outside
    product=np.where(prob>=.5,'RG3','CN7')
    return product,review,prob,outside


def score_dataframe(frame,bundle,product_col='Product_Type'):
    if len(frame)==0: raise ValueError('Input is empty')
    src=frame.reset_index(drop=True).copy()
    if product_col in src.columns:
        product=src[product_col].astype(str).str.upper().to_numpy()
        bad=~np.isin(product,['CN7','RG3'])
        if bad.any(): raise ValueError(f'{product_col} must contain only CN7/RG3. Bad rows: {np.flatnonzero(bad)[:10].tolist()}')
        review=np.zeros(len(src),bool); prob=np.full(len(src),np.nan); outside=np.zeros(len(src),bool); routing=np.repeat('EXPLICIT_PRODUCT',len(src))
    else:
        product,review,prob,outside=_auto_route(src,bundle); routing=np.repeat('AUTO_GATE',len(src))
    out=pd.DataFrame({'Input_Row':np.arange(len(src))})
    if 'Unnamed: 0' in src.columns: out['Record_ID']=src['Unnamed: 0'].to_numpy()
    else: out['Record_ID']=np.arange(len(src))
    out['Product_Type']=product; out['Routing_Source']=routing; out['Product_Review']=review; out['P_RG3_uncalibrated']=prob; out['Outside_Gate_Training_Range']=outside
    # initialize common/detail columns
    cols=['Model_Applied','Native_Score_Name','Native_Risk_Score','RiskScore_0_1','PriorityScore_0_1','PriorityRank','PriorityTopPercent','DecisionThreshold','Suspected_Defect','Predicted_PassFail','Predicted_Label','InspectionTier','Inspect_Top5','Inspect_Top10','Inspect_Top20','CN7_XGB_Core_Score','CN7_LOG_Aux_Score','CN7_TEAM_Aux_Score','CN7_QueueSource','CN7_RescueSupport','RG3_Core_Risk_Score','RG3_Pair_Adjustment','RG3_Pair_Eligible','RG3_Pair_Position']
    for c in cols: out[c]=pd.NA
    for prod, scorer in [('CN7',score_cn7),('RG3',score_rg3)]:
        idx=np.flatnonzero(product==prod)
        if not len(idx): continue
        s=scorer(src.iloc[idx].reset_index(drop=True),bundle)
        threshold=float(bundle['decision_policy']['thresholds'][prod]['threshold'])
        suspect=s['PriorityScore_0_1'].to_numpy(float)>=threshold
        top_pct=100*s['PriorityRank'].to_numpy(int)/len(s)
        s['PriorityTopPercent']=top_pct; s['DecisionThreshold']=threshold; s['Suspected_Defect']=suspect; s['Predicted_PassFail']=suspect.astype(int); s['Predicted_Label']=np.where(suspect,'FAIL_SUSPECT','PASS_PREDICTED')
        s['Model_Applied']='CN7_XGB_Core_FrozenQuota_LOG_TEAM' if prod=='CN7' else 'RG3_Weighted1NN_SoftPair'
        for c in s.columns:
            out.loc[idx,c]=s[c].to_numpy()
    # normalized dtypes where practical
    for c in ['PriorityRank','Predicted_PassFail','CN7_RescueSupport','RG3_Pair_Eligible']:
        out[c]=pd.to_numeric(out[c],errors='coerce').astype('Int64')
    for c in ['Inspect_Top5','Inspect_Top10','Inspect_Top20','Suspected_Defect']:
        out[c]=out[c].astype('boolean')
    if 'PassOrFail' in src.columns:
        actual=pd.to_numeric(src['PassOrFail'],errors='raise').astype(int)
        out['Actual_PassOrFail']=actual
        out['Correct_Prediction']=(out['Predicted_PassFail'].astype('Int64')==actual).astype('boolean')
    return out


def evaluate_labeled(pred):
    if 'Actual_PassOrFail' not in pred.columns:
        return pd.DataFrame([{'Note':'No PassOrFail column; classification metrics unavailable.'}])
    rows=[]
    for prod,g in pred.groupby('Product_Type',sort=False):
        y=g['Actual_PassOrFail'].astype(int).to_numpy(); yh=g['Predicted_PassFail'].astype(int).to_numpy(); s=pd.to_numeric(g['PriorityScore_0_1']).to_numpy(float)
        tn,fp,fn,tp=confusion_matrix(y,yh,labels=[0,1]).ravel()
        rec={'Product':prod,'N':len(g),'Actual_Defects':int(y.sum()),'Predicted_Defects':int(yh.sum()),'Threshold':float(g.DecisionThreshold.iloc[0]),'Accuracy':accuracy_score(y,yh),'Precision':precision_score(y,yh,zero_division=0),'Recall':recall_score(y,yh,zero_division=0),'F1':f1_score(y,yh,zero_division=0),'F1_5':fbeta_score(y,yh,beta=1.5,zero_division=0),'F2':fbeta_score(y,yh,beta=2,zero_division=0),'Average_Precision':average_precision_score(y,s) if y.sum() else np.nan,'TN':int(tn),'FP':int(fp),'FN':int(fn),'TP':int(tp)}
        for pct in [5,10,20]:
            flag=g[f'Inspect_Top{pct}'].astype(bool).to_numpy(); rec[f'Top{pct}_Defect_Capture']=float(y[flag].sum()/y.sum()) if y.sum() else np.nan
        rows.append(rec)
    return pd.DataFrame(rows)
