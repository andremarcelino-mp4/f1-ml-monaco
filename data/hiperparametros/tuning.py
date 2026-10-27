import pandas as pd,numpy as np,warnings,json;warnings.filterwarnings('ignore')
from sklearn.model_selection import GroupKFold,GroupShuffleSplit,RandomizedSearchCV,cross_val_predict
from sklearn.pipeline import make_pipeline;from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso,Ridge,ElasticNet;from sklearn.ensemble import RandomForestRegressor as RF,GradientBoostingRegressor as GB,ExtraTreesRegressor as ET
from sklearn.neural_network import MLPRegressor;from sklearn.dummy import DummyRegressor
from sklearn.metrics import mean_absolute_error as MAE,mean_squared_error as MSE,r2_score
from sklearn.inspection import permutation_importance
from scipy.stats import loguniform,randint,uniform
d=pd.read_csv('voltas.csv');d['tempo_s']=d.tempo_volta_ms/1000;v=d[d.carro!='mclaren'].reset_index(drop=True)
SET=['asa_dianteira','asa_traseira','dif_com_acel','dif_sem_acel','cambagem_diant','cambagem_tras','convergencia_diant','convergencia_tras','suspensao_diant','suspensao_tras','barra_diant','barra_tras','altura_diant','altura_tras','freio_balanco','pneu_tras_esq_psi','pneu_diant_esq_psi','combustivel_setup']
CTX=['idade_pneu_voltas','volta_saida','entrou_box','flashbacks']
X=v[SET+CTX].copy();X['carro_ferrari']=(v.carro=='ferrari').astype(int);X['composto_duro']=(v.composto_visual=='duro').astype(int)
print('NaN',X.isna().sum().sum());X=X.fillna(X.median())
y=v.tempo_s.values;g=v.session_uid.values
tr,te=next(GroupShuffleSplit(1,test_size=.2,random_state=42).split(X,y,g))
S=lambda m:make_pipeline(StandardScaler(),m)
P='__'
M={ # nome: (default, espaço, chave prefixo)
'Lasso':(S(Lasso(max_iter=50000)),{'lasso__alpha':loguniform(1e-3,1)}),
'Ridge':(S(Ridge()),{'ridge__alpha':loguniform(1e-2,1e4)}),
'ElasticNet':(S(ElasticNet(max_iter=50000)),{'elasticnet__alpha':loguniform(1e-3,2),'elasticnet__l1_ratio':uniform(.05,.95)}),
'Random Forest':(RF(random_state=0,n_jobs=1),{'n_estimators':randint(200,600),'max_depth':[3,5,8,None],'min_samples_leaf':randint(1,20),'max_features':[.3,.5,.8,1.0]}),
'Extra Trees':(ET(random_state=0,n_jobs=1),{'n_estimators':randint(200,600),'max_depth':[3,5,8,None],'min_samples_leaf':randint(1,20),'max_features':[.3,.5,.8,1.0]}),
'Gradient Boosting':(GB(random_state=0),{'n_estimators':randint(50,500),'max_depth':randint(1,5),'learning_rate':loguniform(.005,.2),'subsample':uniform(.5,.5),'min_samples_leaf':randint(1,20)}),
'MLP (rede neural)':(S(MLPRegressor(max_iter=2000,random_state=0)),{'mlpregressor__hidden_layer_sizes':[(8,),(16,),(32,),(16,8),(32,16)],'mlpregressor__alpha':loguniform(1e-3,10),'mlpregressor__learning_rate_init':loguniform(1e-4,1e-2),'mlpregressor__early_stopping':[True,False]}),
}
gk=GroupKFold(5)
def met(yt,p):return MAE(yt,p),MSE(yt,p)**.5,r2_score(yt,p)
rows=[];best={}
b=cross_val_predict(DummyRegressor(),X,y,cv=gk,groups=g);bt=DummyRegressor().fit(X.iloc[tr],y[tr]).predict(X.iloc[te])
rows.append(dict(modelo='Baseline (média)',versao='-',**dict(zip(['MAE_CV','RMSE_CV','R2_CV'],met(y,b))),**dict(zip(['MAE_teste','RMSE_teste','R2_teste'],met(y[te],bt)))))
for n,(m,sp) in M.items():
  p=cross_val_predict(m,X,y,cv=gk,groups=g);pt=m.fit(X.iloc[tr],y[tr]).predict(X.iloc[te])
  rows.append(dict(modelo=n,versao='padrão',**dict(zip(['MAE_CV','RMSE_CV','R2_CV'],met(y,p))),**dict(zip(['MAE_teste','RMSE_teste','R2_teste'],met(y[te],pt)))))
  rs=lambda:RandomizedSearchCV(m,sp,n_iter=30,cv=GroupKFold(4),scoring='neg_mean_absolute_error',random_state=0,n_jobs=8)
  # CV aninhada: busca interna só no treino de cada fold externo
  pn=np.zeros(len(y))
  for a,bb in gk.split(X,y,g):
    s=rs().fit(X.iloc[a],y[a],groups=g[a]);pn[bb]=s.predict(X.iloc[bb])
  s=rs().fit(X.iloc[tr],y[tr],groups=g[tr]);pt=s.predict(X.iloc[te])
  full=rs().fit(X,y,groups=g);best[n]={k:(v_ if isinstance(v_,(int,float,str,type(None),tuple)) else float(v_)) for k,v_ in full.best_params_.items()}
  rows.append(dict(modelo=n,versao='tunado',**dict(zip(['MAE_CV','RMSE_CV','R2_CV'],met(y,pn))),**dict(zip(['MAE_teste','RMSE_teste','R2_teste'],met(y[te],pt))),MAE_busca_otimista=-full.best_score_))
  best[n]['_est']=full.best_estimator_;print(rows[-2]['modelo'],rows[-2]['MAE_CV'],rows[-1]['MAE_CV'],flush=True)
R=pd.DataFrame(rows);R.to_csv('hiperparametros/comparacao_padrao_vs_tunado.csv',index=False)
cmp=R[R.versao!='-'].pivot(index='modelo',columns='versao',values=['MAE_CV','MAE_teste']);cmp.columns=[f'{a}_{b}' for a,b in cmp.columns]
cmp['ganho_CV_s']=cmp.MAE_CV_padrão-cmp.MAE_CV_tunado;cmp['ganho_CV_%']=100*cmp.ganho_CV_s/cmp.MAE_CV_padrão
cmp.sort_values('MAE_CV_tunado').to_csv('hiperparametros/resumo_mae.csv');print(cmp.sort_values('MAE_CV_tunado').round(3).to_string())
imp={}
for n in M:
  e=best[n].pop('_est');imp[n]=pd.Series(permutation_importance(e,X,y,n_repeats=20,random_state=0,scoring='neg_mean_absolute_error').importances_mean,index=X.columns)
I=pd.DataFrame(imp);I.to_csv('hiperparametros/importancia_permutacao_tunados.csv')
print(I.rank(ascending=False).head(30).to_string());print(I.apply(lambda c:list(c.sort_values(ascending=False).index[:3])).to_string())
json.dump(best,open('hiperparametros/melhores_parametros.json','w'),indent=1,ensure_ascii=False,default=str);print(json.dumps(best,default=str,indent=0))
