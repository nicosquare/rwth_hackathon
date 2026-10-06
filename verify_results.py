"""Verify exported results and procurement arithmetic after run_levels.py completes."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from run_levels import cost,metric,Accumulator
r=Path(__file__).parent/'results'; s=json.loads((r/'summary.json').read_text()); audit=pd.read_csv(r/'data_audit.csv'); port=pd.read_csv(r/'portfolio_predictions.csv'); business=pd.read_csv(r/'level2_procurement.csv'); metrics=pd.read_csv(r/'forecast_metrics.csv')
assert len(audit)==s['files']==410
assert audit.Household_ID.nunique()==410
assert audit.test_rows.sum()==s['test_rows']==int(port.households.sum())
assert (metrics[metrics.scope=='household'].n==s['test_rows']).all()
assert len(port)==port.Timestamp.nunique()
for row in business.itertuples():
    actual=port.actual.to_numpy(); forecast=port[row.model].to_numpy()
    assert np.isclose(cost(actual,forecast).sum(),row.total_cost_EUR)
    assert np.isclose(row.total_cost_EUR-row.perfect_cost_EUR,row.extra_cost_EUR)
    assert (forecast>=0).all()
small=pd.read_csv(r/'household_predictions.csv.gz',nrows=10000)
assert (small.p10<=small.p50).all() and (small.p50<=small.p90).all()
assert (small[['p10','p50','p90','level0','level1']]>=0).all().all()
assert np.allclose(cost(np.array([10.,10.,10.]),np.array([10.,8.,12.])),[1.,1.16,1.08])
a=Accumulator(); a.add([1.,3.],[2.,2.]); a.add([2.],[2.]); stream=a.result('test','household'); direct=metric([1.,3.,2.],[2.,2.,2.],'test','household')
for k in ['MAE_kWh','RMSE_kWh','R2','WAPE']: assert np.isclose(stream[k],direct[k])
print('PASS: household counts, portfolio aggregation, metrics, procurement costs, quantile ordering and streaming arithmetic.')
