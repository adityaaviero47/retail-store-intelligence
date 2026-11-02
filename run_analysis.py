"""Retail store intelligence on Rossmann data: forecasting, footfall/conversion diagnostics,
promo and competition drivers, and anomaly detection with revenue-at-risk.
Reproduces every number in README.md:  python run_analysis.py"""
import json, numpy as np, pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

R = {}
t = pd.read_csv("data/train.csv.gz", low_memory=False, parse_dates=["Date"])
s = pd.read_csv("data/store.csv")
t["StateHoliday"] = t["StateHoliday"].astype(str)
d = t[(t.Open == 1) & (t.Sales > 0)].merge(s, on="Store", how="left")
R["data"] = dict(stores=int(d.Store.nunique()), store_days=int(len(d)),
                 start=str(d.Date.min().date()), end=str(d.Date.max().date()))

# ---------------- 1. Diagnostics: footfall x ticket, promo, competition ----------------
d["Ticket"] = d.Sales / d.Customers
promo = d.groupby("Promo").agg(sales=("Sales", "mean"), cust=("Customers", "mean"), ticket=("Ticket", "mean"))
R["promo"] = dict(sales_uplift=promo.sales[1] / promo.sales[0] - 1,
                  footfall_uplift=promo.cust[1] / promo.cust[0] - 1,
                  ticket_uplift=promo.ticket[1] / promo.ticket[0] - 1)
# within-store promo uplift (controls for store mix and weekday)
w = d[d.DayOfWeek <= 5].groupby(["Store", "DayOfWeek", "Promo"]).Sales.mean().unstack().dropna()
R["promo"]["within_store_sales_uplift_median"] = float((w[1] / w[0] - 1).median())

# share of store-to-store sales variation explained by footfall vs ticket (log decomposition)
st = d.groupby("Store").agg(sales=("Sales", "mean"), cust=("Customers", "mean"))
st["ticket"] = st.sales / st.cust
ls, lc, lt = np.log(st.sales), np.log(st.cust), np.log(st.ticket)
R["drivers"] = dict(footfall_share_of_variance=float(np.cov(ls, lc)[0, 1] / ls.var()),
                    ticket_share_of_variance=float(np.cov(ls, lt)[0, 1] / ls.var()))

# competitor openings inside the data window: 8 weeks before vs after, same store
s2 = s.dropna(subset=["CompetitionOpenSinceYear", "CompetitionOpenSinceMonth"]).copy()
s2["comp_open"] = pd.to_datetime(dict(year=s2.CompetitionOpenSinceYear.astype(int), month=s2.CompetitionOpenSinceMonth.astype(int), day=1))
s2 = s2[(s2.comp_open > pd.Timestamp("2013-03-01")) & (s2.comp_open < pd.Timestamp("2015-06-01"))]
eff = []
for _, row in s2.iterrows():
    x = d[d.Store == row.Store]
    b = x[(x.Date >= row.comp_open - pd.Timedelta(weeks=8)) & (x.Date < row.comp_open)]
    a = x[(x.Date >= row.comp_open) & (x.Date < row.comp_open + pd.Timedelta(weeks=8))]
    if len(b) > 20 and len(a) > 20:
        eff.append(dict(sales=a.Sales.mean() / b.Sales.mean() - 1, cust=a.Customers.mean() / b.Customers.mean() - 1))
eff = pd.DataFrame(eff)
R["competition"] = dict(stores=int(len(eff)), median_sales_change=float(eff.sales.median()),
                        median_footfall_change=float(eff.cust.median()))

# ---------------- 2. Forecasting: 6-week holdout ----------------
cut = d.Date.max() - pd.Timedelta(weeks=6)
tr, te = d[d.Date <= cut].copy(), d[d.Date > cut].copy()
R["forecast"] = dict(train_end=str(cut.date()), test_days=int(te.Date.nunique()), test_rows=int(len(te)))

def rmspe(y, p): return float(np.sqrt(np.mean(((y - p) / y) ** 2)))
def wape(y, p): return float(np.abs(y - p).sum() / y.sum())

# baseline A: same store, same weekday, same promo flag, last 12 weeks average
recent = tr[tr.Date > cut - pd.Timedelta(weeks=12)]
base = recent.groupby(["Store", "DayOfWeek", "Promo"]).Sales.mean().rename("base")
te = te.join(base, on=["Store", "DayOfWeek", "Promo"])
te["base"] = te["base"].fillna(te.groupby("Store").Sales.transform("mean"))

def feats(x, ref):
    f = pd.DataFrame(index=x.index)
    f["Store"] = x.Store; f["DayOfWeek"] = x.DayOfWeek; f["Promo"] = x.Promo; f["SchoolHoliday"] = x.SchoolHoliday
    f["StateHoliday"] = x.StateHoliday.map({"0": 0, "a": 1, "b": 2, "c": 3}).fillna(0)
    f["Day"] = x.Date.dt.day; f["Month"] = x.Date.dt.month; f["Year"] = x.Date.dt.year
    f["WeekOfYear"] = x.Date.dt.isocalendar().week.astype(int)
    f["StoreType"] = x.StoreType.map({"a": 0, "b": 1, "c": 2, "d": 3}); f["Assortment"] = x.Assortment.map({"a": 0, "b": 1, "c": 2})
    f["CompetitionDistance"] = x.CompetitionDistance.fillna(1e5)
    co = (x.Date.dt.year - x.CompetitionOpenSinceYear) * 12 + (x.Date.dt.month - x.CompetitionOpenSinceMonth)
    f["CompetitionMonths"] = co.fillna(-1).clip(-1, 240)
    f["Promo2"] = x.Promo2
    for k, g in [("st_mean", ["Store"]), ("st_dow", ["Store", "DayOfWeek"]), ("st_promo", ["Store", "Promo"])]:
        m = ref.groupby(g).Sales.mean().rename(k)
        f[k] = x[g].join(m, on=g)[k].values
    f["st_cust"] = x[["Store"]].join(ref.groupby("Store").Customers.mean().rename("c"), on="Store")["c"].values
    return f

Xtr, Xte = feats(tr, tr), feats(te, tr)
m = HistGradientBoostingRegressor(max_iter=600, learning_rate=0.08, max_leaf_nodes=63, min_samples_leaf=40,
                                  random_state=7)
m.fit(Xtr, np.log1p(tr.Sales))
te["pred"] = np.expm1(m.predict(Xte))
R["forecast"]["baseline"] = dict(rmspe=rmspe(te.Sales, te.base), wape=wape(te.Sales, te.base))
R["forecast"]["model"] = dict(rmspe=rmspe(te.Sales, te.pred), wape=wape(te.Sales, te.pred))
R["forecast"]["wape_reduction"] = 1 - R["forecast"]["model"]["wape"] / R["forecast"]["baseline"]["wape"]

# ---------------- 3. Anomaly detection + revenue at risk, evaluated with injected faults ----------------
# residual scale per store measured OUT OF SAMPLE: a second model trained up to 6 weeks before the
# cut forecasts the 6 weeks before the cut; robust (MAD-based) spread of those log residuals per store
cut2 = cut - pd.Timedelta(weeks=6)
tr2, cal = tr[tr.Date <= cut2], tr[tr.Date > cut2].copy()
m2 = HistGradientBoostingRegressor(max_iter=600, learning_rate=0.08, max_leaf_nodes=63, min_samples_leaf=40, random_state=7)
m2.fit(feats(tr2, tr2), np.log1p(tr2.Sales))
cal["pred"] = np.expm1(m2.predict(feats(cal, tr2)))
cal["lr"] = np.log(cal.Sales / cal.pred)
g = cal.groupby("Store").lr
sd = (1.4826 * g.apply(lambda v: np.median(np.abs(v - np.median(v))))).rename("sd")
te = te.join(sd, on="Store")
te["lr"] = np.log(te.Sales / te.pred)
rng = np.random.default_rng(42)
inj = rng.choice(te.index, size=500, replace=False)
te["injected"] = False; te.loc[inj, "injected"] = True
te["Sales_obs"] = te.Sales.astype(float)
te.loc[inj, "Sales_obs"] = te.loc[inj, "Sales"] * 0.6          # simulated 40% shortfall (e.g. stock-out, staff gap)
te["z"] = np.log(te.Sales_obs / te.pred) / te.sd
res = {}
for k in [2.5, 3.0, 3.5]:
    flag = te.z < -k
    tp = int((flag & te.injected).sum()); fp = int((flag & ~te.injected).sum())
    res[str(k)] = dict(recall=tp / 500, precision=tp / max(1, tp + fp), flags=int(flag.sum()),
                       false_alarm_rate=fp / int((~te.injected).sum()),
                       injected_loss_caught=float((te.Sales - te.Sales_obs)[flag & te.injected].sum()
                                                  / (te.Sales - te.Sales_obs)[te.injected].sum()))
R["anomaly"] = dict(injected=500, injected_drop=0.4, thresholds=res)
k = 3.0; flag = te.z < -k
R["anomaly"]["revenue_at_risk_flagged"] = float((te.pred - te.Sales_obs)[flag].clip(lower=0).sum())
R["anomaly"]["revenue_at_risk_true_injected"] = float((te.Sales - te.Sales_obs)[te.injected].sum())

json.dump(R, open("results/results.json", "w"), indent=2, default=float)
print(json.dumps(R, indent=2, default=float))
