"""Charts for README from results/results.json."""
import json, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
R = json.load(open("results/results.json"))
fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
f = R["forecast"]
ax[0].bar(["12-week store\nbaseline", "Gradient-boosted\nmodel"], [f["baseline"]["wape"]*100, f["model"]["wape"]*100], color=["#9aa5ad", "#0052CC"])
ax[0].set_ylabel("WAPE on 6-week holdout (%)"); ax[0].set_title("Forecast error")
for i, v in enumerate([f["baseline"]["wape"]*100, f["model"]["wape"]*100]): ax[0].text(i, v+0.2, f"{v:.1f}%", ha="center")
th = R["anomaly"]["thresholds"]; ks = list(th)
ax[1].plot([th[k]["false_alarm_rate"]*100 for k in ks], [th[k]["recall"]*100 for k in ks], "o-", color="#0052CC")
for k in ks: ax[1].annotate(f"z<-{k}", (th[k]["false_alarm_rate"]*100, th[k]["recall"]*100), textcoords="offset points", xytext=(6, -12))
ax[1].set_xlabel("False-alarm rate (% of normal store-days)"); ax[1].set_ylabel("Simulated drops caught (%)"); ax[1].set_title("Anomaly detector trade-off")
for a in ax: a.spines[["top", "right"]].set_visible(False)
plt.tight_layout(); plt.savefig("results/summary.png", dpi=150)
