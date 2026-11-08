from pathlib import Path
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
root=Path(__file__).resolve().parent
raw=json.loads((root.parent/"p23_p24_2026-11-06/raw_verification.json").read_text())
keys=["19C","19E","1C","1E"]
fig,axs=plt.subplots(1,2,figsize=(11,4.3),layout="constrained")
x=np.arange(4)
for ax,field,title in zip(axs,["dead_recall","precision"],["Internal Dead recall","All-object unique-match precision"]):
 for offset,cohort,color in [(-.18,"fit","#247b9b"),(.18,"audit","#d88735")]:
  vals=[raw["summaries"][k][cohort][field] for k in keys]
  bars=ax.bar(x+offset,vals,width=.36,label=cohort,color=color)
  ax.bar_label(bars,labels=[f"{v:.3f}" for v in vals],fontsize=8,padding=3)
 ax.set_xticks(x,["C seed19","EMA seed19","C seed1","EMA seed1"])
 ax.set_ylim(0,1.10);ax.set_title(title);ax.spines[["top","right"]].set_visible(False)
 ax.axhline(.60 if field=="dead_recall" else .75,color="#555555",linestyle="--",linewidth=1,label="audit admission floor")
 ax.legend(loc="lower left",fontsize=8)
fig.suptitle("P23: strong fit performance, substantial audit gap (reused fold1 development)")
(root/"figures").mkdir(exist_ok=True)
for ext in ["png","pdf"]:fig.savefig(root/"figures"/("fit_audit_gap."+ext),dpi=180)
plt.close(fig)
metrics={k:{"fit_dead":raw["summaries"][k]["fit"]["dead_recall"],"audit_dead":raw["summaries"][k]["audit"]["dead_recall"],"gap":raw["summaries"][k]["fit"]["dead_recall"]-raw["summaries"][k]["audit"]["dead_recall"]} for k in keys}
(root/"figures/plot_values.json").write_text(json.dumps(metrics,indent=2)+"\n")
print(json.dumps(metrics))
