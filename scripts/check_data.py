import numpy as np, json, os
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

splits = json.load(open("results/splits.json"))
print("Dropped from train:", [i for i in splits["train"]["requested"] if i not in splits["train"]["used"]])
print("Dropped from val:",   [i for i in splits["val"]["requested"]   if i not in splits["val"]["used"]])
print("Dropped from test:",  [i for i in splits["test"]["requested"]  if i not in splits["test"]["used"]])

all_ppg, all_ecg = [], []
segs_per_subject_train = {}
for split in ["train","val","test"]:
    d = np.load(f"data/{split}.npz")
    all_ppg.append(d["ppg"]); all_ecg.append(d["ecg"])

ppg = np.concatenate(all_ppg); ecg = np.concatenate(all_ecg)
ppg_ptp = ppg.max(1) - ppg.min(1)
ecg_ptp = ecg.max(1) - ecg.min(1)
print(f"PPG ptp  mean={ppg_ptp.mean():.3f}  std={ppg_ptp.std():.3f}  p5={np.percentile(ppg_ptp,5):.3f}  p95={np.percentile(ppg_ptp,95):.3f}")
print(f"ECG ptp  mean={ecg_ptp.mean():.3f}  std={ecg_ptp.std():.3f}  p5={np.percentile(ecg_ptp,5):.3f}  p95={np.percentile(ecg_ptp,95):.3f}")

# HR estimates from ECG (simple RR via autocorrelation)
import scipy.signal
hrs = []
for seg in ecg[::5]:  # sample every 5th
    peaks, _ = scipy.signal.find_peaks(seg, distance=50)
    if len(peaks) >= 2:
        rr = np.diff(peaks).mean()
        hrs.append(60.0 / (rr / 125.0))
hrs = np.array(hrs)
hrs = hrs[(hrs > 30) & (hrs < 200)]
print(f"HR  mean={hrs.mean():.1f}  std={hrs.std():.1f}  min={hrs.min():.1f}  max={hrs.max():.1f}")
print(f"HR dist: <60={np.sum(hrs<60)}  60-100={np.sum((hrs>=60)&(hrs<=100))}  >100={np.sum(hrs>100)}")
print("DONE")
