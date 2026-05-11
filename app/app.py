"""PPG/ECG analysis web app.

Upload a CSV containing a PPG column, an ECG column, or both.
The app reconstructs ECG from PPG when ECG is absent, then classifies
cardiac rhythm using rule-based analysis on the ECG signal.

Run:
    cd ppg2ecg
    streamlit run app/app.py
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components
from plotly.subplots import make_subplots

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from classifier import classify         # noqa: E402
from inference import filter_ecg, load_model, reconstruct_ecg  # noqa: E402

FS           = 125
CKPT_DEFAULT = ROOT / "checkpoints" / "bilstm" / "best.pt"


# ── Canvas-based ECG animation (60 fps, no Plotly overhead) ──────────────────
def _ecg_canvas_html(ecg: np.ndarray, r_peaks: list[int],
                     beat_labels: list[str], fs: int = 125) -> str:
    ecg_js     = json.dumps([round(float(x), 4) for x in ecg])
    peak_map   = {str(p): lbl for p, lbl in zip(r_peaks, beat_labels)}
    peaks_js   = json.dumps(peak_map)
    colours_js = json.dumps({
        "N": "#00FF41", "S": "#FF9800", "V": "#FF3333",
        "F": "#CC88FF", "Q": "#888888",
    })
    # Auto-scale: map 95th-percentile amplitude to 38% of half-height
    p05, p95  = float(np.percentile(ecg, 5)), float(np.percentile(ecg, 95))
    amp       = max(abs(p05), abs(p95), 0.3)
    amp_frac  = round(0.38 / amp, 5)

    return f"""<!DOCTYPE html><html><head><meta charset="utf-8">
<style>
  *{{margin:0;padding:0;box-sizing:border-box}}
  html,body{{width:100%;height:100%;background:#0d0d0d;overflow:hidden;font-family:monospace}}
  canvas{{display:block;width:100%}}
  #bar{{position:absolute;top:8px;left:10px;display:flex;gap:8px;align-items:center;z-index:9}}
  button{{background:#0d0d0d;color:#00FF41;border:1px solid #00FF41;
          padding:5px 14px;font-family:monospace;font-size:13px;
          cursor:pointer;border-radius:4px;transition:background .15s}}
  button:hover{{background:#1a3a1a}}
  #spd{{color:#00FF41;font-size:13px;min-width:70px;user-select:none}}
</style></head><body>
<div id="bar">
  <button id="bPlay">&#9654; Play</button>
  <button id="bPause" style="display:none">&#9646;&#9646; Pause</button>
  <button id="bReset">&#8635; Reset</button>
  <button id="bSlow">&#8722;</button>
  <span   id="spd">1.0&times;</span>
  <button id="bFast">&#43;</button>
</div>
<canvas id="c"></canvas>
<script>
const ECG={ecg_js}, PEAK_LABELS={peaks_js}, BEAT_COL={colours_js};
const FS={fs}, N=ECG.length, AMP={amp_frac};
const LH=300, DISP=10*FS;
const dpr=window.devicePixelRatio||1;

const canvas=document.getElementById('c');
const ctx=canvas.getContext('2d');
let LW, PX, gc, gx;

function initDims(){{
  LW = Math.floor(document.documentElement.clientWidth||window.innerWidth||1200);
  PX = LW/DISP;
  canvas.width=LW*dpr; canvas.height=LH*dpr;
  canvas.style.height=LH+'px';
  ctx.setTransform(dpr,0,0,dpr,0,0);
  gc=document.createElement('canvas');
  gc.width=LW*dpr; gc.height=LH*dpr;
  gx=gc.getContext('2d');
  gx.setTransform(dpr,0,0,dpr,0,0);
  buildGrid();
}}

function buildGrid(){{
  gx.fillStyle='#0d0d0d'; gx.fillRect(0,0,LW,LH);
  [[PX*FS*0.04,.4,'#0d1a0d'],[PX*FS*0.2,.9,'#1a3a1a'],[PX*FS,1.4,'#254025']].forEach(([s,w,c])=>{{
    gx.strokeStyle=c; gx.lineWidth=w;
    for(let x=0;x<LW;x+=s){{gx.beginPath();gx.moveTo(x,0);gx.lineTo(x,LH);gx.stroke();}}
  }});
  gx.strokeStyle='#1a3a1a'; gx.lineWidth=0.9;
  for(let y=LH/4;y<LH;y+=LH/4){{gx.beginPath();gx.moveTo(0,y);gx.lineTo(LW,y);gx.stroke();}}
  gx.strokeStyle='#254025'; gx.lineWidth=1.2;
  gx.beginPath();gx.moveTo(0,LH/2);gx.lineTo(LW,LH/2);gx.stroke();
}}

initDims();

const sx=s=>(s%DISP)*PX;
const sy=v=>LH/2-v*(LH*AMP);

let pos=0,lastTs=null,running=false,spd=1.0;

function blitGrid(){{ctx.drawImage(gc,0,0,LW*dpr,LH*dpr,0,0,LW,LH);}}

function fullDraw(){{
  blitGrid();
  if(pos<1)return;
  ctx.strokeStyle='#00FF41';ctx.lineWidth=1.5;ctx.lineJoin='round';
  ctx.beginPath();
  const s0=Math.max(0,Math.floor(pos)-DISP);
  let pen=false,px2=-1;
  for(let s=s0;s<Math.floor(pos)&&s<N;s++){{
    const x=sx(s),y=sy(ECG[s]);
    if(!pen||x<px2){{ctx.moveTo(x,y);pen=true;}}else{{ctx.lineTo(x,y);}}
    px2=x;
  }}
  ctx.stroke();
  for(const[sp,lbl]of Object.entries(PEAK_LABELS)){{
    const si=+sp;
    if(si>=s0&&si<Math.floor(pos)){{
      ctx.fillStyle=BEAT_COL[lbl]||'#fff';
      ctx.beginPath();ctx.arc(sx(si),sy(ECG[si]),4,0,6.28);ctx.fill();
    }}
  }}
}}

blitGrid();

function animate(ts){{
  if(!running)return;
  if(!lastTs){{lastTs=ts;requestAnimationFrame(animate);return;}}
  const dt=Math.min(ts-lastTs,50);lastTs=ts;
  const adv=(dt/1000)*FS*spd, old=pos;
  pos=Math.min(pos+adv,N-1);

  const cx=sx(Math.floor(pos)), ew=PX*28, ex=(cx+2)%LW;
  if(ex+ew<=LW){{
    ctx.drawImage(gc,ex*dpr,0,ew*dpr,LH*dpr,ex,0,ew,LH);
  }}else{{
    ctx.drawImage(gc,ex*dpr,0,(LW-ex)*dpr,LH*dpr,ex,0,LW-ex,LH);
    const w2=(ex+ew)%LW;
    ctx.drawImage(gc,0,0,w2*dpr,LH*dpr,0,0,w2,LH);
  }}

  ctx.strokeStyle='#00FF41';ctx.lineWidth=1.5;ctx.lineJoin='round';
  ctx.beginPath();
  let pen=false,px2=-1;
  for(let s=Math.floor(old);s<=Math.floor(pos)&&s<N;s++){{
    const x=sx(s),y=sy(ECG[s]);
    if(!pen||x<px2){{ctx.moveTo(x,y);pen=true;}}else{{ctx.lineTo(x,y);}}
    px2=x;
  }}
  ctx.stroke();

  for(let s=Math.floor(old);s<=Math.floor(pos)&&s<N;s++){{
    const lbl=PEAK_LABELS[s];
    if(lbl){{ctx.fillStyle=BEAT_COL[lbl]||'#fff';ctx.beginPath();ctx.arc(sx(s),sy(ECG[s]),4,0,6.28);ctx.fill();}}
  }}

  ctx.shadowColor='#00FF41';ctx.shadowBlur=14;
  ctx.fillStyle='#fff';ctx.beginPath();ctx.arc(cx,sy(ECG[Math.floor(pos)]),5,0,6.28);ctx.fill();
  ctx.shadowBlur=0;

  if(pos<N-1){{requestAnimationFrame(animate);}}
  else{{running=false;document.getElementById('bPlay').style.display='inline';document.getElementById('bPause').style.display='none';}}
}}

document.getElementById('bPlay').onclick=()=>{{
  if(pos>=N-1){{pos=0;fullDraw();}}
  running=true;lastTs=null;
  document.getElementById('bPlay').style.display='none';
  document.getElementById('bPause').style.display='inline';
  requestAnimationFrame(animate);
}};
document.getElementById('bPause').onclick=()=>{{
  running=false;
  document.getElementById('bPlay').style.display='inline';
  document.getElementById('bPause').style.display='none';
}};
document.getElementById('bReset').onclick=()=>{{
  running=false;pos=0;lastTs=null;fullDraw();
  document.getElementById('bPlay').style.display='inline';
  document.getElementById('bPause').style.display='none';
}};
document.getElementById('bSlow').onclick=()=>{{spd=Math.max(.25,+(spd-.25).toFixed(2));document.getElementById('spd').textContent=spd+'×';}};
document.getElementById('bFast').onclick=()=>{{spd=Math.min(8,+(spd+.25).toFixed(2));document.getElementById('spd').textContent=spd+'×';}};
window.addEventListener('resize',()=>{{const r=running;running=false;initDims();fullDraw();if(r){{running=true;lastTs=null;requestAnimationFrame(animate);}}}});
</script></body></html>"""

LABEL_COLOURS = {
    "Normal Sinus Rhythm":        "green",
    "Bradycardia":                 "blue",
    "Tachycardia":                 "orange",
    "Probable Atrial Fibrillation": "red",
    "Frequent Ectopy":             "orange",
    "Unanalysable":                "grey",
}
BEAT_COLOURS = {"N": "#00FF41", "S": "#FF9800", "V": "#FF3333",
                "F": "#CC88FF", "Q": "#888888"}

# ── CSS injected into every page load ────────────────────────────────────────
_APP_CSS = """
<style>
/* Dark app background */
.stApp, [data-testid="stAppViewContainer"] {
    background: #080808 !important;
}
[data-testid="stSidebar"] {
    background: #0d0d0d !important;
    border-right: 1px solid #1a3a1a !important;
}
/* File uploader — green dashed card */
[data-testid="stFileUploaderDropzone"] {
    background: rgba(0,255,65,0.03) !important;
    border: 2px dashed rgba(0,255,65,0.4) !important;
    border-radius: 14px !important;
    transition: all 0.2s ease !important;
}
[data-testid="stFileUploaderDropzone"]:hover {
    background: rgba(0,255,65,0.07) !important;
    border-color: #00FF41 !important;
}
[data-testid="stFileUploaderDropzoneInstructions"] span,
[data-testid="stFileUploaderDropzoneInstructions"] p {
    color: #336633 !important;
    font-family: monospace !important;
}
/* Browse button */
[data-testid="stFileUploaderDropzone"] button {
    background: rgba(0,255,65,0.08) !important;
    color: #00FF41 !important;
    border: 1px solid #00FF41 !important;
    font-family: monospace !important;
    font-weight: 700 !important;
    letter-spacing: 1px !important;
    border-radius: 8px !important;
}
[data-testid="stFileUploaderDropzone"] button:hover {
    background: rgba(0,255,65,0.18) !important;
}
/* Sidebar text */
.stSidebar label, .stSidebar .stTextInput label,
.stSidebar p, .stSidebar span {
    color: #336633 !important;
    font-family: monospace !important;
}
.stSidebar input {
    background: #111 !important;
    color: #00FF41 !important;
    border-color: #1a3a1a !important;
    font-family: monospace !important;
}
</style>
"""

# ── Animated landing page (shown when no file is uploaded) ───────────────────
def _landing_html() -> str:
    return """<!DOCTYPE html><html><head><meta charset="utf-8">
<style>
  *{margin:0;padding:0;box-sizing:border-box}
  html,body{width:100%;height:100%;background:#080808;overflow:hidden}
  canvas{position:absolute;top:0;left:0;width:100%;height:100%}
  .card{
    position:absolute;top:50%;left:50%;transform:translate(-50%,-50%);
    text-align:center;pointer-events:none;
  }
  .heart{
    width:90px;height:90px;display:block;margin:0 auto 18px;
    animation:hb 1.1s ease-in-out infinite;
    filter:drop-shadow(0 0 22px rgba(255,34,68,0.8));
  }
  @keyframes hb{
    0%,100%{transform:scale(1)}
    14%{transform:scale(1.28)}
    28%{transform:scale(1.04)}
    42%{transform:scale(1.18)}
    70%{transform:scale(1)}
  }
  h1{
    color:#00FF41;font-family:monospace;font-size:52px;font-weight:900;
    letter-spacing:6px;text-shadow:0 0 40px rgba(0,255,65,0.55);
    margin-bottom:10px;
  }
  .sub{
    color:#2d6b2d;font-family:monospace;font-size:14px;letter-spacing:3px;
    margin-bottom:30px;
  }
  .arrow{
    color:rgba(0,255,65,0.5);font-family:monospace;font-size:28px;
    animation:bounce .9s ease-in-out infinite;display:block;
  }
  @keyframes bounce{0%,100%{transform:translateY(0)}50%{transform:translateY(10px)}}
</style></head><body>
<canvas id="c"></canvas>
<div class="card">
  <svg class="heart" viewBox="0 0 100 90">
    <defs>
      <radialGradient id="hg" cx="50%" cy="40%">
        <stop offset="0%" stop-color="#ff6688"/>
        <stop offset="100%" stop-color="#cc0022"/>
      </radialGradient>
    </defs>
    <path fill="url(#hg)" d="M50,82 C50,82 4,52 4,27 C4,13 15,4 27,4 C36,4 44,9 50,18 C56,9 64,4 73,4 C85,4 96,13 96,27 C96,52 50,82 50,82Z"/>
  </svg>
  <h1>ECG ANALYSIS</h1>
  <div class="sub">DEEP LEARNING CARDIAC RHYTHM MONITORING</div>
  <span class="arrow">&#8595; upload your csv below &#8595;</span>
</div>
<script>
const canvas=document.getElementById('c');
const ctx=canvas.getContext('2d');
const dpr=window.devicePixelRatio||1;
const LH=420, FS=125, HR=72, DISP=10*FS;
let LW, PX, gc;

// Synthetic PQRST waveform
function pqrst(tn){
  return  0.14*Math.exp(-Math.pow(tn-.15,2)/.00097)
        - 0.06*Math.exp(-Math.pow(tn-.295,2)/.000162)
        + 1.55*Math.exp(-Math.pow(tn-.33,2)/.000098)
        - 0.21*Math.exp(-Math.pow(tn-.365,2)/.0002)
        + 0.33*Math.exp(-Math.pow(tn-.55,2)/.0032);
}
const BEAT=Math.ceil(FS*(60/HR));
const WAVE=Array.from({length:BEAT},(_,i)=>pqrst(i/BEAT));
const waveAt=s=>WAVE[((s%BEAT)+BEAT)%BEAT]||0;

function buildOffscreen(){
  gc=document.createElement('canvas');
  gc.width=LW*dpr; gc.height=LH*dpr;
  const g=gc.getContext('2d');
  g.setTransform(dpr,0,0,dpr,0,0);
  g.fillStyle='#080808'; g.fillRect(0,0,LW,LH);
  PX=LW/DISP;
  // fine grid
  g.strokeStyle='#0c170c'; g.lineWidth=0.5;
  for(let x=0;x<LW;x+=PX*FS*.04){g.beginPath();g.moveTo(x,0);g.lineTo(x,LH);g.stroke();}
  // medium grid
  g.strokeStyle='#132813'; g.lineWidth=0.9;
  for(let x=0;x<LW;x+=PX*FS*.2){g.beginPath();g.moveTo(x,0);g.lineTo(x,LH);g.stroke();}
  for(let y=LH*.25;y<LH;y+=LH*.25){g.beginPath();g.moveTo(0,y);g.lineTo(LW,y);g.stroke();}
  // major
  g.strokeStyle='#1c3a1c'; g.lineWidth=1.3;
  for(let x=0;x<LW;x+=PX*FS){g.beginPath();g.moveTo(x,0);g.lineTo(x,LH);g.stroke();}
}

function resize(){
  LW=document.documentElement.clientWidth||window.innerWidth||1200;
  canvas.width=LW*dpr; canvas.height=LH*dpr;
  canvas.style.height=LH+'px';
  ctx.setTransform(dpr,0,0,dpr,0,0);
  buildOffscreen();
}
resize();

const sy=v=>LH*.5-v*(LH*.175);
const sx=s=>(s%DISP)*PX;

let sweep=0, lastTs=null;
function animate(ts){
  if(!lastTs){lastTs=ts;requestAnimationFrame(animate);return;}
  const dt=Math.min(ts-lastTs,50); lastTs=ts;
  const adv=(dt/1000)*FS, old=sweep;
  sweep+=adv;

  const cx=sx(Math.floor(sweep)), ew=PX*32, ex=(cx+2)%LW;
  if(ex+ew<=LW){
    ctx.drawImage(gc,ex*dpr,0,ew*dpr,LH*dpr,ex,0,ew,LH);
  }else{
    ctx.drawImage(gc,ex*dpr,0,(LW-ex)*dpr,LH*dpr,ex,0,LW-ex,LH);
    const w2=(ex+ew)%LW;
    ctx.drawImage(gc,0,0,w2*dpr,LH*dpr,0,0,w2,LH);
  }

  ctx.strokeStyle='rgba(0,255,65,0.6)'; ctx.lineWidth=2; ctx.lineJoin='round';
  ctx.beginPath();
  let pen=false, px2=-1;
  for(let s=Math.floor(old);s<=Math.floor(sweep);s++){
    const x=sx(s),y=sy(waveAt(s));
    if(!pen||x<px2){ctx.moveTo(x,y);pen=true;}else{ctx.lineTo(x,y);}
    px2=x;
  }
  ctx.stroke();

  // cursor glow
  ctx.shadowColor='#00FF41'; ctx.shadowBlur=20;
  ctx.fillStyle='#00FF41';
  ctx.beginPath();ctx.arc(cx,sy(waveAt(Math.floor(sweep))),4,0,6.28);ctx.fill();
  ctx.shadowBlur=0;

  requestAnimationFrame(animate);
}
ctx.drawImage(gc,0,0,LW*dpr,LH*dpr,0,0,LW,LH);
requestAnimationFrame(animate);
window.addEventListener('resize',()=>{resize();ctx.drawImage(gc,0,0,LW*dpr,LH*dpr,0,0,LW,LH);});
</script></body></html>"""

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ECG Analysis",
    page_icon="🫀",
    layout="wide",
)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("Settings")
    ckpt_path    = st.text_input("BiLSTM checkpoint", str(CKPT_DEFAULT))
    ppg_col      = st.text_input("PPG column name", "PLETH")
    ecg_col      = st.text_input("ECG column name", "II")
    hospital_mode = st.toggle("Hospital monitor view", value=False)
    st.markdown("---")
    st.caption(
        "Rhythm classification uses a ResNet1D beat classifier (MIT-BIH, 99.4% val accuracy) "
        "combined with RR-interval analysis. "
        "This tool is for research purposes only and is not a medical device."
    )

# ── Load reconstruction model (cached) ───────────────────────────────────────
@st.cache_resource(show_spinner="Loading reconstruction model...")
def get_model(path: str):
    try:
        return load_model(path)
    except Exception as e:
        return None, str(e)

model, dev_or_err = get_model(ckpt_path)
if model is None:
    st.sidebar.error(f"Model load failed: {dev_or_err}")

# ── Global CSS ────────────────────────────────────────────────────────────────
st.markdown(_APP_CSS, unsafe_allow_html=True)

# ── Landing placeholder (replaced by analysis content once a file is loaded) ──
_hero = st.empty()

# ── File uploader ─────────────────────────────────────────────────────────────
uploaded = st.file_uploader(
    "📂  Drop CSV here or click to browse  —  expected columns: PLETH (PPG) and/or II (ECG)",
    type=["csv"],
)

if uploaded is None:
    with _hero:
        components.html(_landing_html(), height=450, scrolling=False)
    st.stop()

_hero.empty()   # file is present — clear the landing hero

# ── Parse CSV ─────────────────────────────────────────────────────────────────
try:
    df = pd.read_csv(uploaded)
    df.columns = [c.strip() for c in df.columns]
except Exception as e:
    st.error(f"Could not read CSV: {e}")
    st.stop()

has_ppg = ppg_col in df.columns
has_ecg = ecg_col in df.columns

if not has_ppg and not has_ecg:
    st.error(
        f"Neither '{ppg_col}' nor '{ecg_col}' column found. "
        f"Available columns: {list(df.columns)}"
    )
    st.stop()

if not hospital_mode:
    col_info = []
    if has_ppg:
        col_info.append(f"PPG (`{ppg_col}`)")
    if has_ecg:
        col_info.append(f"ECG (`{ecg_col}`)")
    st.success(f"Detected: {' and '.join(col_info)}  —  {len(df):,} samples  ({len(df)/FS:.1f} s)")

# ── Extract arrays ────────────────────────────────────────────────────────────
ppg_raw = df[ppg_col].to_numpy(dtype=np.float32) if has_ppg else None
ecg_raw = df[ecg_col].to_numpy(dtype=np.float32) if has_ecg else None

# ── Reconstruction ────────────────────────────────────────────────────────────
reconstructed = False
if not has_ecg:
    if model is None:
        st.error("No ECG column found and reconstruction model failed to load.")
        st.stop()
    with st.spinner("Reconstructing ECG from PPG..."):
        ecg_proc = reconstruct_ecg(ppg_raw, model, dev_or_err)
    reconstructed = True
    if not hospital_mode:
        st.info("ECG was not found in the CSV — shown signal is reconstructed from PPG.")
else:
    ecg_proc = filter_ecg(ecg_raw)

# ── Clip to 60 s for display ──────────────────────────────────────────────────
MAX_SAMPLES = 60 * FS
t          = np.arange(len(ecg_proc)) / FS
ecg_disp   = ecg_proc[:MAX_SAMPLES]
t_disp     = t[:MAX_SAMPLES]
ppg_disp   = ppg_raw[:MAX_SAMPLES] if ppg_raw is not None else None

# ── Classification ────────────────────────────────────────────────────────────
with st.spinner("Analysing rhythm..."):
    result = classify(ecg_disp, fs=FS, reconstructed=reconstructed)

colour     = LABEL_COLOURS.get(result.label, "grey")
conf_emoji = {"High": "🟢", "Medium": "🟡", "Low": "🔴"}.get(result.confidence, "")
hr_str     = f"{result.heart_rate:.0f}" if result.heart_rate == result.heart_rate else "---"

# ─────────────────────────────────────────────────────────────────────────────
#  HOSPITAL MONITOR VIEW
# ─────────────────────────────────────────────────────────────────────────────
if hospital_mode:
    _ALARM_COLOUR = {
        "Normal Sinus Rhythm":         "#00FF41",
        "Bradycardia":                 "#33AAFF",
        "Tachycardia":                 "#FF9900",
        "Probable Atrial Fibrillation": "#FF3333",
        "Frequent Ectopy":             "#FF9900",
        "Unanalysable":                "#888888",
    }
    alarm_col = _ALARM_COLOUR.get(result.label, "#888888")
    spo2_str  = "--"   # placeholder — BIDMC has no SpO2 column
    rr_str    = f"{int(60000 / np.mean(result.rr_intervals_ms))}" if result.rr_intervals_ms else "---"

    # ── Vitals header bar ────────────────────────────────────────────────────
    st.markdown(
        f"""
        <div style='background:#0d0d0d;padding:18px 28px;border-radius:10px;
                    display:flex;align-items:flex-end;gap:60px;margin-bottom:8px;
                    border:1px solid #1a3a1a;'>

          <div>
            <div style='color:#888;font-family:monospace;font-size:13px;letter-spacing:2px;'>HR</div>
            <div style='color:#00FF41;font-family:monospace;font-size:72px;
                        font-weight:700;line-height:1;'>{hr_str}</div>
            <div style='color:#00FF41;font-family:monospace;font-size:16px;'>bpm</div>
          </div>

          <div>
            <div style='color:#888;font-family:monospace;font-size:13px;letter-spacing:2px;'>RHYTHM</div>
            <div style='color:{alarm_col};font-family:monospace;font-size:28px;
                        font-weight:700;line-height:1.3;'>{result.label}</div>
            <div style='color:{alarm_col};font-family:monospace;font-size:14px;'>
                {conf_emoji} {result.confidence} confidence</div>
          </div>

          <div>
            <div style='color:#888;font-family:monospace;font-size:13px;letter-spacing:2px;'>RMSSD</div>
            <div style='color:#FFD700;font-family:monospace;font-size:36px;
                        font-weight:700;line-height:1;'>
              {f"{result.hrv['RMSSD_ms']:.0f}" if result.hrv.get("RMSSD_ms") else "---"}
            </div>
            <div style='color:#FFD700;font-family:monospace;font-size:14px;'>ms</div>
          </div>

          <div>
            <div style='color:#888;font-family:monospace;font-size:13px;letter-spacing:2px;'>BEATS</div>
            <div style='color:#00BFFF;font-family:monospace;font-size:36px;
                        font-weight:700;line-height:1;'>{len(result.r_peaks)}</div>
            <div style='color:#00BFFF;font-family:monospace;font-size:14px;'>detected</div>
          </div>

        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── Animated ECG (60 fps Canvas) ─────────────────────────────────────────
    canvas_html = _ecg_canvas_html(ecg_disp, result.r_peaks, result.beat_labels, FS)
    components.html(canvas_html, height=340)

    # ── Beat legend ──────────────────────────────────────────────────────────
    if result.beat_labels:
        bc = Counter(result.beat_labels)
        total = len(result.beat_labels)
        legend_html = "<div style='background:#0d0d0d;padding:10px 20px;border-radius:8px;" \
                      "border:1px solid #1a3a1a;display:flex;gap:30px;font-family:monospace;'>"
        labels_desc = {"N": "Normal", "S": "Supra-V", "V": "Ventricular", "F": "Fusion", "Q": "Unknown"}
        for cls in ["N", "S", "V", "F", "Q"]:
            count = bc.get(cls, 0)
            pct   = count / total * 100
            col   = BEAT_COLOURS[cls]
            legend_html += (
                f"<div><span style='color:{col};font-size:18px;'>●</span> "
                f"<span style='color:{col};font-size:13px;'>{cls} {labels_desc[cls]}</span><br>"
                f"<span style='color:#888;font-size:12px;'>{count} beats ({pct:.0f}%)</span></div>"
            )
        legend_html += "</div>"
        st.markdown(legend_html, unsafe_allow_html=True)

    # ── Alarm notes ──────────────────────────────────────────────────────────
    for note in result.notes:
        st.warning(note)

    # ── PPG sub-panel (if available) ─────────────────────────────────────────
    if ppg_disp is not None:
        with st.expander("SpO₂ / PPG waveform"):
            ppg_fig = go.Figure()
            ppg_fig.add_trace(go.Scatter(
                x=t_disp, y=ppg_disp, mode="lines",
                line=dict(color="#33AAFF", width=1.2), name="PPG",
            ))
            ppg_fig.update_layout(
                paper_bgcolor="#0d0d0d", plot_bgcolor="#0d0d0d",
                font=dict(color="#33AAFF", family="monospace"),
                xaxis=dict(showgrid=True, gridcolor="#111a2a", zeroline=False,
                           tickfont=dict(color="#336688")),
                yaxis=dict(showgrid=True, gridcolor="#111a2a", zeroline=False,
                           tickfont=dict(color="#336688")),
                height=180, margin=dict(l=40, r=20, t=10, b=30),
                showlegend=False,
            )
            st.plotly_chart(ppg_fig, use_container_width=True)

    # ── HRV + RR in hospital style ───────────────────────────────────────────
    if result.hrv or len(result.rr_intervals_ms) > 2:
        with st.expander("HRV / RR tachogram"):
            if result.hrv:
                hcols = st.columns(len(result.hrv))
                for col, (k, v) in zip(hcols, result.hrv.items()):
                    col.metric(k, f"{v:.3f}")
            if len(result.rr_intervals_ms) > 2:
                rr_fig = go.Figure()
                rr_fig.add_trace(go.Scatter(
                    y=result.rr_intervals_ms, mode="lines+markers",
                    line=dict(color="#FFD700", width=1.5),
                    marker=dict(size=4, color="#FFD700"),
                ))
                rr_fig.update_layout(
                    paper_bgcolor="#0d0d0d", plot_bgcolor="#0d0d0d",
                    font=dict(color="#FFD700", family="monospace"),
                    xaxis=dict(showgrid=True, gridcolor="#1a1a00",
                               title="Beat number", zeroline=False,
                               tickfont=dict(color="#666633")),
                    yaxis=dict(showgrid=True, gridcolor="#1a1a00",
                               title="RR interval (ms)", zeroline=False,
                               tickfont=dict(color="#666633")),
                    height=220, margin=dict(l=50, r=20, t=10, b=40),
                    showlegend=False,
                )
                st.plotly_chart(rr_fig, use_container_width=True)

    if reconstructed:
        out_df = pd.DataFrame({"time_s": t[:len(ecg_proc)], "ecg_reconstructed": ecg_proc})
        st.download_button(
            "Download reconstructed ECG (CSV)",
            out_df.to_csv(index=False).encode(),
            file_name="ecg_reconstructed.csv",
            mime="text/csv",
        )

    st.stop()

# ─────────────────────────────────────────────────────────────────────────────
#  STANDARD VIEW
# ─────────────────────────────────────────────────────────────────────────────
st.markdown(
    f"<h2 style='color:{colour}'>{result.label}</h2>",
    unsafe_allow_html=True,
)

metric_cols = st.columns(4)
metric_cols[0].metric("Heart Rate", f"{hr_str} bpm")
metric_cols[1].metric("Confidence", f"{conf_emoji} {result.confidence}")
metric_cols[2].metric("RR intervals", f"{len(result.rr_intervals_ms)}")
if result.hrv.get("RMSSD_ms"):
    metric_cols[3].metric("RMSSD", f"{result.hrv['RMSSD_ms']:.1f} ms")

if result.beat_labels:
    bc = Counter(result.beat_labels)
    total_beats = len(result.beat_labels)
    beat_summary = "  ".join(
        f"**{cls}**: {bc.get(cls, 0)} ({bc.get(cls, 0)/total_beats*100:.0f}%)"
        for cls in ["N", "S", "V", "F", "Q"]
    )
    st.info(f"Beat classification (ResNet1D / MIT-BIH):  {beat_summary}")

for note in result.notes:
    st.warning(note)

# ── Waveform plot ─────────────────────────────────────────────────────────────
n_rows = 2 if (has_ppg and ppg_disp is not None) else 1
titles = []
if has_ppg and ppg_disp is not None:
    titles.append("PPG")
titles.append("ECG (reconstructed)" if reconstructed else "ECG")

fig = make_subplots(rows=n_rows, cols=1, shared_xaxes=True,
                    subplot_titles=titles, vertical_spacing=0.08)

row = 1
if has_ppg and ppg_disp is not None:
    fig.add_trace(
        go.Scatter(x=t_disp, y=ppg_disp, mode="lines",
                   line=dict(color="#2196F3", width=1), name="PPG"),
        row=1, col=1,
    )
    row = 2

ecg_colour = "#FF9800" if reconstructed else "#4CAF50"
fig.add_trace(
    go.Scatter(x=t_disp, y=ecg_disp, mode="lines",
               line=dict(color=ecg_colour, width=1),
               name="ECG (reconstructed)" if reconstructed else "ECG"),
    row=row, col=1,
)

if result.r_peaks:
    rp = [(i, p) for i, p in enumerate(result.r_peaks) if p < len(ecg_disp)]
    if result.beat_labels and len(result.beat_labels) == len(result.r_peaks):
        STD_BEAT_COLOURS = {"N": "green", "S": "orange", "V": "red",
                            "F": "purple", "Q": "grey"}
        for beat_cls, bc_col in STD_BEAT_COLOURS.items():
            pts = [(i, p) for i, p in rp if result.beat_labels[i] == beat_cls]
            if not pts:
                continue
            _, peak_samples = zip(*pts)
            fig.add_trace(
                go.Scatter(
                    x=[t_disp[p] for p in peak_samples],
                    y=[ecg_disp[p] for p in peak_samples],
                    mode="markers",
                    marker=dict(color=bc_col, size=7, symbol="triangle-up"),
                    name=f"Beat {beat_cls}",
                ),
                row=row, col=1,
            )
    elif rp:
        _, peak_samples = zip(*rp)
        fig.add_trace(
            go.Scatter(
                x=[t_disp[p] for p in peak_samples],
                y=[ecg_disp[p] for p in peak_samples],
                mode="markers",
                marker=dict(color="red", size=6, symbol="triangle-up"),
                name="R-peaks",
            ),
            row=row, col=1,
        )

fig.update_xaxes(title_text="Time (s)", row=row, col=1)
fig.update_layout(
    height=350 * n_rows,
    showlegend=True,
    margin=dict(l=40, r=20, t=40, b=40),
    hovermode="x unified",
)
st.plotly_chart(fig, use_container_width=True)

# ── HRV detail ────────────────────────────────────────────────────────────────
if result.hrv:
    with st.expander("HRV detail"):
        hrv_cols = st.columns(len(result.hrv))
        for col, (k, v) in zip(hrv_cols, result.hrv.items()):
            col.metric(k, f"{v:.3f}")

# ── RR interval tachogram ─────────────────────────────────────────────────────
if len(result.rr_intervals_ms) > 2:
    with st.expander("RR interval tachogram"):
        rr_fig = go.Figure()
        rr_fig.add_trace(go.Scatter(
            y=result.rr_intervals_ms, mode="lines+markers",
            line=dict(color="#9C27B0", width=1.5),
            marker=dict(size=4),
        ))
        rr_fig.update_layout(
            xaxis_title="Beat number",
            yaxis_title="RR interval (ms)",
            height=280,
            margin=dict(l=40, r=20, t=20, b=40),
        )
        st.plotly_chart(rr_fig, use_container_width=True)

# ── Download reconstructed ECG ────────────────────────────────────────────────
if reconstructed:
    out_df = pd.DataFrame({"time_s": t[:len(ecg_proc)], "ecg_reconstructed": ecg_proc})
    st.download_button(
        "Download reconstructed ECG (CSV)",
        out_df.to_csv(index=False).encode(),
        file_name="ecg_reconstructed.csv",
        mime="text/csv",
    )
