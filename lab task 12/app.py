from flask import Flask, request, jsonify, render_template_string, send_file
import pandas as pd
import pickle, os, csv
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error

app = Flask(__name__)
MODEL_FILE = "mobile_price_model.pkl"
HISTORY_FILE = "prediction_history.csv"

# ---------- FRONTEND ----------
HTML_PAGE = """
<!DOCTYPE html>
<html>
<head>
  <title>📱 Mobile Price Predictor</title>
  <style>
    :root{--bg:#f7f7f7;--text:#222;--card:#fff;--border:#ddd;--button:#333;--button-text:#fff;--accent:#4CAF50}
    body.dark{--bg:#0f1113;--text:#eaeaea;--card:#111;--border:#333}
    body{font-family:Segoe UI,Arial;padding:30px;background:var(--bg);color:var(--text)}
    .card{max-width:900px;margin:auto;background:var(--card);padding:24px;border-radius:10px;border:1px solid var(--border)}
    h2{text-align:center}
    label,input,button{display:block;margin:8px 0}
    input[type=number]{padding:8px;width:220px}
    .row{display:flex;gap:12px;flex-wrap:wrap}
    .metric{border:1px solid var(--border);padding:10px;border-radius:6px;width:260px}
    #result{display:none;padding:10px;background:var(--accent);color:white;border-radius:6px}
  </style>
  <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
 </head>
<body>
<div class="card">
  <h2>📱 Mobile Price Predictor</h2>

  <section>
    <h3>Train model (upload CSV)</h3>
    <input type="file" id="fileInput" accept=".csv">
    <button onclick="uploadFile()">Train Model</button>
    <div id="trainStatus"></div>
    <div class="metric"><strong>Model metrics</strong><div id="metrics">No model</div></div>
  </section>

  <section>
    <h3>Predict Price</h3>
    <div class="row">
      <div>
        <label>Ratings</label>
        <input id="Ratings" type="number" step="0.1" value="4.0">
        <label>RAM (GB)</label>
        <input id="RAM" type="number" step="1" value="4">
        <label>ROM (GB)</label>
        <input id="ROM" type="number" step="1" value="64">
      </div>
      <div>
        <label>Mobile Size (inches)</label>
        <input id="Mobile_Size" type="number" step="0.01" value="6.0">
        <label>Primary Camera (MP)</label>
        <input id="Primary_Cam" type="number" step="1" value="48">
        <label>Selfie Camera (MP)</label>
        <input id="Selfi_Cam" type="number" step="1" value="12">
        <label>Battery Power (mAh)</label>
        <input id="Battery_Power" type="number" step="1" value="4000">
      </div>
    </div>
    <button onclick="predict()">Predict Price</button>
    <div id="result"></div>
  </section>

  <section>
    <h3>Feature importance</h3>
    <canvas id="featureChart" width="600" height="200"></canvas>
  </section>

  <section>
    <h3>Prediction history</h3>
    <button onclick="loadHistory()">Refresh</button>
    <button onclick="downloadHistory()">Download CSV</button>
    <div id="history"></div>
  </section>
</div>

<script>
async function uploadFile(){
  const f=document.getElementById('fileInput'); if(!f.files.length){alert('Select CSV');return}
  const form=new FormData(); form.append('file', f.files[0]);
  const res=await (await fetch('/train',{method:'POST',body:form})).json();
  document.getElementById('trainStatus').innerText=res.message||res.error; loadMetrics(); loadFeatureChart();
}

async function loadMetrics(){
  const r=await fetch('/metrics'); const d=await r.json(); document.getElementById('metrics').innerText=d.r2?`R2: ${d.r2}  MAE: ${d.mae}`:'No model';
}

async function predict(){
  const fields=['Ratings','RAM','ROM','Mobile_Size','Primary_Cam','Selfi_Cam','Battery_Power'];
  const payload={}; fields.forEach(f=>{payload[f]=parseFloat(document.getElementById(f).value)});
  const r=await (await fetch('/predict',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(payload)})).json();
  const el=document.getElementById('result'); if(r.prediction!==undefined){el.style.display='block';el.innerText='Predicted Price: '+r.prediction}else{el.style.display='block';el.innerText='Error: '+(r.error||'unknown')}
  loadHistory();
}

async function loadFeatureChart(){
  const r=await fetch('/feature_importance'); const d=await r.json(); if(d.error) return; const ctx=document.getElementById('featureChart').getContext('2d'); if(window.chart)window.chart.destroy(); window.chart=new Chart(ctx,{type:'bar',data:{labels:d.features,datasets:[{label:'importance',data:d.importances,backgroundColor:'#4CAF50'}]}});
}

async function loadHistory(){const r=await fetch('/history'); const d=await r.json(); const div=document.getElementById('history'); if(!d.length){div.innerHTML='<p>No history</p>';return} let t='<table border=1><tr>'+Object.keys(d[0]).map(k=>`<th>${k}</th>`).join('')+'</tr>'; d.forEach(r=>t+='<tr>'+Object.values(r).map(v=>`<td>${v}</td>`).join('')+'</tr>'); div.innerHTML=t+'</table>'}

function downloadHistory(){window.location.href='/download_history'}

loadMetrics(); loadHistory(); loadFeatureChart();
</script>
</body>
</html>
"""

# ---------- BACKEND ----------
@app.route("/")
def home(): return render_template_string(HTML_PAGE)

@app.route("/options")
def options():
  # Return the expected feature names (case-sensitive for frontend inputs)
  features = ['Ratings','RAM','ROM','Mobile_Size','Primary_Cam','Selfi_Cam','Battery_Power']
  return jsonify({"features": features})

@app.route("/metrics")
def metrics():
  if not os.path.exists(MODEL_FILE): return jsonify({"r2": None, "mae": None})
  m = pickle.load(open(MODEL_FILE, "rb"))
  return jsonify(m.get("metrics", {"r2": None, "mae": None}))

@app.route("/feature_importance")
def feature_importance():
  if not os.path.exists(MODEL_FILE): return jsonify({"error": "Model not trained yet"})
  m = pickle.load(open(MODEL_FILE,"rb"))
  return jsonify({"features": m["features"], "importances": list(m["model"].feature_importances_)})

@app.route("/train", methods=["POST"])
def train():
  file = request.files.get("file")
  if not file:
    return jsonify({"error": "Please upload a CSV file"}), 400

  # Read CSV and normalize column names
  df = pd.read_csv(file)
  # normalize: strip and lowercase with underscores
  orig_cols = {c.strip().lower().replace(' ', '_'): c for c in df.columns}
  # required columns in normalized form
  req_norm = ['ratings', 'ram', 'rom', 'mobile_size', 'primary_cam', 'selfi_cam', 'battery_power', 'price']
  if not all(k in orig_cols for k in req_norm):
    return jsonify({"error": "CSV must include columns: Ratings,RAM,ROM,Mobile_Size,Primary_Cam,Selfi_Cam,Battery_Power,Price"}), 400

  # rename to standard names
  df = df.rename(columns={orig_cols[k]: k for k in orig_cols})

  # drop NA in required features
  features = ['ratings', 'ram', 'rom', 'mobile_size', 'primary_cam', 'selfi_cam', 'battery_power']
  df = df.dropna(subset=features + ['price'])
  X = df[features]
  y = df['price']

  Xt, Xv, yt, yv = train_test_split(X, y, test_size=0.2, random_state=42)
  model = RandomForestRegressor(n_estimators=100, random_state=42).fit(Xt, yt)
  yp = model.predict(Xv)
  metrics = {"r2": round(r2_score(yv, yp), 3), "mae": round(mean_absolute_error(yv, yp), 3)}

  # save model; store features in TitleCase to match frontend input ids
  features_title = ['Ratings', 'RAM', 'ROM', 'Mobile_Size', 'Primary_Cam', 'Selfi_Cam', 'Battery_Power']
  pickle.dump({"model": model, "features": features_title, "metrics": metrics}, open(MODEL_FILE, 'wb'))

  return jsonify({"message": "✅ Model trained successfully!", "metrics": metrics})

@app.route("/predict", methods=["POST"])
def predict():
    if not os.path.exists(MODEL_FILE):
        return jsonify({"error": "Model not trained yet."}), 400

    data = request.get_json()
    m = pickle.load(open(MODEL_FILE, "rb"))
    model = m['model']
    features = m['features']

    # Build input row in same order as features
    row = []
    for f in features:
        v = data.get(f)
        if v is None:
            return jsonify({"error": f"Missing feature {f}"}), 400
        try:
            row.append(float(v))
        except:
            return jsonify({"error": f"Invalid value for {f}: {v}"}), 400

    p = round(float(model.predict([row])[0]), 2)
    save_history(data, p)
    return jsonify({"prediction": p})

@app.route("/history")
def history():
    if not os.path.exists(HISTORY_FILE): return jsonify([])
    return jsonify(pd.read_csv(HISTORY_FILE).to_dict(orient="records"))

@app.route("/download_history")
def download_history():
    if not os.path.exists(HISTORY_FILE): return jsonify({"error":"No history found"}),404
    return send_file(HISTORY_FILE, as_attachment=True)

def save_history(inp,pred):
    row={**inp,"predicted_mpg":pred}; exist=os.path.exists(HISTORY_FILE)
    with open(HISTORY_FILE,"a",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=list(row.keys()))
        if not exist: w.writeheader()
        w.writerow(row)

if __name__=="__main__":
    app.run(debug=True)