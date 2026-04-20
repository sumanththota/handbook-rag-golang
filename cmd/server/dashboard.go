package main

import (
	"bufio"
	"encoding/json"
	"fmt"
	"net/http"
	"os"
)

type evalRecord struct {
	Timestamp       string      `json:"timestamp"`
	ChunkWords      int         `json:"chunk_words"`
	TopK            int         `json:"top_k"`
	Collection      string      `json:"collection"`
	Question        string      `json:"question"`
	RetrievedChunks []evalChunk `json:"retrieved_chunks"`
	AvgScore        float64     `json:"avg_score"`
	Answer          string      `json:"answer,omitempty"`
}

type evalChunk struct {
	Text  string  `json:"text"`
	Page  int     `json:"page"`
	Score float64 `json:"score"`
}

func handleDashboard(experimentsPath string) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/html; charset=utf-8")
		fmt.Fprint(w, dashboardHTML)
	}
}

func handleDashboardData(experimentsPath string) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		f, err := os.Open(experimentsPath)
		if err != nil {
			http.Error(w, "experiments file not found — run eval first", http.StatusNotFound)
			return
		}
		defer f.Close()

		var records []evalRecord
		scanner := bufio.NewScanner(f)
		scanner.Buffer(make([]byte, 0, 1024*1024), 4*1024*1024)
		for scanner.Scan() {
			var rec evalRecord
			if err := json.Unmarshal(scanner.Bytes(), &rec); err != nil {
				continue
			}
			records = append(records, rec)
		}

		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(records)
	}
}

const dashboardHTML = `<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1"/>
  <title>RAG Eval Dashboard</title>
  <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
  <style>
    *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
    body { font-family: system-ui, sans-serif; background: #f4f6f9; color: #1a1a2e; line-height: 1.5; }
    h1 { font-size: 1.5rem; font-weight: 700; }
    h2 { font-size: 1.1rem; font-weight: 600; margin-bottom: 1rem; color: #333; }

    header { background: #1a1a2e; color: #fff; padding: 1.25rem 2rem; display: flex; align-items: center; gap: 1.5rem; }
    header span { font-size: 0.85rem; color: #aab4c8; }

    .stats { display: flex; gap: 1rem; padding: 1.25rem 2rem; }
    .stat-card { background: #fff; border-radius: 10px; padding: 1rem 1.5rem; flex: 1; box-shadow: 0 1px 4px rgba(0,0,0,.08); }
    .stat-card .value { font-size: 2rem; font-weight: 700; color: #4361ee; }
    .stat-card .label { font-size: 0.8rem; color: #666; margin-top: 0.25rem; }

    .grid { display: grid; grid-template-columns: 1fr 1fr; gap: 1.25rem; padding: 0 2rem 1.25rem; }
    .card { background: #fff; border-radius: 10px; padding: 1.25rem; box-shadow: 0 1px 4px rgba(0,0,0,.08); }
    .card.full { grid-column: 1 / -1; }

    canvas { max-height: 300px; }

    /* Heatmap table */
    .heatmap-wrap { overflow-x: auto; }
    table { width: 100%; border-collapse: collapse; font-size: 0.82rem; }
    th { background: #f0f2f5; padding: 0.5rem 0.75rem; text-align: left; font-weight: 600; }
    td { padding: 0.45rem 0.75rem; border-top: 1px solid #eee; }
    td.score { text-align: center; font-weight: 600; border-radius: 4px; }
    tr:hover td { background: rgba(67,97,238,.04); }

    /* Explorer */
    .explorer-controls { display: flex; gap: 0.75rem; margin-bottom: 1rem; flex-wrap: wrap; }
    select { padding: 0.4rem 0.75rem; border: 1px solid #d0d5dd; border-radius: 6px; font-size: 0.85rem; background: #fff; }
    .answer-grid { display: grid; gap: 1rem; }
    .answer-card { border: 1px solid #e0e4ef; border-radius: 8px; padding: 1rem; }
    .answer-card .config-label { font-size: 0.75rem; font-weight: 700; color: #4361ee; text-transform: uppercase; letter-spacing: .05em; margin-bottom: 0.5rem; }
    .answer-card .answer-text { font-size: 0.85rem; white-space: pre-wrap; color: #333; }
    .answer-card .score-badge { display: inline-block; font-size: 0.72rem; background: #eef0fb; color: #4361ee; border-radius: 99px; padding: 0.1rem 0.5rem; margin-bottom: 0.5rem; }
    .chunks-toggle { font-size: 0.75rem; color: #4361ee; cursor: pointer; margin-top: 0.5rem; display: inline-block; }
    .chunks-list { display: none; margin-top: 0.5rem; }
    .chunks-list.open { display: block; }
    .chunk-item { font-size: 0.78rem; background: #f7f8fc; border-radius: 6px; padding: 0.5rem; margin-bottom: 0.4rem; border-left: 3px solid #4361ee; }
    .chunk-meta { font-size: 0.7rem; color: #888; margin-bottom: 0.2rem; }

    .empty { color: #999; font-size: 0.85rem; padding: 2rem; text-align: center; }
    #error-msg { color: #b00020; padding: 2rem; text-align: center; font-size: 0.9rem; }
  </style>
</head>
<body>

<header>
  <h1>RAG Eval Dashboard</h1>
  <span id="header-meta">Loading...</span>
</header>

<div class="stats">
  <div class="stat-card"><div class="value" id="stat-runs">—</div><div class="label">Total Eval Runs</div></div>
  <div class="stat-card"><div class="value" id="stat-configs">—</div><div class="label">Configs Tested</div></div>
  <div class="stat-card"><div class="value" id="stat-questions">—</div><div class="label">Questions</div></div>
  <div class="stat-card"><div class="value" id="stat-best">—</div><div class="label">Best Avg Score</div></div>
</div>

<div class="grid">
  <div class="card">
    <h2>Avg Retrieval Score by Config</h2>
    <canvas id="bar-chart"></canvas>
  </div>

  <div class="card">
    <h2>Score Distribution by Config</h2>
    <canvas id="box-chart"></canvas>
  </div>

  <div class="card full">
    <h2>Score Heatmap — Question × Config</h2>
    <div class="heatmap-wrap" id="heatmap"></div>
  </div>

  <div class="card full">
    <h2>Answer Explorer</h2>
    <div class="explorer-controls">
      <select id="q-select"><option>Loading questions...</option></select>
    </div>
    <div id="answer-grid" class="answer-grid"></div>
  </div>
</div>

<div id="error-msg" style="display:none"></div>

<script>
(async () => {
  let records;
  try {
    const res = await fetch('/dashboard/data');
    if (!res.ok) throw new Error(await res.text());
    records = await res.json();
  } catch (e) {
    document.getElementById('error-msg').style.display = 'block';
    document.getElementById('error-msg').textContent = 'Could not load data: ' + e.message;
    return;
  }

  if (!records || records.length === 0) {
    document.getElementById('error-msg').style.display = 'block';
    document.getElementById('error-msg').textContent = 'No experiment records found. Run the eval tool first.';
    return;
  }

  // ── Derive structure ──────────────────────────────────────────────────────
  const configKey  = r => r.chunk_words + 'w / k=' + r.top_k;
  const configSort = (a, b) => {
    const [aw, ak] = a.split('w / k=').map(Number);
    const [bw, bk] = b.split('w / k=').map(Number);
    return aw - bw || ak - bk;
  };

  const configs   = [...new Set(records.map(configKey))].sort(configSort);
  const questions = [...new Set(records.map(r => r.question))];

  // Map: question → config → record
  const matrix = {};
  for (const r of records) {
    const q = r.question, c = configKey(r);
    if (!matrix[q]) matrix[q] = {};
    matrix[q][c] = r;
  }

  // Avg score per config
  const avgByConfig = {};
  for (const c of configs) {
    const scores = records.filter(r => configKey(r) === c).map(r => r.avg_score);
    avgByConfig[c] = scores.reduce((a, b) => a + b, 0) / scores.length;
  }

  const bestConfig = configs.reduce((a, b) => avgByConfig[a] > avgByConfig[b] ? a : b);

  // ── Stats cards ───────────────────────────────────────────────────────────
  document.getElementById('stat-runs').textContent      = records.length;
  document.getElementById('stat-configs').textContent   = configs.length;
  document.getElementById('stat-questions').textContent = questions.length;
  document.getElementById('stat-best').textContent      = avgByConfig[bestConfig].toFixed(4);
  document.getElementById('header-meta').textContent    =
    configs.length + ' configs · ' + questions.length + ' questions · ' + records.length + ' records';

  // ── Colour helpers ────────────────────────────────────────────────────────
  const allScores = records.map(r => r.avg_score);
  const minS = Math.min(...allScores), maxS = Math.max(...allScores);

  function scoreColor(score, alpha = 1) {
    const t = maxS === minS ? 0.5 : (score - minS) / (maxS - minS);
    // red → yellow → green
    const r = t < 0.5 ? 255 : Math.round(255 * (1 - (t - 0.5) * 2));
    const g = t < 0.5 ? Math.round(255 * t * 2) : 255;
    return 'rgba(' + r + ',' + g + ',80,' + alpha + ')';
  }

  const CONFIG_COLORS = ['#4361ee', '#f72585', '#4cc9f0', '#7209b7', '#3a0ca3'];

  // ── Bar chart ─────────────────────────────────────────────────────────────
  new Chart(document.getElementById('bar-chart'), {
    type: 'bar',
    data: {
      labels: configs,
      datasets: [{
        label: 'Avg Score',
        data: configs.map(c => +avgByConfig[c].toFixed(5)),
        backgroundColor: configs.map((_, i) => CONFIG_COLORS[i % CONFIG_COLORS.length] + 'cc'),
        borderColor:     configs.map((_, i) => CONFIG_COLORS[i % CONFIG_COLORS.length]),
        borderWidth: 2,
        borderRadius: 6,
      }]
    },
    options: {
      plugins: { legend: { display: false } },
      scales: {
        y: {
          min: Math.max(0, minS - 0.02),
          max: Math.min(1, maxS + 0.02),
          ticks: { callback: v => v.toFixed(3) }
        }
      }
    }
  });

  // ── Box / violin approximation via scatter ────────────────────────────────
  // Use a scatter chart showing all individual question scores per config
  new Chart(document.getElementById('box-chart'), {
    type: 'scatter',
    data: {
      datasets: configs.map((c, i) => {
        const scores = records.filter(r => configKey(r) === c).map(r => r.avg_score);
        return {
          label: c,
          data: scores.map((s, j) => ({ x: i + 1 + (Math.random() - 0.5) * 0.25, y: s })),
          backgroundColor: CONFIG_COLORS[i % CONFIG_COLORS.length] + '99',
          pointRadius: 7,
          pointHoverRadius: 9,
        };
      })
    },
    options: {
      plugins: { tooltip: { callbacks: {
        label: ctx => ctx.dataset.label + ': ' + ctx.parsed.y.toFixed(4)
      }}},
      scales: {
        x: { min: 0.5, max: configs.length + 0.5, ticks: {
          stepSize: 1,
          callback: (_, i) => configs[i] || ''
        }},
        y: {
          min: Math.max(0, minS - 0.02),
          max: Math.min(1, maxS + 0.02),
          ticks: { callback: v => v.toFixed(3) }
        }
      }
    }
  });

  // ── Heatmap table ─────────────────────────────────────────────────────────
  const shortQ = q => q.length > 55 ? q.slice(0, 52) + '…' : q;

  let html = '<table><thead><tr><th>Question</th>';
  configs.forEach(c => { html += '<th style="text-align:center">' + c + '</th>'; });
  html += '</tr></thead><tbody>';

  for (const q of questions) {
    html += '<tr><td title="' + q + '">' + shortQ(q) + '</td>';
    for (const c of configs) {
      const rec = matrix[q]?.[c];
      if (rec) {
        const s = rec.avg_score;
        const bg = scoreColor(s, 0.55);
        html += '<td class="score" style="background:' + bg + '" title="' + s.toFixed(5) + '">' + s.toFixed(4) + '</td>';
      } else {
        html += '<td class="score" style="color:#ccc">—</td>';
      }
    }
    html += '</tr>';
  }
  html += '</tbody></table>';
  document.getElementById('heatmap').innerHTML = html;

  // ── Answer explorer ───────────────────────────────────────────────────────
  const qSelect = document.getElementById('q-select');
  qSelect.innerHTML = questions.map((q, i) =>
    '<option value="' + i + '">' + shortQ(q) + '</option>'
  ).join('');

  function renderExplorer(qIdx) {
    const q = questions[qIdx];
    const grid = document.getElementById('answer-grid');
    grid.style.gridTemplateColumns = 'repeat(' + Math.min(configs.length, 3) + ', 1fr)';

    grid.innerHTML = configs.map((c, ci) => {
      const rec = matrix[q]?.[c];
      if (!rec) return '<div class="answer-card"><div class="config-label">' + c + '</div><div class="empty">No data</div></div>';

      const chunksId = 'chunks-' + ci;
      const chunksHTML = rec.retrieved_chunks.map((ch, k) =>
        '<div class="chunk-item"><div class="chunk-meta">Page ' + ch.page + ' · score ' + ch.score.toFixed(4) + '</div>' +
        (ch.text.length > 200 ? ch.text.slice(0, 197) + '…' : ch.text) + '</div>'
      ).join('');

      return '<div class="answer-card">' +
        '<div class="config-label">' + c + '</div>' +
        '<span class="score-badge">avg score ' + rec.avg_score.toFixed(4) + '</span>' +
        '<div class="answer-text">' + (rec.answer || '<em style="color:#aaa">No LLM answer recorded</em>') + '</div>' +
        '<span class="chunks-toggle" onclick="toggleChunks(\'' + chunksId + '\', this)">▶ Show ' + rec.retrieved_chunks.length + ' retrieved chunks</span>' +
        '<div class="chunks-list" id="' + chunksId + '">' + chunksHTML + '</div>' +
        '</div>';
    }).join('');
  }

  window.toggleChunks = function(id, el) {
    const el2 = document.getElementById(id);
    el2.classList.toggle('open');
    el.textContent = el2.classList.contains('open')
      ? '▼ Hide chunks'
      : '▶ Show ' + el2.querySelectorAll('.chunk-item').length + ' retrieved chunks';
  };

  qSelect.addEventListener('change', () => renderExplorer(+qSelect.value));
  renderExplorer(0);
})();
</script>
</body>
</html>`
