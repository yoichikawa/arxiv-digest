#!/usr/bin/env python3
"""
arXiv 最新AI論文ダイジェスト
- arXiv APIから最新AI/ML論文を取得
- 送信済み論文を除外し、毎回PAPERS_PER_RUN件ずつ追加
- docs/papers.json に蓄積、docs/index.html を再生成
- 新規追加論文をSlackに通知
"""

import json
import os
import re
import sys
import time
import xml.etree.ElementTree as ET
from datetime import datetime, timezone, timedelta
from pathlib import Path

import requests
from google import genai

JST = timezone(timedelta(hours=9))
GEMINI_MODEL = "gemini-3.8-flash"
PAPERS_PER_RUN = 30
DOCS_DIR = Path("docs")
PAPERS_JSON = DOCS_DIR / "papers.json"
SS_QUERIES = [
    "deep learning neural network",
    "large language model transformer",
    "machine learning optimization",
    "computer vision image recognition",
    "natural language processing",
    "reinforcement learning policy",
    "generative model diffusion",
    "graph neural network",
]

# ── Fetch ─────────────────────────────────────────────────────────────────────────────────

def fetch_candidates(ss_api_key: str = "") -> list[dict]:
    """Semantic Scholar 検索APIで高影響力論文を取得し、引用数順で返す。
    API key があればそれを使う。なければ非認証で試みる（レート制限あり）。"""
    headers = {"x-api-key": ss_api_key} if ss_api_key else {}
    url = "https://api.semanticscholar.org/graph/v1/paper/search"
    fields = "paperId,title,abstract,citationCount,influentialCitationCount,year,publicationDate,externalIds"

    seen = set()
    papers = []

    for query in SS_QUERIES:
        params = {
            "query": query,
            "fields": fields,
            "fieldsOfStudy": "Computer Science",
            "limit": 100,
        }
        for attempt in range(4):
            try:
                print(f"  Querying SS: {query!r} (attempt {attempt+1})...")
                resp = requests.get(url, params=params, headers=headers, timeout=20)
                if resp.status_code == 429:
                    wait = 20 * (attempt + 1)
                    print(f"  Rate limited, waiting {wait}s...")
                    time.sleep(wait)
                    continue
                resp.raise_for_status()
                for p in resp.json().get("data", []):
                    pid = p.get("paperId", "")
                    if not pid or pid in seen:
                        continue
                    seen.add(pid)
                    arxiv_id = (p.get("externalIds") or {}).get("ArXiv", "")
                    papers.append({
                        "paperId": pid,
                        "arxivId": arxiv_id,
                        "title": p.get("title", ""),
                        "year": p.get("year", ""),
                        "published": p.get("publicationDate") or str(p.get("year", "")),
                        "categories": [],
                        "citationCount": p.get("citationCount") or 0,
                        "influentialCitationCount": p.get("influentialCitationCount") or 0,
                        "abstract": (p.get("abstract") or "")[:500],
                        "url": "https://arxiv.org/abs/" + arxiv_id if arxiv_id
                               else "https://www.semanticscholar.org/paper/" + pid,
                    })
                break
            except Exception as e:
                if attempt == 3:
                    print(f"  Warning ({query!r}): {e}")
                else:
                    time.sleep(5 * (attempt + 1))
        time.sleep(1)  # polite delay

    papers.sort(
        key=lambda p: (p.get("influentialCitationCount", 0), p.get("citationCount", 0)),
        reverse=True,
    )
    top = papers[0] if papers else {}
    print(f"  Total candidates: {len(papers)}, top influential={top.get('influentialCitationCount', 0)}: {top.get('title','')[:50]}")
    return papers

def load_papers() -> list[dict]:
    if PAPERS_JSON.exists():
        return json.loads(PAPERS_JSON.read_text(encoding="utf-8-sig"))
    return []

def save_papers(papers: list[dict]) -> None:
    DOCS_DIR.mkdir(exist_ok=True)
    PAPERS_JSON.write_text(
        json.dumps(papers, ensure_ascii=False, indent=2), encoding="utf-8"
    )

def pick_new(candidates: list[dict], sent_ids: set) -> list[dict]:
    result = []
    for p in candidates:
        uid = p.get("paperId", "")
        if uid in sent_ids:
            continue
        result.append(p)
        if len(result) >= PAPERS_PER_RUN:
            break
    return result

def _fetch_abstract(paper_id: str, ss_api_key: str = "") -> str:
    """Semantic Scholar から abstract を再取得する（要約なし論文の修復用）。"""
    if not paper_id:
        return ""
    headers = {"x-api-key": ss_api_key} if ss_api_key else {}
    try:
        resp = requests.get(
            f"https://api.semanticscholar.org/graph/v1/paper/{paper_id}",
            params={"fields": "abstract"},
            headers=headers,
            timeout=10,
        )
        resp.raise_for_status()
        return resp.json().get("abstract") or ""
    except Exception as e:
        print(f"  Failed to fetch abstract for {paper_id}: {e}")
        return ""

# ── Summarize ─────────────────────────────────────────────────────────────────────────────

SUMMARIZE_BATCH = 10  # Gemini に一度に送る論文数

class GeminiQuotaExceeded(Exception):
    pass

def _summarize_batch(client, papers: list[dict]) -> list[dict]:
    parts = [
        f"[{i}] {p['title']} ({p['year']})\n{p['abstract']}"
        for i, p in enumerate(papers)
    ]
    prompt = (
        "以下のAI・機械学習論文を、日本語でわかりやすく解説してください。\n"
        "対象読者はAIや機械学習をあまり知らない初心者です。\n\n"
        "各論文について以下を返してください：\n"
        "1. summary: 4～6文程度の解説（背景・手法・成果）\n"
        "   - 「歴史的な論文」「分野を変えた」などの評価コメントは書かない\n"
        "   - 初めて読む人がイメージできる具体的な表現を使う\n"
        "2. terms: 論文中の重要な専門用語を3～5個ピックアップし、平易な日本語で1文ずつ解説\n"
        "3. formulas: 論文中に登場する数式・記号を2～4個抽出し、それぞれ expression（LaTeX文字列）と meaning（1〜2文の平易な説明）をリストで返す。数式がなければ空リスト []\n\n"
        "必ず以下のJSON形式のみで回答:\n"
        '{"0": {"summary": "解説文", "terms": {"用語1": "説明"}, "formulas": [{"expression": "数式", "meaning": "説明"}]}, "1": {...}, ...}\n\n'
        + "\n\n".join(parts)
    )
    empty: dict = {"summary": "", "terms": {}, "formulas": []}
    for attempt in range(3):
        try:
            resp = client.models.generate_content(model=GEMINI_MODEL, contents=prompt)
            raw = resp.text.strip()
            m = re.search(r'\{[\s\S]*\}', raw)
            if m:
                data = json.loads(m.group())
                return [data.get(str(i), empty) for i in range(len(papers))]
        except Exception as e:
            err = str(e).lower()
            if any(kw in err for kw in ("quota", "resource_exhausted", "429", "rate limit")):
                raise GeminiQuotaExceeded(str(e))
            if attempt == 2:
                print(f"  Gemini error: {e}")
                return [empty] * len(papers)
            time.sleep(30 * (attempt + 1))
    return [empty] * len(papers)

def summarize(api_key: str, papers: list[dict]) -> list[dict]:
    client = genai.Client(api_key=api_key)
    results = []
    for i in range(0, len(papers), SUMMARIZE_BATCH):
        batch = papers[i:i + SUMMARIZE_BATCH]
        print(f"  Summarizing batch {i // SUMMARIZE_BATCH + 1} ({len(batch)} papers)...")
        results.extend(_summarize_batch(client, batch))
        if i + SUMMARIZE_BATCH < len(papers):
            time.sleep(5)
    return results

# ── HTML ─────────────────────────────────────────────────────────────────────────────────────

def generate_index_html(all_papers: list[dict]) -> str:
    count = len(all_papers)
    updated = datetime.now(JST).strftime("%Y-%m-%d %H:%M JST")
    papers_json = json.dumps(all_papers, ensure_ascii=False).replace('</', '<\\/')
    return f"""<!doctype html>
<html lang="ja">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">
<title>AI論文まとめ</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Noto+Sans+JP:wght@400;500;700&display=swap" media="print" onload="this.media='all'">
<noscript><link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Noto+Sans+JP:wght@400;500;700&display=swap"></noscript>
<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/KaTeX/0.16.9/katex.min.css">
<script src="https://cdnjs.cloudflare.com/ajax/libs/KaTeX/0.16.9/katex.min.js"></script>
<style>
:root {{
  --bg:#fff;--bg2:#f7f8fa;--fg:#1a1a1a;--fg2:#555;--fg3:#888;
  --border:#e0e3e8;--accent:#3563e9;--accent-light:#e8eeff;--link:#3563e9;
}}
@media (prefers-color-scheme:dark) {{
  :root:not([data-theme=light]) {{
    --bg:#0f1117;--bg2:#181c25;--fg:#e8eaf0;--fg2:#a0a8b8;--fg3:#6b7280;
    --border:#2a2f3e;--accent:#6b9aff;--accent-light:#1a2340;--link:#6b9aff;
    color-scheme:dark;
  }}
}}
:root[data-theme=dark] {{
  --bg:#0f1117;--bg2:#181c25;--fg:#e8eaf0;--fg2:#a0a8b8;--fg3:#6b7280;
  --border:#2a2f3e;--accent:#6b9aff;--accent-light:#1a2340;--link:#6b9aff;
  color-scheme:dark;
}}
*,*::before,*::after{{box-sizing:border-box}}
body{{
  background:var(--bg);color:var(--fg);
  font-family:'Noto Sans JP',sans-serif;font-size:15px;line-height:1.8;
  margin:0;
  padding-left:max(16px,env(safe-area-inset-left,0px));
  padding-right:max(16px,env(safe-area-inset-right,0px));
  padding-block:0 64px;
}}
.container{{max-width:760px;margin:0 auto}}
.site-header{{padding-block:40px 24px;border-bottom:1px solid var(--border);margin-bottom:24px}}
.site-header h1{{margin:0 0 4px;font-size:24px;font-weight:700}}
.site-header p{{margin:0;font-size:13px;color:var(--fg3)}}
.search-wrap{{margin-bottom:16px}}
#searchInput{{
  width:100%;padding:10px 14px;font-size:15px;
  border:1px solid var(--border);border-radius:8px;
  background:var(--bg2);color:var(--fg);outline:none;font-family:inherit;
}}
#searchInput:focus{{border-color:var(--accent);box-shadow:0 0 0 3px var(--accent-light)}}
.tabs{{display:flex;flex-wrap:wrap;gap:8px;margin-bottom:20px}}
.tab{{
  padding:6px 14px;border-radius:20px;border:1px solid var(--border);
  background:var(--bg2);color:var(--fg2);font-size:13px;font-weight:500;
  cursor:pointer;font-family:inherit;transition:background .15s,color .15s;
}}
.tab:hover,.tab.active{{background:var(--accent);color:#fff;border-color:var(--accent)}}
.top-bar{{
  display:flex;justify-content:space-between;align-items:center;
  flex-wrap:wrap;gap:8px;margin-bottom:12px;font-size:13px;color:var(--fg3);
}}
.top-bar strong{{color:var(--fg)}}
.pagination{{display:flex;flex-wrap:wrap;gap:6px;align-items:center}}
.page-btn{{
  min-width:36px;height:36px;padding:0 10px;border-radius:6px;
  border:1px solid var(--border);background:var(--bg2);color:var(--fg2);
  font-size:13px;cursor:pointer;font-family:inherit;
}}
.page-btn:hover,.page-btn.active{{background:var(--accent);color:#fff;border-color:var(--accent)}}
.bottom-bar{{display:flex;justify-content:flex-end;margin-top:20px}}
.card{{border:1px solid var(--border);border-radius:10px;padding:20px 24px;margin-bottom:16px;background:var(--bg)}}
.card-meta{{display:flex;flex-wrap:wrap;gap:10px;font-size:12px;color:var(--fg3);margin-bottom:8px}}
.date{{color:var(--accent);font-weight:700}}
.card-cat{{background:var(--accent-light);color:var(--accent);padding:1px 8px;border-radius:20px;font-size:11px;font-weight:700}}
.card-title{{margin:0 0 10px;font-size:16px;font-weight:700;line-height:1.4}}
.card-title a{{color:var(--link);text-decoration:none}}
.card-title a:hover{{text-decoration:underline}}
.card-summary{{margin:0;font-size:14px;color:var(--fg2);line-height:1.7}}
.card-terms{{margin-top:12px;padding-top:12px;border-top:1px dashed var(--border)}}
.card-terms-label{{font-size:11px;font-weight:700;color:var(--fg3);text-transform:uppercase;letter-spacing:.05em;margin-bottom:6px}}
.card-terms dl{{margin:0;display:flex;flex-direction:column;gap:3px}}
.card-terms dt{{font-size:12px;font-weight:700;color:var(--accent);display:inline}}
.card-terms dd{{font-size:12px;color:var(--fg2);margin:0;display:inline}}
.card-formulas{{margin-top:12px;padding-top:12px;border-top:1px dashed var(--border)}}
.card-formulas-label{{font-size:11px;font-weight:700;color:var(--fg3);text-transform:uppercase;letter-spacing:.05em;margin-bottom:8px}}
.formula-item{{margin-bottom:10px;padding:8px 10px;background:var(--bg2);border-radius:6px;border-left:3px solid var(--accent)}}
.formula-expr{{display:block;margin-bottom:4px;overflow-x:auto;padding:2px 0}}
.formula-expr .katex{{font-size:1.05em}}
.formula-expr-fallback{{font-size:12px;font-family:monospace;color:var(--accent);font-weight:700}}
.formula-meaning{{font-size:12px;color:var(--fg2);line-height:1.5}}
.no-results{{text-align:center;padding:64px 0;color:var(--fg3)}}
.loading{{text-align:center;padding:40px 0;color:var(--fg3)}}
.sort-bar{{display:flex;gap:8px;align-items:center;margin-bottom:12px;font-size:13px;color:var(--fg3)}}
.sort-btn{{
  padding:5px 12px;border-radius:20px;border:1px solid var(--border);
  background:var(--bg2);color:var(--fg2);font-size:12px;font-weight:500;
  cursor:pointer;font-family:inherit;transition:background .15s,color .15s;
}}
.sort-btn:hover,.sort-btn.active{{background:var(--accent);color:#fff;border-color:var(--accent)}}
.read-filter-bar{{display:flex;gap:8px;align-items:center;margin-bottom:12px;font-size:13px;color:var(--fg3)}}
.read-filter-btn{{
  padding:5px 12px;border-radius:20px;border:1px solid var(--border);
  background:var(--bg2);color:var(--fg2);font-size:12px;font-weight:500;
  cursor:pointer;font-family:inherit;transition:background .15s,color .15s;
}}
.read-filter-btn:hover,.read-filter-btn.active{{background:var(--accent);color:#fff;border-color:var(--accent)}}
.card.is-read{{opacity:0.6}}
.card.is-read .card-title a{{text-decoration:line-through}}
.read-check-wrap{{display:flex;align-items:center;gap:6px;margin-top:10px;padding-top:10px;border-top:1px solid var(--border)}}
.read-check{{width:16px;height:16px;cursor:pointer;accent-color:var(--accent)}}
.read-check-label{{font-size:12px;color:var(--fg3);cursor:pointer;user-select:none}}
</style>
</head>
<body>
<div class="container">
  <header class="site-header">
    <h1>AI論文まとめ</h1>
    <p>arXivの最新AI・機械学習論文の日本語要約集 &nbsp;·&nbsp; 全 {count} 件 &nbsp;·&nbsp; 最終更新 {updated}</p>
  </header>
  <div class="search-wrap">
    <input type="search" id="searchInput" placeholder="タイトル・要約を検索…" autocomplete="off">
  </div>
  <div class="tabs" role="tablist">
    <button class="tab active" onclick="setTab(this,'all')">すべて</button>
    <button class="tab" onclick="setTab(this,'cs.AI')">🤖 AI・推論</button>
    <button class="tab" onclick="setTab(this,'cs.LG')">📊 機械学習</button>
    <button class="tab" onclick="setTab(this,'cs.CL')">💬 自然言語処理</button>
    <button class="tab" onclick="setTab(this,'cs.CV')">🖼 画像・Vision</button>
  </div>
  <div class="sort-bar">
    <span>並び替え:</span>
    <button class="sort-btn active" onclick="setSort(this,'date')">📅 投稿日順</button>
    <button class="sort-btn" onclick="setSort(this,'cite')">📈 引用が多い順</button>
  </div>
  <div class="read-filter-bar">
    <span>閲覧状態:</span>
    <button class="read-filter-btn active" onclick="setReadFilter(this,'all')">すべて</button>
    <button class="read-filter-btn" onclick="setReadFilter(this,'read')">✅ 読了済み</button>
    <button class="read-filter-btn" onclick="setReadFilter(this,'unread')">📄 未読</button>
  </div>
  <div class="top-bar">
    <span id="statsText"></span>
    <div class="pagination" id="paginTop"></div>
  </div>
  <div id="cards"><div class="loading">読み込み中…</div></div>
  <div class="bottom-bar">
    <div class="pagination" id="paginBottom"></div>
  </div>
</div>
<script>
(function() {{
  var PER_PAGE = 10, page = 1, cat = 'all', q = '', sortOrder = 'date', readFilter = 'all', PAPERS = [];
  var readPapers = (function() {{
    try {{ return new Set(JSON.parse(localStorage.getItem('arxiv_read') || '[]')); }}
    catch(e) {{ return new Set(); }}
  }})();
  function saveRead() {{
    try {{ localStorage.setItem('arxiv_read', JSON.stringify(Array.from(readPapers))); }} catch(e) {{}}
  }}
  var CAT_LABELS = {{'cs.AI':'AI・推論','cs.LG':'機械学習','cs.CL':'自然言語処理','cs.CV':'画像・Vision'}};
  function getCat(p) {{
    var c = (p.categories || [])[0];
    if (c) return c;
    var t = ((p.title||'') + ' ' + (p.summary||'')).toLowerCase();
    if (/language model|llm|nlp|translation|tokeniz|speech|chat|dialogue|prompt/.test(t)) return 'cs.CL';
    if (/image|vision|visual|pixel|video|3d|segmentation|detection|recognition|diffusion/.test(t)) return 'cs.CV';
    if (/reinforcement|reward|policy|agent|planning|reasoning|logic|knowledge/.test(t)) return 'cs.AI';
    return 'cs.LG';
  }}
  function esc(s) {{
    return String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
  }}
  function renderInline(text) {{
    if (typeof katex === 'undefined') return esc(text);
    return String(text).split(/(\$[^$]+\$)/g).map(function(part) {{
      if (part.charAt(0) === '$' && part.charAt(part.length-1) === '$') {{
        try {{ return katex.renderToString(part.slice(1,-1), {{throwOnError: false, displayMode: false}}); }} catch(e) {{ return esc(part); }}
      }}
      return esc(part);
    }}).join('');
  }}
  function filtered() {{
    return PAPERS.filter(function(p) {{
      if (cat !== 'all' && getCat(p) !== cat) return false;
      if (q && ((p.title||'') + ' ' + (p.summary||'')).toLowerCase().indexOf(q) < 0) return false;
      var isRead = readPapers.has(p.paperId||'');
      if (readFilter === 'read' && !isRead) return false;
      if (readFilter === 'unread' && isRead) return false;
      return true;
    }});
  }}
  function sorted(arr) {{
    var a = arr.slice();
    if (sortOrder === 'cite') {{
      a.sort(function(x, y) {{ return (y.citationCount||0) - (x.citationCount||0); }});
    }} else {{
      a.sort(function(x, y) {{
        var da = x.published||'', db = y.published||'';
        if (db !== da) return db > da ? 1 : -1;
        var ea = x.addedDate||'', eb = y.addedDate||'';
        return eb > ea ? 1 : eb < ea ? -1 : 0;
      }});
    }}
    return a;
  }}
  function renderPagin(total, id) {{
    var el = document.getElementById(id);
    var pages = Math.ceil(total / PER_PAGE);
    if (pages <= 1) {{ el.innerHTML = ''; return; }}
    var h = '';
    if (page > 1) h += '<button class="page-btn" onclick="go(' + (page-1) + ')">← 前へ</button>';
    for (var i = 1; i <= pages; i++) {{
      if (i === 1 || i === pages || Math.abs(i-page) <= 2)
        h += '<button class="page-btn' + (i===page?' active':'') + '" onclick="go(' + i + ')">' + i + '</button>';
      else if (Math.abs(i-page) === 3)
        h += '<span style="color:var(--fg3);padding:0 4px">…</span>';
    }}
    if (page < pages) h += '<button class="page-btn" onclick="go(' + (page+1) + ')">次へ →</button>';
    el.innerHTML = h;
  }}
  function render() {{
    var f = sorted(filtered()), total = f.length;
    var slice = f.slice((page-1)*PER_PAGE, page*PER_PAGE);
    document.getElementById('statsText').innerHTML =
      '<strong>' + total + '</strong> 件 / 全 <strong>' + PAPERS.length + '</strong> 件';
    renderPagin(total,'paginTop');
    renderPagin(total,'paginBottom');
    if (!slice.length) {{
      document.getElementById('cards').innerHTML = '<div class="no-results">該当する論文が見つかりませんでした。</div>';
      return;
    }}
    var html = '';
    for (var j = 0; j < slice.length; j++) {{
      var p = slice[j], c = getCat(p), label = CAT_LABELS[c] || c;
      var terms = p.terms || {{}}, termKeys = Object.keys(terms), termsHtml = '';
      if (termKeys.length) {{
        termsHtml = '<div class="card-terms"><div class="card-terms-label">📖 用語解説</div><dl>' +
          termKeys.map(function(k) {{ return '<dt>' + esc(k) + '</dt><dd>: ' + esc(terms[k]) + '</dd>'; }}).join('') +
          '</dl></div>';
      }}
      var formulas = p.formulas || [], formulasHtml = '';
      if (formulas.length) {{
        formulasHtml = '<div class="card-formulas"><div class="card-formulas-label">📐 数式解説</div>' +
          formulas.map(function(f) {{
            var expr = f.expression || '';
            var rendered;
            try {{
              rendered = '<div class="formula-expr">' + katex.renderToString(expr, {{throwOnError: false, displayMode: true}}) + '</div>';
            }} catch(e) {{
              rendered = '<div class="formula-expr"><span class="formula-expr-fallback">' + esc(expr) + '</span></div>';
            }}
            return '<div class="formula-item">' + rendered +
              '<span class="formula-meaning">' + renderInline(f.meaning||'') + '</span></div>';
          }}).join('') + '</div>';
      }}
      var pid = p.paperId || '';
      var isRead = pid && readPapers.has(pid);
      var chkId = 'chk_' + pid.replace(/[^a-zA-Z0-9]/g,'_');
      var readCheckHtml = '<div class="read-check-wrap">' +
        '<input type="checkbox" class="read-check" id="' + chkId + '"' +
        (isRead ? ' checked' : '') +
        (pid ? ' data-pid="' + pid + '" onchange="toggleRead(this.dataset.pid,this)"' : ' disabled') + '>' +
        '<label class="read-check-label" for="' + chkId + '">読了済みにする</label>' +
        '</div>';
      html += '<article class="card' + (isRead ? ' is-read' : '') + '">' +
        '<div class="card-meta">' +
        '<span class="date">📅 ' + esc(p.published||p.year||'') + '</span>' +
        '<span class="card-cat">' + label + '</span>' +
        '</div>' +
        '<h2 class="card-title"><a href="' + esc(p.url||'#') + '" target="_blank" rel="noopener">' + esc(p.title||'') + '</a></h2>' +
        '<p class="card-summary">' + esc(p.summary||'（要約なし）') + '</p>' +
        termsHtml +
        formulasHtml +
        readCheckHtml +
        '</article>';
    }}
    document.getElementById('cards').innerHTML = html;
  }}
  window.go = function(n) {{ page = n; render(); window.scrollTo(0,0); }};
  window.setSort = function(btn, s) {{
    document.querySelectorAll('.sort-btn').forEach(function(t) {{ t.classList.remove('active'); }});
    btn.classList.add('active');
    sortOrder = s; page = 1; render();
  }};
  window.setReadFilter = function(btn, f) {{
    document.querySelectorAll('.read-filter-btn').forEach(function(t) {{ t.classList.remove('active'); }});
    btn.classList.add('active');
    readFilter = f; page = 1; render();
  }};
  window.toggleRead = function(paperId, cb) {{
    if (cb.checked) {{
      readPapers.add(paperId);
    }} else {{
      readPapers.delete(paperId);
    }}
    saveRead();
    var card = cb.closest('.card');
    if (card) {{
      if (cb.checked) {{ card.classList.add('is-read'); }} else {{ card.classList.remove('is-read'); }}
    }}
    if (readFilter !== 'all') {{ page = 1; render(); }}
  }};
  window.setTab = function(btn, c) {{
    document.querySelectorAll('.tab').forEach(function(t) {{ t.classList.remove('active'); }});
    btn.classList.add('active');
    cat = c; page = 1; render();
  }};
  document.getElementById('searchInput').addEventListener('input', function(e) {{
    q = e.target.value.toLowerCase().trim(); page = 1; render();
  }});
  PAPERS = {papers_json};
  render();
}})();
</script>
</body>
</html>"""

# ── Main ──────────────────────────────────────────────────────────────────────────────────

def main():
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        sys.exit("Error: GEMINI_API_KEY is not set.")

    ss_api_key = os.environ.get("SEMANTIC_SCHOLAR_API_KEY", "")
    print("Fetching Semantic Scholar candidates...")
    candidates = fetch_candidates(ss_api_key)

    existing = load_papers()
    sent_ids = {p.get("paperId", "") for p in existing}

    new_papers = pick_new(candidates, sent_ids)
    if not new_papers:
        print("No new papers to add.")
    else:
        print(f"Summarizing {len(new_papers)} new papers...")
        try:
            summaries = summarize(api_key, new_papers)
        except GeminiQuotaExceeded as e:
            print(f"Gemini quota exceeded, skipping new papers: {e}")
            new_papers = []
        else:
            today = datetime.now(JST).strftime("%Y-%m-%d")
            for p, s in zip(new_papers, summaries):
                p["summary"] = s.get("summary", "") if isinstance(s, dict) else s
                p["terms"] = s.get("terms", {}) if isinstance(s, dict) else {}
                p["formulas"] = s.get("formulas", []) if isinstance(s, dict) else []
                p["addedDate"] = today
                del p["abstract"]

    all_papers = existing + new_papers

    # Re-summarize existing papers with missing summaries (up to 5 per run)
    unsummarized = [p for p in all_papers if not p.get("summary")][:5]
    if unsummarized:
        print(f"Re-fetching abstracts for {len(unsummarized)} papers missing summaries...")
        for p in unsummarized:
            abstract = _fetch_abstract(p.get("paperId", ""), ss_api_key)
            if abstract:
                p["abstract"] = abstract
        with_abstract = [p for p in unsummarized if p.get("abstract")]
        if with_abstract:
            print(f"Re-summarizing {len(with_abstract)} papers...")
            try:
                re_summaries = summarize(api_key, with_abstract)
                for p, s in zip(with_abstract, re_summaries):
                    p["summary"] = s.get("summary", "") if isinstance(s, dict) else s
                    p["terms"] = s.get("terms", {}) if isinstance(s, dict) else {}
                    p["formulas"] = s.get("formulas", []) if isinstance(s, dict) else []
                print(f"Re-summarized {len(with_abstract)} papers successfully.")
            except GeminiQuotaExceeded:
                print("Quota exceeded during re-summarization, will retry next run.")
            finally:
                for p in with_abstract:
                    p.pop("abstract", None)

    before = len(all_papers)
    all_papers = [p for p in all_papers if p.get("summary")]
    removed = before - len(all_papers)
    if removed:
        print(f"Excluded {removed} papers without summaries.")

    save_papers(all_papers)
    print(f"papers.json updated: {len(all_papers)} papers total.")

    html = generate_index_html(all_papers)
    (DOCS_DIR / "index.html").write_text(html, encoding="utf-8")
    print("docs/index.html generated.")


if __name__ == "__main__":
    main()
