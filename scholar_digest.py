#!/usr/bin/env python3
"""
Semantic Scholar 高影響力論文ダイジェスト
- Semantic Scholar APIからAI分野の高影響力論文を取得
- 送信済み論文を除外し、毎回PAPERS_PER_RUN件ずつ追加
- docs/papers.json に蓄積、docs/index.html を再生成
- 新規追加論文をSlackに通知
"""

import json
import os
import re
import sys
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path

import requests
from google import genai

JST = timezone(timedelta(hours=9))
GEMINI_MODEL = "gemini-3.8-flash"
PAPERS_PER_RUN = 30
DOCS_DIR = Path("docs")
PAPERS_JSON = DOCS_DIR / "papers.json"

QUERIES = [
    "large language model",
    "deep learning transformer",
    "diffusion model generative",
    "reinforcement learning",
    "computer vision neural network",
]


# ── Fetch ─────────────────────────────────────────────────────────────────────

def fetch_candidates() -> list[dict]:
    """複数クエリでSemantic ScholarからAI論文を取得し、影響力スコア順で返す。"""
    seen = set()
    papers = []
    for query in QUERIES:
        print(f"  Querying: {query}")
        url = "https://api.semanticscholar.org/graph/v1/paper/search"
        params = {
            "query": query,
            "fields": "paperId,title,abstract,citationCount,influentialCitationCount,year,externalIds",
            "fieldsOfStudy": "Computer Science",
            "limit": 100,
        }
        for attempt in range(4):
            try:
                resp = requests.get(url, params=params, timeout=20)
                if resp.status_code == 429:
                    wait = 15 * (attempt + 1)
                    print(f"    Rate limited, waiting {wait}s...")
                    time.sleep(wait)
                    continue
                resp.raise_for_status()
                for p in resp.json().get("data", []):
                    pid = p.get("paperId", "")
                    if pid and pid not in seen:
                        seen.add(pid)
                        papers.append(p)
                break
            except Exception as e:
                print(f"    Warning: {e}")
                break
        time.sleep(3)  # rate limit

    papers.sort(key=lambda p: p.get("influentialCitationCount", 0), reverse=True)
    print(f"  Total candidates: {len(papers)}")
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
        if p.get("paperId") in sent_ids:
            continue
        arxiv_id = (p.get("externalIds") or {}).get("ArXiv", "")
        result.append({
            "paperId": p["paperId"],
            "arxivId": arxiv_id,
            "title": p.get("title", ""),
            "year": p.get("year", ""),
            "citationCount": p.get("citationCount", 0),
            "influentialCitationCount": p.get("influentialCitationCount", 0),
            "abstract": (p.get("abstract") or "")[:500],
            "url": f"https://arxiv.org/abs/{arxiv_id}" if arxiv_id else
                   f"https://www.semanticscholar.org/paper/{p['paperId']}",
        })
        if len(result) >= PAPERS_PER_RUN:
            break
    return result


# ── Summarize ─────────────────────────────────────────────────────────────────

SUMMARIZE_BATCH = 10  # Gemini に一度に送る論文数


def _summarize_batch(client, papers: list[dict]) -> list[str]:
    """papers（最大SUMMARIZE_BATCH件）をGeminiで要約して返す。"""
    parts = [
        f"[{i}] {p['title']} ({p['year']})\n{p['abstract']}"
        for i, p in enumerate(papers)
    ]
    prompt = (
        "以下のAI・機械学習論文を、日本語でわかりやすく解説してください。\n"
        "対象読者はAIや機械学習をあまり知らない初心者です。\n\n"
        "各論文について、以下の点を含めて4〜6文程度で解説してください：\n"
        "1. どんな課題・問題を解こうとしているか（背景）\n"
        "2. どんな手法やアルゴリズムを使っているか（内容）\n"
        "3. 実験でどんな結果が得られたか（成果）\n\n"
        "注意事項：\n"
        "- 「歴史的な論文」「分野を変えた」「金字塔」などの評価コメントは一切書かない\n"
        "- 論文の社会的意義や影響の説明も不要\n"
        "- 専門用語は必ずかっこ内で平易に説明する（例：CNN（画像を格子状に処理するニューラルネットワーク）、勾配降下法（損失が小さくなる方向にパラメータを少しずつ更新する手法））\n"
        "- 初めて読む人が「何をしている論文か」をイメージできる具体的な表現を使う\n\n"
        "必ず以下のJSON形式のみで回答:\n"
        '{"0": "解説文", "1": "解説文", ...}\n\n'
        + "\n\n".join(parts)
    )
    for attempt in range(3):
        try:
            resp = client.models.generate_content(model=GEMINI_MODEL, contents=prompt)
            raw = resp.text.strip()
            m = re.search(r'\{[\s\S]*\}', raw)
            if m:
                data = json.loads(m.group())
                return [data.get(str(i), "") for i in range(len(papers))]
        except Exception as e:
            if attempt == 2:
                print(f"  Gemini error: {e}")
                return [""] * len(papers)
            time.sleep(30 * (attempt + 1))
    return [""] * len(papers)


def summarize(api_key: str, papers: list[dict]) -> list[str]:
    """全論文をSUMMARIZE_BATCH件ずつバッチ処理して要約する。"""
    client = genai.Client(api_key=api_key)
    results = []
    for i in range(0, len(papers), SUMMARIZE_BATCH):
        batch = papers[i:i + SUMMARIZE_BATCH]
        print(f"  Summarizing batch {i // SUMMARIZE_BATCH + 1} ({len(batch)} papers)...")
        results.extend(_summarize_batch(client, batch))
        if i + SUMMARIZE_BATCH < len(papers):
            time.sleep(5)  # バッチ間のレート制限対策
    return results


# ── HTML ──────────────────────────────────────────────────────────────────────

def generate_index_html(all_papers: list[dict]) -> str:
    sorted_papers = sorted(
        all_papers,
        key=lambda p: p.get("influentialCitationCount", 0),
        reverse=True,
    )

    cards = ""
    for p in sorted_papers:
        summary = p.get("summary", "").replace("<", "&lt;").replace(">", "&gt;")
        title = p.get("title", "").replace("<", "&lt;").replace(">", "&gt;")
        cards += f"""
    <article class="card">
      <div class="card-meta">
        <span class="score">★ {p.get('influentialCitationCount', 0)} influential</span>
        <span class="cite">引用 {p.get('citationCount', 0)}</span>
        <span class="year">{p.get('year', '')}</span>
        <span class="added">追加: {p.get('addedDate', '')}</span>
      </div>
      <h2 class="card-title">
        <a href="{p.get('url','#')}" target="_blank" rel="noopener">{title}</a>
      </h2>
      <p class="card-summary">{summary if summary else '（要約なし）'}</p>
    </article>"""

    count = len(all_papers)
    updated = datetime.now(JST).strftime("%Y-%m-%d %H:%M JST")

    return f"""<!doctype html>
<html lang="ja">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">
<title>AI論文まとめ</title>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Noto+Sans+JP:wght@400;500;700&display=swap">
<style>
:root {{
  --bg: #ffffff; --bg2: #f7f8fa; --fg: #1a1a1a; --fg2: #555; --fg3: #888;
  --border: #e0e3e8; --accent: #3563e9; --accent-light: #e8eeff;
  --link: #3563e9;
}}
@media (prefers-color-scheme: dark) {{
  :root:not([data-theme="light"]) {{
    --bg: #0f1117; --bg2: #181c25; --fg: #e8eaf0; --fg2: #a0a8b8; --fg3: #6b7280;
    --border: #2a2f3e; --accent: #6b9aff; --accent-light: #1a2340; --link: #6b9aff;
    color-scheme: dark;
  }}
}}
:root[data-theme="dark"] {{
  --bg: #0f1117; --bg2: #181c25; --fg: #e8eaf0; --fg2: #a0a8b8; --fg3: #6b7280;
  --border: #2a2f3e; --accent: #6b9aff; --accent-light: #1a2340; --link: #6b9aff;
  color-scheme: dark;
}}
*, *::before, *::after {{ box-sizing: border-box; }}
body {{
  background: var(--bg); color: var(--fg);
  font-family: 'Noto Sans JP', sans-serif; font-size: 15px; line-height: 1.8;
  margin: 0;
  padding-left: max(16px, env(safe-area-inset-left, 0px));
  padding-right: max(16px, env(safe-area-inset-right, 0px));
  padding-block: 0 64px;
}}
.container {{ max-width: 760px; margin: 0 auto; }}
.site-header {{
  padding-block: 40px 24px;
  border-bottom: 1px solid var(--border);
  margin-bottom: 32px;
}}
.site-header h1 {{ margin: 0 0 8px; font-size: 24px; font-weight: 700; }}
.site-header p {{ margin: 0; font-size: 13px; color: var(--fg3); }}
.stats {{
  display: flex; gap: 16px; flex-wrap: wrap;
  margin-bottom: 28px; font-size: 13px; color: var(--fg3);
}}
.stats strong {{ color: var(--fg); }}
.card {{
  border: 1px solid var(--border); border-radius: 10px;
  padding: 20px 24px; margin-bottom: 16px;
  background: var(--bg);
}}
.card-meta {{
  display: flex; flex-wrap: wrap; gap: 10px;
  font-size: 12px; color: var(--fg3); margin-bottom: 8px;
}}
.score {{ color: var(--accent); font-weight: 700; }}
.card-title {{ margin: 0 0 10px; font-size: 16px; font-weight: 700; line-height: 1.4; }}
.card-title a {{ color: var(--link); text-decoration: none; }}
.card-title a:hover {{ text-decoration: underline; }}
.card-summary {{ margin: 0; font-size: 14px; color: var(--fg2); line-height: 1.7; }}
</style>
</head>
<body>
<div class="container">
  <header class="site-header">
    <h1>AI論文まとめ</h1>
    <p>Semantic Scholarの影響力スコア順に厳選したAI論文の日本語要約集</p>
  </header>
  <div class="stats">
    <span>論文数: <strong>{count}</strong></span>
    <span>最終更新: <strong>{updated}</strong></span>
    <span>影響力スコア（influential citations）順</span>
  </div>
  {cards}
</div>
</body>
</html>"""


# ── Slack ─────────────────────────────────────────────────────────────────────

def post_to_slack(webhook_url: str, new_papers: list[dict]) -> None:
    date_str = datetime.now(JST).strftime("%Y年%m月%d日")
    blocks = [
        {
            "type": "header",
            "text": {"type": "plain_text", "text": f"📚 AI論文まとめ 新着 — {date_str}"},
        },
        {"type": "divider"},
    ]
    for p in new_papers:
        summary = p.get("summary", "（要約なし）")
        blocks.append({
            "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": (
                    f"*{p['title']}* ({p.get('year','')})\n"
                    f"★ {p.get('influentialCitationCount',0)} influential · 引用 {p.get('citationCount',0)}\n"
                    f"{summary}"
                ),
            },
        })
        blocks.append({
            "type": "context",
            "elements": [{"type": "mrkdwn", "text": f"<{p['url']}|論文を読む>"}],
        })
        blocks.append({"type": "divider"})

    resp = requests.post(webhook_url, json={"blocks": blocks}, timeout=15)
    resp.raise_for_status()
    print("Slack post OK.")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    webhook = os.environ.get("SLACK_WEBHOOK_URL")
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        sys.exit("Error: GEMINI_API_KEY is not set.")

    print("Fetching Semantic Scholar candidates...")
    candidates = fetch_candidates()

    existing = load_papers()
    sent_ids = {p["paperId"] for p in existing}

    new_papers = pick_new(candidates, sent_ids)
    if not new_papers:
        print("No new papers to add.")
        return

    print(f"Summarizing {len(new_papers)} new papers...")
    summaries = summarize(api_key, new_papers)

    today = datetime.now(JST).strftime("%Y-%m-%d")
    for p, s in zip(new_papers, summaries):
        p["summary"] = s
        p["addedDate"] = today
        del p["abstract"]

    all_papers = existing + new_papers
    save_papers(all_papers)
    print(f"papers.json updated: {len(all_papers)} papers total.")

    html = generate_index_html(all_papers)
    (DOCS_DIR / "index.html").write_text(html, encoding="utf-8")
    print("docs/index.html generated.")

    if webhook:
        post_to_slack(webhook, new_papers)
    else:
        print("SLACK_WEBHOOK_URL not set, skipping Slack post.")


if __name__ == "__main__":
    main()
