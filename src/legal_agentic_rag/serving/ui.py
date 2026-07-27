"""Same-origin diagnostic UI for local and proxied baseline inspection."""

from __future__ import annotations

from html import escape
import json

from fastapi import FastAPI
from fastapi.responses import HTMLResponse


def mount_diagnostic_ui(
    app: FastAPI,
    *,
    path: str,
    title: str,
    answer_endpoint: str,
) -> FastAPI:
    """Mount a lightweight UI that calls the public answer API directly."""
    page = _render_page(title=title, answer_endpoint=answer_endpoint)

    @app.get(path, response_class=HTMLResponse, include_in_schema=False)
    async def diagnostic_ui() -> HTMLResponse:
        return HTMLResponse(page)

    return app


def _render_page(*, title: str, answer_endpoint: str) -> str:
    safe_title = escape(title)
    endpoint_json = json.dumps(answer_endpoint)
    return f"""<!doctype html>
<html lang="vi">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{safe_title}</title>
  <style>
    :root {{
      color-scheme: light dark;
      font-family: Inter, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }}
    body {{ margin: 0; background: #0f172a; color: #e2e8f0; }}
    main {{ width: min(960px, calc(100% - 32px)); margin: 32px auto; }}
    h1 {{ margin-bottom: 8px; }}
    .notice {{ color: #cbd5e1; margin-bottom: 24px; }}
    label {{ display: block; font-weight: 700; margin: 18px 0 8px; }}
    textarea {{
      box-sizing: border-box;
      width: 100%;
      min-height: 120px;
      resize: vertical;
      padding: 14px;
      border: 1px solid #475569;
      border-radius: 10px;
      background: #111827;
      color: #f8fafc;
      font: inherit;
    }}
    button {{
      margin-top: 12px;
      padding: 11px 22px;
      border: 0;
      border-radius: 9px;
      background: #2563eb;
      color: white;
      font: inherit;
      font-weight: 700;
      cursor: pointer;
    }}
    button:disabled {{ cursor: wait; opacity: .65; }}
    section {{
      margin-top: 24px;
      padding: 18px;
      border: 1px solid #334155;
      border-radius: 12px;
      background: #111827;
    }}
    pre {{
      margin: 0;
      white-space: pre-wrap;
      overflow-wrap: anywhere;
      font-family: inherit;
      line-height: 1.55;
    }}
    details {{ margin-top: 14px; }}
    summary {{ cursor: pointer; font-weight: 700; }}
    .status {{ min-height: 24px; margin-top: 12px; color: #93c5fd; }}
    .error {{ color: #fca5a5; }}
  </style>
</head>
<body>
  <main>
    <h1>{safe_title}</h1>
    <p class="notice">
      Câu trả lời chỉ mang tính hỗ trợ tra cứu và phải được đối chiếu với
      văn bản pháp luật chính thức.
    </p>

    <form id="question-form">
      <label for="question">Câu hỏi pháp luật</label>
      <textarea id="question" required
        placeholder="Nhập câu hỏi bằng tiếng Việt..."></textarea>
      <button id="submit" type="submit">Tra cứu</button>
      <div id="status" class="status" role="status"></div>
    </form>

    <section>
      <label>Câu trả lời</label>
      <pre id="answer">Chưa có câu trả lời.</pre>
      <details>
        <summary>Trích dẫn</summary>
        <pre id="citations">[]</pre>
      </details>
      <details>
        <summary>Cảnh báo và Trace ID</summary>
        <pre id="metadata"></pre>
      </details>
    </section>
  </main>

  <script>
    const answerEndpoint = {endpoint_json};
    const form = document.getElementById("question-form");
    const question = document.getElementById("question");
    const submit = document.getElementById("submit");
    const status = document.getElementById("status");
    const answer = document.getElementById("answer");
    const citations = document.getElementById("citations");
    const metadata = document.getElementById("metadata");

    form.addEventListener("submit", async (event) => {{
      event.preventDefault();
      const text = question.value.trim();
      if (!text) return;

      submit.disabled = true;
      status.className = "status";
      status.textContent = "Đang xử lý...";
      const started = performance.now();

      try {{
        const response = await fetch(answerEndpoint, {{
          method: "POST",
          headers: {{ "Content-Type": "application/json" }},
          body: JSON.stringify({{ question: text }})
        }});
        const payload = await response.json();
        if (!response.ok) {{
          const message = payload?.error?.message || `HTTP ${{response.status}}`;
          throw new Error(message);
        }}
        answer.textContent = payload.answer || "";
        citations.textContent = JSON.stringify(payload.citations || [], null, 2);
        metadata.textContent = JSON.stringify({{
          insufficient_evidence: payload.insufficient_evidence,
          warnings: payload.warnings || [],
          trace_id: payload.trace_id
        }}, null, 2);
        const seconds = ((performance.now() - started) / 1000).toFixed(2);
        status.textContent = `Hoàn thành trong ${{seconds}} giây`;
      }} catch (error) {{
        status.className = "status error";
        status.textContent = `Không thể xử lý: ${{error.message}}`;
      }} finally {{
        submit.disabled = false;
      }}
    }});
  </script>
</body>
</html>
"""
