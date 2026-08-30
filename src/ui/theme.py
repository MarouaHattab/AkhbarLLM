_FONT_IMPORT = (
    "@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+"
    "Sans:wght@400;500;600;700&family=Tajawal:wght@400;500;700&display=swap');"
)

_LIGHT_VARS = """
:root {
  --studio-bg: #ffffff;
  --studio-panel: #f0f2f6;
  --studio-card: #ffffff;
  --studio-line: #d6d6d6;
  --studio-ink: #31333f;
  --studio-muted: #808495;
  --studio-red: #ff4b4b;
  --studio-green: #10b981;
}
"""

_DARK_VARS = """
:root {
  --studio-bg: #0e1117;
  --studio-panel: #161b22;
  --studio-card: #1c212c;
  --studio-line: #30363d;
  --studio-ink: #e6edf3;
  --studio-muted: #8b949e;
  --studio-red: #ff4b4b;
  --studio-green: #3fb950;
}
"""

_SHARED_CSS = """
html, body, .stApp, [data-testid="stAppViewContainer"], [data-testid="stSidebar"] {
  font-family: "IBM Plex Sans", "Tajawal", sans-serif;
  background: var(--studio-bg) !important;
  color: var(--studio-ink) !important;
}
[data-testid="stHeader"], [data-testid="stToolbar"] {
  background: var(--studio-bg) !important;
}
.block-container {
  padding-top: 1.5rem;
  padding-bottom: 3rem;
  max-width: 1200px;
}
[data-testid="stSidebar"] {
  background: var(--studio-panel) !important;
  border-right: 1px solid var(--studio-line);
}
h1, h2, h3, h4 {
  font-family: "IBM Plex Sans", "Tajawal", sans-serif !important;
  color: var(--studio-ink);
}
h1 { font-size: 28px !important; }
.deck {
  color: var(--studio-muted);
  font-size: 0.92rem;
  margin-bottom: 0.4rem;
}
.result-label {
  color: var(--studio-red);
  font-size: 0.72rem;
  letter-spacing: 0.08em;
  text-transform: uppercase;
  margin: 0 0 0.6rem;
}
.result-card {
  background: var(--studio-card);
  border: 1px solid var(--studio-line);
  border-radius: 12px;
  padding: 1.1rem 1.15rem 1.2rem;
  margin: 0.4rem 0 1rem;
}
.story-title {
  font-family: "Tajawal", "IBM Plex Sans", sans-serif !important;
  font-size: 1.35rem;
  font-weight: 700;
  line-height: 1.7;
  margin: 0 0 1rem;
}
.meta-grid {
  display: grid;
  grid-template-columns: 160px 1fr;
  gap: 0.85rem;
  margin-bottom: 1rem;
}
.meta-box {
  background: var(--studio-panel);
  border: 1px solid var(--studio-line);
  border-radius: 10px;
  padding: 0.75rem 0.9rem;
}
.meta-label {
  display: block;
  color: var(--studio-muted);
  font-size: 0.75rem;
  margin-bottom: 0.35rem;
}
.meta-value {
  font-weight: 700;
  font-size: 1.05rem;
}
.chip-row { display: flex; flex-wrap: wrap; gap: 0.4rem; }
.keyword-chip {
  display: inline-block;
  font-family: "Tajawal", "IBM Plex Sans", sans-serif !important;
  background: var(--studio-card);
  border: 1px solid var(--studio-line);
  border-radius: 999px;
  padding: 0.2rem 0.7rem;
  font-size: 0.9rem;
}
.section-heading {
  font-size: 0.78rem;
  font-weight: 700;
  letter-spacing: 0.04em;
  text-transform: uppercase;
  color: var(--studio-muted);
  margin: 1.1rem 0 0.45rem;
}
.summary-list {
  margin: 0;
  padding: 0;
  list-style: none;
  counter-reset: summary;
}
.summary-list li {
  font-family: "Tajawal", "IBM Plex Sans", sans-serif !important;
  display: grid;
  grid-template-columns: 1.6rem 1fr;
  gap: 0.55rem;
  border-bottom: 1px solid var(--studio-line);
  padding: 0.65rem 0;
  line-height: 1.85;
}
.summary-list li:last-child { border-bottom: none; }
.summary-list li::before {
  counter-increment: summary;
  content: counter(summary);
  width: 1.45rem;
  height: 1.45rem;
  border-radius: 999px;
  background: var(--studio-panel);
  border: 1px solid var(--studio-line);
  color: var(--studio-muted);
  font-size: 0.72rem;
  font-weight: 700;
  display: inline-flex;
  align-items: center;
  justify-content: center;
  margin-top: 0.2rem;
}
.entity-table {
  width: 100%;
  border-collapse: collapse;
  font-size: 0.95rem;
}
.entity-table th {
  text-align: left;
  color: var(--studio-muted);
  font-size: 0.72rem;
  font-weight: 600;
  letter-spacing: 0.04em;
  text-transform: uppercase;
  padding: 0.45rem 0.4rem;
  border-bottom: 1px solid var(--studio-line);
}
.entity-table td {
  padding: 0.65rem 0.4rem;
  border-bottom: 1px solid var(--studio-line);
  vertical-align: middle;
}
.entity-table tr:last-child td { border-bottom: none; }
.type-chip {
  display: inline-block;
  background: var(--studio-panel);
  border: 1px solid var(--studio-line);
  border-radius: 999px;
  padding: 0.15rem 0.65rem;
  font-size: 0.78rem;
  color: var(--studio-ink);
}
.raw-block {
  margin-top: 0.75rem;
  border: 1px solid var(--studio-line);
  border-radius: 10px;
  background: var(--studio-card);
  padding: 0.55rem 0.8rem 0.7rem;
}
.raw-block summary {
  cursor: pointer;
  color: var(--studio-muted);
  font-size: 0.82rem;
  font-weight: 600;
}
[data-testid="stCode"] {
  direction: ltr !important;
  unicode-bidi: isolate;
  text-align: left;
}
[data-testid="stCode"] pre,
[data-testid="stCode"] code {
  direction: ltr !important;
  unicode-bidi: isolate;
  font-family: "Tajawal", "IBM Plex Sans", sans-serif !important;
  font-size: 16px !important;
  line-height: 1.8 !important;
  white-space: pre-wrap !important;
  word-break: break-word;
}
[data-testid="stIconMaterial"],
[data-testid="stExpanderToggleIcon"] {
  font-family: "Material Symbols Rounded", "Material Symbols Outlined",
    "Material Icons" !important;
  font-style: normal !important;
  font-weight: 400 !important;
  speak: never;
}
.auto-direction {
  direction: auto;
  unicode-bidi: plaintext;
  white-space: pre-wrap;
  line-height: 1.8;
  font-family: "Tajawal", "IBM Plex Sans", sans-serif !important;
}
.story-body {
  background: var(--studio-panel);
  border: 1px solid var(--studio-line);
  border-radius: 10px;
  padding: 0.9rem 1rem;
  font-size: 1.02rem;
}
.live-label {
  color: var(--studio-muted);
  font-size: 0.75rem;
  margin: 0 0 0.4rem;
}
.output-box {
  background: var(--studio-panel);
  border: 1px solid var(--studio-line);
  border-radius: 10px;
  padding: 18px;
  font-family: "Tajawal", "IBM Plex Sans", sans-serif !important;
  font-size: 16px;
  line-height: 1.8;
  white-space: pre-wrap;
  min-height: 140px;
}
.empty-proof { color: var(--studio-muted); }
.status-online,
.status-offline,
.status-idle {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  padding: 4px 12px;
  border-radius: 20px;
  font-size: 12px;
  font-weight: 600;
}
.status-online {
  background: rgba(16, 185, 129, 0.12);
  border: 1px solid rgba(16, 185, 129, 0.4);
  color: var(--studio-green);
}
.status-offline {
  background: rgba(255, 75, 75, 0.1);
  border: 1px solid rgba(255, 75, 75, 0.35);
  color: var(--studio-red);
}
.status-idle {
  background: rgba(128, 132, 149, 0.12);
  border: 1px solid var(--studio-line);
  color: var(--studio-muted);
}
.dot {
  width: 7px;
  height: 7px;
  border-radius: 50%;
  display: inline-block;
  background: currentColor;
}
[data-testid="stMetric"], [data-testid="metric-container"] {
  background: var(--studio-card);
  border: 1px solid var(--studio-line);
  border-radius: 8px;
  padding: 12px;
}
.stButton > button[kind="primary"] {
  background: var(--studio-red) !important;
  color: #ffffff !important;
  border: 1px solid var(--studio-red) !important;
  font-family: "IBM Plex Sans", sans-serif !important;
}
.stButton > button[kind="primary"]:hover {
  background: #ff2b2b !important;
}
.stButton > button {
  color: var(--studio-ink) !important;
  border-color: var(--studio-line) !important;
}
.stTextArea textarea,
[data-testid="stChatMessage"],
[data-testid="stChatMessage"] p,
[data-testid="stChatMessage"] *,
[data-testid="stMarkdownContainer"] p,
[data-testid="stDataFrame"],
.stCode, pre, code {
  font-family: "Tajawal", "IBM Plex Sans", sans-serif !important;
}
.stTextArea textarea,
.stTextInput input,
[data-baseweb="input"] input {
  background: var(--studio-card) !important;
  color: var(--studio-ink) !important;
  border-color: var(--studio-line) !important;
}
.stTextArea textarea {
  font-size: 16px !important;
  direction: auto;
  unicode-bidi: plaintext;
}
.stCode, pre, code {
  font-family: "Tajawal", "IBM Plex Sans", sans-serif !important;
}
@media (max-width: 900px) {
  .block-container { padding: 1rem 0.85rem 2.4rem; }
  .meta-grid { grid-template-columns: 1fr; }
}
"""

_DARK_WIDGET_CSS = """
[data-testid="stAppViewContainer"],
[data-testid="stMain"],
section.main,
[data-testid="stVerticalBlock"] {
  background: var(--studio-bg) !important;
}
[data-testid="stTabs"] button {
  color: var(--studio-ink) !important;
}
[data-testid="stExpander"] details,
[data-testid="stExpander"] summary {
  background: var(--studio-card) !important;
  color: var(--studio-ink) !important;
}
div[data-baseweb="select"] > div,
div[data-baseweb="base-input"] {
  background-color: var(--studio-card) !important;
  color: var(--studio-ink) !important;
}
"""


def page_css(theme: str = "light") -> str:
    variables = _DARK_VARS if theme == "dark" else _LIGHT_VARS
    extra = _DARK_WIDGET_CSS if theme == "dark" else ""
    return f"<style>\n{_FONT_IMPORT}\n{variables}{_SHARED_CSS}{extra}\n</style>"


PAGE_CSS = page_css("light")
