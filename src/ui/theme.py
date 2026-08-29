PAGE_CSS = """
<style>
:root {
  --studio-bg: #0b0c0a;
  --studio-panel: #111410;
  --studio-line: #293027;
  --studio-ink: #ecf4ea;
  --studio-muted: #8b9788;
  --studio-signal: #00e676;
}
html, body, .stApp, [data-testid="stSidebar"], [data-testid="stAppViewContainer"] {
  font-family: "Cascadia Mono", "IBM Plex Mono", ui-monospace, monospace;
  font-variant-numeric: tabular-nums;
}
.stApp, [data-testid="stSidebar"] {
  background: var(--studio-bg);
  color: var(--studio-ink);
}
[data-testid="stHeader"] { background: transparent; }
[data-testid="stSidebar"] { border-right: 1px solid var(--studio-line); }
.block-container {
  max-width: 1320px;
  padding-top: 1.6rem;
  padding-bottom: 3.2rem;
}
.deck { color: var(--studio-muted); max-width: 46rem; font-size: 0.95rem; }
.proof-identity {
  color: var(--studio-muted);
  font-size: 0.8rem;
  margin: 0 0 0.8rem;
}
.result-label {
  color: var(--studio-signal);
  font-size: 0.72rem;
  letter-spacing: 0.08em;
  text-transform: uppercase;
}
.result-rule {
  border-top: 1px solid var(--studio-line);
  margin: 0.6rem 0 1rem;
}
.auto-direction {
  direction: auto;
  unicode-bidi: plaintext;
  white-space: pre-wrap;
  line-height: 1.7;
}
.empty-proof {
  border-top: 1px solid var(--studio-line);
  border-bottom: 1px solid var(--studio-line);
  color: var(--studio-muted);
  padding: 2.4rem 0;
}
input, textarea, [data-testid="stTextArea"] textarea {
  font-size: 16px !important;
}
.stButton > button {
  border: 1px solid var(--studio-signal);
  border-radius: 0;
  background: var(--studio-signal);
  color: #000;
  box-shadow: none;
}
[data-testid="stChatMessage"],
[data-testid="stFileUploaderDropzone"],
[data-testid="stExpander"],
[data-testid="stStatusWidget"] {
  border: 1px solid var(--studio-line);
  border-radius: 0;
  background: var(--studio-panel);
  box-shadow: none;
}
[data-testid="stWidgetLabel"] p,
[data-testid="stRadio"] label,
[data-testid="stCaptionContainer"] p {
  color: var(--studio-ink) !important;
}
[data-testid="stCaptionContainer"] p,
[data-testid="stFileUploader"] small {
  color: var(--studio-muted) !important;
}
@media (max-width: 900px) {
  .block-container { padding: 1rem 0.85rem 2.4rem; }
}
</style>
"""
