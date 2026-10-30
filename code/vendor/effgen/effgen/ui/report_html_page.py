"""The page shell every HTML report is wrapped in.

Holds the inline stylesheet, the theme-toggle and copy-button script, and the
document assembly. Nothing here reaches the network: the stylesheet is built
from :mod:`effgen.ui.palette` and emitted inline, the script is inline, and the
document carries no ``src``, ``href`` or ``@import`` to any host. That is what
lets a report render identically from disk with the network disabled.

Colors are written as CSS custom properties for a light and a dark palette. The
page follows the reader's ``prefers-color-scheme`` and the toggle overrides it
per view.
"""

from __future__ import annotations

from .palette import DASHBOARD_DARK, DASHBOARD_LIGHT
from .report_html_format import _esc


def _css() -> str:
    def block(selector: str, palette: dict[str, str]) -> str:
        body = "".join(f"--{k}:{v};" for k, v in palette.items())
        return f"{selector}{{{body}}}"

    return (
        block(":root", DASHBOARD_LIGHT)
        + "@media (prefers-color-scheme: dark){" + block(":root", DASHBOARD_DARK) + "}"
        + block(':root[data-theme="light"]', DASHBOARD_LIGHT)
        + block(':root[data-theme="dark"]', DASHBOARD_DARK)
        + """
*{box-sizing:border-box}
body{margin:0;padding:0;background:var(--bg);color:var(--text);
 font-family:ui-sans-serif,-apple-system,"Segoe UI",Roboto,Helvetica,Arial,sans-serif;
 font-size:15px;line-height:1.5}
main{max-width:1040px;margin:0 auto;padding:2rem 1.25rem 4rem}
header.report-head{border-bottom:1px solid var(--border);padding-bottom:1rem;margin-bottom:1.75rem;
 display:flex;flex-wrap:wrap;gap:1rem;align-items:flex-start;justify-content:space-between}
h1{font-size:1.6rem;margin:0 0 .35rem}
h2{font-size:1.15rem;margin:2rem 0 .75rem;padding-bottom:.35rem;border-bottom:1px solid var(--border)}
h3{font-size:.95rem;margin:0 0 .6rem;color:var(--text-muted);font-weight:600;
 letter-spacing:.02em;text-transform:uppercase}
.subtitle{color:var(--text-muted);margin:0}
.provenance{margin:.6rem 0 0;padding:0;list-style:none;display:flex;flex-wrap:wrap;gap:.4rem}
.provenance li{background:var(--bg-card);border:1px solid var(--border);border-radius:999px;
 padding:.15rem .6rem;font-size:.78rem;color:var(--text-muted)}
.provenance code{font-size:.78rem;color:var(--text-muted);word-break:break-all}
button.theme-toggle{background:var(--bg-card);color:var(--text);border:1px solid var(--border);
 border-radius:8px;padding:.4rem .75rem;font:inherit;font-size:.82rem;cursor:pointer}
button.theme-toggle:focus-visible{outline:2px solid var(--focus);outline-offset:2px}
.verdict{background:var(--bg-panel);border:1px solid var(--border);border-left:4px solid var(--accent);
 border-radius:10px;padding:1rem 1.15rem;margin:0 0 1.5rem}
.verdict-label{font-size:.78rem;text-transform:uppercase;letter-spacing:.04em;color:var(--text-muted)}
.verdict-value{font-size:1.3rem;font-weight:650;margin:.15rem 0 .3rem;word-break:break-word}
.verdict-note{color:var(--text-muted);margin:0}
.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(160px,1fr));gap:.75rem;margin-bottom:1.5rem}
.card{background:var(--bg-panel);border:1px solid var(--border);border-radius:10px;padding:.85rem 1rem}
.card-label{font-size:.75rem;text-transform:uppercase;letter-spacing:.04em;color:var(--text-muted)}
.card-value{font-size:1.35rem;font-weight:650;margin-top:.2rem;word-break:break-word}
.card-note{font-size:.78rem;color:var(--text-faint);margin-top:.15rem}
.chart{background:var(--bg-panel);border:1px solid var(--border);border-radius:10px;
 padding:1rem 1.15rem;margin-bottom:1rem}
.bar-row{display:grid;grid-template-columns:minmax(90px,1.3fr) 3fr minmax(84px,auto);
 gap:.6rem;align-items:center;margin-bottom:.45rem}
.bar-label{font-size:.85rem;color:var(--text-muted);overflow-wrap:anywhere}
.bar-track{background:var(--bg-inset);border-radius:5px;height:14px;overflow:hidden}
.bar-fill{height:100%;border-radius:5px}
.bar-none{height:100%;background:repeating-linear-gradient(45deg,var(--border),var(--border) 4px,
 transparent 4px,transparent 8px)}
.bar-accent{background:var(--accent)}.bar-accent2{background:var(--accent2)}
.bar-ok{background:var(--ok)}.bar-warn{background:var(--warn)}.bar-err{background:var(--err)}
.bar-value{font-variant-numeric:tabular-nums;font-size:.85rem;text-align:right}
.meter-note{margin:.5rem 0 0;color:var(--text-muted);font-size:.85rem}
.donut-wrap{display:flex;flex-wrap:wrap;gap:1.25rem;align-items:center}
.slice-accent{stroke:var(--accent)}.slice-accent2{stroke:var(--accent2)}
.slice-ok{stroke:var(--ok)}.slice-warn{stroke:var(--warn)}.slice-err{stroke:var(--err)}
.legend{list-style:none;margin:0;padding:0;font-size:.85rem}
.legend li{display:flex;align-items:center;gap:.45rem;margin-bottom:.3rem}
.legend-value{color:var(--text-muted);font-variant-numeric:tabular-nums}
.swatch{width:11px;height:11px;border-radius:3px;display:inline-block;flex:none}
.swatch-accent{background:var(--accent)}.swatch-accent2{background:var(--accent2)}
.swatch-ok{background:var(--ok)}.swatch-warn{background:var(--warn)}.swatch-err{background:var(--err)}
.table-wrap{overflow-x:auto;border:1px solid var(--border);border-radius:10px;background:var(--bg-panel)}
table{border-collapse:collapse;width:100%;font-size:.88rem}
caption{caption-side:top;text-align:left;padding:.7rem 1rem 0;color:var(--text-muted);font-size:.82rem}
th,td{text-align:left;padding:.55rem .8rem;border-bottom:1px solid var(--border);vertical-align:top}
th{background:var(--bg-card);font-weight:600;font-size:.78rem;text-transform:uppercase;
 letter-spacing:.03em;color:var(--text-muted);position:sticky;top:0}
tbody tr:last-child td{border-bottom:none}
.num{display:block;text-align:right;font-variant-numeric:tabular-nums;white-space:nowrap}
.wrap{display:block;max-width:340px;overflow-wrap:anywhere}
.badge{display:inline-block;border-radius:999px;padding:.1rem .55rem;font-size:.76rem;
 font-weight:600;border:1px solid}
.badge-ok{color:var(--ok);border-color:var(--ok)}
.badge-err{color:var(--err);border-color:var(--err)}
.badge-warn{color:var(--warn);border-color:var(--warn)}
.badge-muted{color:var(--text-muted);border-color:var(--border)}
.empty{color:var(--text-muted);margin:.25rem 0}
code{font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace}
a{color:var(--accent)}
.prose{background:var(--bg-panel);border:1px solid var(--border);border-radius:10px;
 padding:1rem 1.15rem;margin:0 0 1rem;white-space:pre-wrap;overflow-wrap:anywhere;
 max-width:72ch;font-size:.95rem}
.mono{font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;font-size:.82rem;
 white-space:pre-wrap;overflow-wrap:anywhere;display:block;max-width:420px}
.task-text{white-space:pre-wrap;overflow-wrap:anywhere}
.actions{display:flex;flex-wrap:wrap;gap:.5rem;margin:0 0 1.5rem}
button.copy-btn{background:var(--bg-card);color:var(--text);border:1px solid var(--border);
 border-radius:8px;padding:.4rem .75rem;font:inherit;font-size:.82rem;cursor:pointer}
button.copy-btn:focus-visible{outline:2px solid var(--focus);outline-offset:2px}
.sources{margin:0;padding-left:1.4rem;font-size:.9rem}
.sources li{margin-bottom:.3rem;overflow-wrap:anywhere}
.quote{border-left:3px solid var(--border);margin:.3rem 0 0;padding-left:.7rem;
 color:var(--text-muted);font-size:.86rem}
footer{margin-top:2.5rem;padding-top:1rem;border-top:1px solid var(--border);
 color:var(--text-faint);font-size:.8rem}
@media print{body{background:#fff}.theme-toggle,.actions{display:none}
 .card,.chart,.table-wrap,.verdict,.prose{break-inside:avoid}}
"""
    )


_THEME_SCRIPT = """
(function(){
  var root=document.documentElement,btn=document.querySelector('.theme-toggle');
  if(!btn){return;}
  function label(){
    var explicit=root.getAttribute('data-theme');
    var dark=explicit?explicit==='dark'
      :window.matchMedia('(prefers-color-scheme: dark)').matches;
    btn.textContent=dark?'Light theme':'Dark theme';
    btn.setAttribute('aria-label','Switch to '+(dark?'light':'dark')+' theme');
  }
  btn.addEventListener('click',function(){
    var explicit=root.getAttribute('data-theme');
    var dark=explicit?explicit==='dark'
      :window.matchMedia('(prefers-color-scheme: dark)').matches;
    root.setAttribute('data-theme',dark?'light':'dark');
    label();
  });
  label();
})();
(function(){
  var buttons=document.querySelectorAll('button.copy-btn');
  if(!buttons.length){return;}
  Array.prototype.forEach.call(buttons,function(btn){
    var node=document.getElementById(btn.getAttribute('data-copy-from'));
    if(!node){btn.disabled=true;return;}
    var original=btn.textContent;
    btn.addEventListener('click',function(){
      var text=node.textContent;
      function done(ok){
        btn.textContent=ok?'Copied':'Press Ctrl+C to copy';
        window.setTimeout(function(){btn.textContent=original;},2000);
      }
      if(navigator.clipboard&&navigator.clipboard.writeText){
        navigator.clipboard.writeText(text).then(function(){done(true);},
          function(){fallback(text,done);});
      }else{fallback(text,done);}
    });
  });
  function fallback(text,done){
    var area=document.createElement('textarea');
    area.value=text;area.setAttribute('readonly','');
    area.style.position='fixed';area.style.opacity='0';
    document.body.appendChild(area);area.select();
    var ok=false;
    try{ok=document.execCommand('copy');}catch(e){ok=false;}
    document.body.removeChild(area);done(ok);
  }
})();
"""


def _provenance_items(command: str | None, generated_at: str) -> str:
    from effgen import __version__

    items = [
        f"<li>Generated {_esc(generated_at)}</li>",
        f"<li>effGen {_esc(__version__)}</li>",
    ]
    if command:
        items.append(f"<li><code>{_esc(command)}</code></li>")
    return '<ul class="provenance">' + "".join(items) + "</ul>"


def _page(
    *,
    title: str,
    subtitle: str,
    command: str | None,
    generated_at: str,
    body: str,
) -> str:
    return (
        "<!DOCTYPE html>\n"
        '<html lang="en">\n<head>\n<meta charset="utf-8">\n'
        '<meta name="viewport" content="width=device-width, initial-scale=1">\n'
        f"<title>{_esc(title)}</title>\n"
        f"<style>{_css()}</style>\n</head>\n<body>\n<main>\n"
        '<header class="report-head"><div>'
        f"<h1>{_esc(title)}</h1>"
        f'<p class="subtitle">{_esc(subtitle)}</p>'
        f"{_provenance_items(command, generated_at)}"
        "</div>"
        '<button type="button" class="theme-toggle">Dark theme</button>'
        "</header>\n"
        f"{body}\n"
        "<footer>Rendered from the result of the command above. "
        "All styles, scripts, and charts are contained in this file.</footer>\n"
        "</main>\n"
        f"<script>{_THEME_SCRIPT}</script>\n</body>\n</html>\n"
    )
