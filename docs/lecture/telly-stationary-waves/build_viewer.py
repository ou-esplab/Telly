#!/usr/bin/env python3
"""
Build viewer.html: a standalone page that shows the deck in project/ as slides.

    python3 build_viewer.py          # writes viewer.html next to this script

Reads project/deck.json (slide order, title, fonts) and project/slides/<id>.html, and
embeds every slide in one HTML file (images are read from images/ via images.json, so keep
the images folder next to viewer.html). Open viewer.html in any browser: left/right arrows,
space, PageUp/PageDown or a click to move; Home/End to jump; F for fullscreen. Fonts load
from Google Fonts, so the browser needs internet access for the intended typefaces.

This is a convenience preview. The published deck on claude.ai is the reference rendering
(and the place to export PDF or PowerPoint).
"""
import html
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))

CSS = """
html,body{margin:0;height:100%;background:#0b1320;overflow:hidden;font-family:Arial,sans-serif}
#stage{position:absolute;left:50%;top:50%;width:1920px;height:1080px;transform-origin:center center}
#stage section{position:absolute;left:0;top:0;width:1920px;height:1080px;
  box-sizing:border-box;overflow:hidden;margin:0}
/* slides set their own inline display (flex or grid), so hiding needs !important */
#stage section:not(.on){display:none !important}
#stage aside{display:none}
#stage h1,#stage h2,#stage h3,#stage p,#stage ul,#stage ol{margin:0}
#stage ul,#stage ol{padding-left:1.1em}
#stage table{border-collapse:collapse}
#stage th,#stage td{padding:.35em .6em;border-bottom:1px solid #D9D4C7;vertical-align:top}
#stage th{font-weight:600}
#stage x-shape{display:block}
#stage x-shape[kind="arrow-right"]{clip-path:polygon(0 25%,60% 25%,60% 0,100% 50%,60% 100%,60% 75%,0 75%)}
#hud{position:fixed;right:16px;bottom:12px;color:#8fa0b5;font:14px/1 Arial,sans-serif;user-select:none}
"""

JS = """
const slides=[...document.querySelectorAll('#stage section')];
let i=Math.max(0,Math.min(slides.length-1,(parseInt(location.hash.slice(1))||1)-1));
const stage=document.getElementById('stage'),hud=document.getElementById('hud');
function fit(){const s=Math.min(innerWidth/1920,innerHeight/1080);
  stage.style.transform='translate(-50%,-50%) scale('+s+')';}
function show(n){i=Math.max(0,Math.min(slides.length-1,n));
  slides.forEach((s,k)=>s.classList.toggle('on',k===i));
  hud.textContent=(i+1)+' / '+slides.length+'  \\u00b7  '+slides[i].id;
  history.replaceState(null,'','#'+(i+1));}
addEventListener('resize',fit);
addEventListener('keydown',e=>{
  if(['ArrowRight','ArrowDown','PageDown',' '].includes(e.key)){e.preventDefault();show(i+1)}
  else if(['ArrowLeft','ArrowUp','PageUp'].includes(e.key)){e.preventDefault();show(i-1)}
  else if(e.key==='Home')show(0);else if(e.key==='End')show(slides.length-1);
  else if(e.key==='f'||e.key==='F'){document.fullscreenElement?document.exitFullscreen():document.documentElement.requestFullscreen()}});
addEventListener('click',e=>show(e.clientX<innerWidth/3?i-1:i+1));
fit();show(i);
"""


def main():
    with open(os.path.join(HERE, "project", "deck.json")) as f:
        deck = json.load(f)
    links = "\n".join(
        f'<link rel="stylesheet" href="{html.escape(face["href"])}">'
        for face in deck.get("faces", {}).values() if face.get("href"))
    # Slides refer to uploaded images as /_blob/<id>; images.json maps each id to a local file.
    blobs = {}
    ipath = os.path.join(HERE, "images.json")
    if os.path.exists(ipath):
        with open(ipath) as f:
            blobs = json.load(f)
    parts = []
    for sid in deck["order"]:
        with open(os.path.join(HERE, "project", "slides", f"{sid}.html")) as f:
            text = f.read().strip()
        for blob_id, local in blobs.items():
            text = text.replace(f"/_blob/{blob_id}", local)
        parts.append(text)
    out = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{html.escape(deck.get("title", "Slides"))}</title>
{links}
<style>{CSS}</style></head>
<body><div id="stage">
{chr(10).join(parts)}
</div><div id="hud"></div>
<script>{JS}</script></body></html>
"""
    path = os.path.join(HERE, "viewer.html")
    with open(path, "w") as f:
        f.write(out)
    print(f"Wrote {path} ({len(deck['order'])} slides, {len(out) // 1024} KB)")


if __name__ == "__main__":
    main()
