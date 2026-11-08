"""
petrand.server
==============
Zero-dependency HTTP Web Server & REST API for PetRand.
Unifies:
  1. PetRand console (`/`)
  2. Zoomies!, the motion game (`/zoomies`)
  3. Everyday Randomness Studio for Non-Technical Users (`/api/v1/toolkit`, `/api/v1/export.csv`)
  4. Live In-Browser Ball Color Preset Configuration (`/api/v1/config/colors`)
Validated against the App & Dashboard Validation Harness (Gates A0-A10).
"""

import json
import time
import struct
import hashlib
from http.server import HTTPServer, ThreadingHTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs
from typing import Optional

from petrand.engine import PetRandEngine, RawPhysicalEvent, fill_until_ready, SENSOR_POLL_LOCK, EngineHaltedError
from petrand.health import HealthError
from petrand.lifecycle import TwoBallManager, BallStatus, COLOR_PRESETS
from petrand.sensors import (
    SyntheticArenaSimulator,
    CameraMotionSensor,
    BaseSensor,
)
from petrand.zoomies import (
    FORM_CARDS,
    ZoomiesScorer,
    render_zoomies_html,
)
from petrand.adapters import SeedSpooler
from petrand.toolkit import EverydaySeedToolkit
from petrand.http_guard import REQUEST_TIMEOUT_S, read_body, request_refusal


DASHBOARD_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<link rel="icon" href="data:,">
<title>PetRand // Chaos Entropy Console</title>
<style>
  :root {
    color-scheme: dark;
    --bg: #0b0f19; --card: #121826; --card-2: #0e1420; --line: #243049;
    --text: #f1f5f9; --muted: #aab5c9;
    --orange: #ff8c1a; --blue: #38bdf8; --green: #34d399; --red: #f87171; --yellow: #facc15;
  }
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { background: var(--bg); color: var(--text); font: 15px/1.45 -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; padding: 16px; overflow-x: hidden; }
  .wrap { max-width: 1180px; margin: 0 auto; display: grid; gap: 16px; }
  .mono { font-family: ui-monospace, "SF Mono", Menlo, monospace; }
  header { display: flex; flex-wrap: wrap; align-items: center; justify-content: space-between; gap: 10px; }
  .brand { display: flex; align-items: center; gap: 8px; flex-wrap: wrap; }
  .brand h1 { font-size: 1.2rem; font-weight: 850; letter-spacing: -0.02em; margin-right: 4px; }
  .brand h1 .dot { color: var(--orange); }
  .brand h1 small { font-weight: 650; color: var(--muted); font-size: 0.86rem; }
  .badge { display: inline-block; font-size: 0.72rem; padding: 4px 10px; border-radius: 999px; font-weight: 800; letter-spacing: 0.02em; border: 1px solid; white-space: nowrap; }
  .b-green { background: #052e22; color: #6ee7b7; border-color: #10b981; }
  .b-orange { background: #3b1a06; color: #fed7aa; border-color: #f97316; }
  .b-blue { background: #082f49; color: #bae6fd; border-color: #0284c7; }
  .b-red { background: #450a0a; color: #fecaca; border-color: #ef4444; }
  .btn { appearance: none; background: #1b2436; color: var(--text); border: 1px solid #3a4763; border-radius: 10px; padding: 9px 14px; font-weight: 750; font-size: 0.88rem; cursor: pointer; text-decoration: none; display: inline-flex; align-items: center; justify-content: center; gap: 6px; }
  .btn:hover { background: #24304a; }
  .btn:focus-visible, select:focus-visible, input:focus-visible, summary:focus-visible, a:focus-visible { outline: 3px solid var(--blue); outline-offset: 2px; }
  .btn-game { border-color: #a855f7; background: #2a1245; }
  .btn-game:hover { background: #3a1a5e; }
  .btn-small { padding: 4px 10px; font-size: 0.78rem; border-radius: 8px; }
  .error-banner { display: none; background: #450a0a; border: 1px solid #ef4444; border-radius: 10px; padding: 10px 14px; font-weight: 700; }
  .card { background: var(--card); border: 1px solid var(--line); border-radius: 14px; padding: 18px; min-width: 0; }
  .card h2 { font-size: 1rem; font-weight: 850; margin-bottom: 12px; display: flex; justify-content: space-between; align-items: center; gap: 8px; flex-wrap: wrap; }

  .hero { display: grid; grid-template-columns: minmax(0, 1fr) minmax(0, 1.25fr); gap: 20px; }
  @media (max-width: 900px) { .hero { grid-template-columns: 1fr; } }
  .hero .title { font-size: 1.55rem; line-height: 1.2; letter-spacing: -0.02em; margin-bottom: 6px; display: block; }
  .mode-line { color: var(--muted); font-size: 0.92rem; margin-bottom: 16px; }
  .cta { width: 100%; font-size: 1.12rem; padding: 16px 18px; border-radius: 14px; background: var(--orange); color: #1a0d00; border: none; font-weight: 900; box-shadow: 0 6px 0 #9a4a00; }
  .cta:hover { background: #ff9d3d; }
  .cta:active { transform: translateY(3px); box-shadow: 0 3px 0 #9a4a00; }
  .cta[disabled] { opacity: 0.75; cursor: progress; }
  .seed-wrap { margin-top: 16px; }
  .label { font-size: 0.78rem; color: var(--muted); font-weight: 750; margin-bottom: 6px; display: flex; justify-content: space-between; align-items: center; gap: 8px; }
  .seed-box { background: var(--card-2); border: 1px solid var(--line); border-radius: 10px; padding: 10px 12px; color: var(--green); font-size: 0.84rem; word-break: break-all; min-height: 44px; }
  .seed-box.msg { color: var(--muted); font-family: inherit; word-break: normal; overflow-wrap: anywhere; }
  .progress-line { color: var(--muted); font-size: 0.86rem; margin: 10px 0 0; }
  .stopped { background: rgba(239, 68, 68, 0.12); border: 1px solid #ef4444; border-radius: 10px; padding: 10px 12px; margin-top: 12px; }
  .stopped p { margin: 0 0 8px; font-size: 0.9rem; }
  .card-head { display: flex; justify-content: space-between; align-items: center; gap: 8px; flex-wrap: wrap; margin-bottom: 12px; }
  .card-head h2 { margin-bottom: 0; }
  .muted-note { font-weight: 700; color: var(--muted); font-size: 0.86rem; }
  .field-label { display: block; font-size: 0.8rem; color: var(--muted); font-weight: 750; margin: 10px 0 4px; }
  textarea.raffle-input { width: 100%; resize: vertical; font: inherit; font-size: 0.9rem; }
  details.check-draw { margin-top: 12px; border-top: 1px solid var(--line); padding-top: 10px; }
  details.check-draw > summary { cursor: pointer; font-weight: 800; font-size: 0.9rem; color: var(--blue); }
  .check-row { display: flex; gap: 8px; align-items: center; margin-top: 10px; flex-wrap: wrap; }
  .no-caps { text-transform: none; }

  .tiles { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 12px; align-content: start; }
  @media (max-width: 520px) { .tiles { grid-template-columns: 1fr; } }
  .tile { background: var(--card-2); border: 1px solid var(--line); border-radius: 12px; padding: 12px 14px; min-width: 0; }
  .tile .k { font-size: 0.74rem; color: var(--muted); font-weight: 800; text-transform: uppercase; letter-spacing: 0.05em; display: flex; justify-content: space-between; align-items: center; gap: 6px; margin-bottom: 8px; }
  .verb { color: var(--yellow); font-size: 0.72rem; font-weight: 900; font-style: italic; letter-spacing: 0.04em; text-transform: uppercase; transform: rotate(-4deg); pointer-events: none; }
  .big { font-size: 1.06rem; font-weight: 850; color: var(--yellow); overflow-wrap: anywhere; }
  .rolls { display: grid; grid-template-columns: repeat(3, 1fr); gap: 8px; text-align: center; }
  .rolls .v { font-size: 1.3rem; font-weight: 900; color: #6ee7b7; }
  .rolls .n { font-size: 0.72rem; color: var(--muted); font-weight: 750; }
  .sub { font-size: 0.8rem; color: var(--muted); margin-top: 6px; }
  .row { display: flex; justify-content: space-between; align-items: center; gap: 8px; flex-wrap: wrap; }

  .raffle-row { display: flex; gap: 8px; flex-wrap: wrap; align-items: center; }
  .raffle-input { flex: 1 1 240px; min-width: 0; background: var(--card-2); color: var(--text); border: 1px solid #3a4763; border-radius: 10px; padding: 10px 12px; font-size: 0.9rem; }
  .winner-line { margin-top: 12px; font-size: 0.95rem; }
  .winner-line strong { color: var(--green); font-size: 1.1rem; }

  .how { display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 10px; list-style: none; }
  @media (max-width: 900px) { .how { grid-template-columns: repeat(2, minmax(0, 1fr)); } }
  @media (max-width: 520px) { .how { grid-template-columns: 1fr; } }
  .step { background: var(--card-2); border: 1px solid var(--line); border-radius: 12px; padding: 12px; }
  .s-head { display: flex; align-items: center; gap: 8px; font-weight: 850; font-size: 0.92rem; margin-bottom: 4px; }
  .num { width: 24px; height: 24px; border-radius: 50%; display: inline-grid; place-items: center; font-size: 0.78rem; font-weight: 900; background: #1f2a40; color: var(--blue); border: 1px solid #33415c; flex: none; }
  .step.ok .num { background: #052e22; color: #6ee7b7; border-color: #10b981; }
  .step.warn .num { background: #3b1a06; color: #fed7aa; border-color: #f97316; }
  .step.bad .num { background: #450a0a; color: #fecaca; border-color: #ef4444; }
  .step p { font-size: 0.86rem; color: var(--muted); }
  .step p strong { color: var(--text); }

  .lower { display: grid; grid-template-columns: minmax(0, 1fr) minmax(0, 1fr); gap: 16px; align-items: start; }
  @media (max-width: 900px) { .lower { grid-template-columns: 1fr; } }
  .canvas-wrap { width: 100%; height: 240px; }
  canvas#arena-canvas { width: 100%; height: 100%; display: block; background: #06090f; border-radius: 10px; border: 1px solid var(--line); }
  .ball-row { display: grid; grid-template-columns: 1fr 1fr; gap: 10px; margin-top: 12px; }
  @media (max-width: 480px) { .ball-row { grid-template-columns: 1fr; } }
  .ball-card { border: 1px solid var(--line); border-radius: 10px; padding: 10px; font-size: 0.82rem; }
  .ball-card.active { border-color: var(--orange); background: rgba(255, 140, 26, 0.07); }
  .ball-card.active.blue { border-color: var(--blue); background: rgba(56, 189, 248, 0.07); }
  .ball-name { font-weight: 800; display: flex; justify-content: space-between; align-items: center; gap: 6px; }
  .battery-bar { height: 6px; background: #1e293b; border-radius: 3px; margin: 8px 0; overflow: hidden; }
  .battery-fill { height: 100%; background: var(--green); width: 100%; transition: width 0.3s; }
  .color-select { background: var(--bg); color: var(--text); border: 1px solid #475569; border-radius: 6px; padding: 3px 6px; font-size: 0.8rem; font-weight: 700; }

  details.hood > summary { cursor: pointer; font-weight: 850; font-size: 1rem; list-style: none; display: flex; justify-content: space-between; align-items: center; gap: 8px; }
  details.hood > summary::-webkit-details-marker { display: none; }
  .hood-toggle::after { content: "Show"; font-size: 0.8rem; color: var(--blue); font-weight: 800; }
  details.hood[open] .hood-toggle::after { content: "Hide"; }
  .hood-hint { font-size: 0.82rem; color: var(--muted); margin-top: 4px; }
  .hood-section { margin-top: 14px; }
  .hood-section h3 { font-size: 0.76rem; text-transform: uppercase; letter-spacing: 0.06em; color: var(--muted); margin-bottom: 2px; }
  .metric-row { display: flex; justify-content: space-between; align-items: center; gap: 10px; padding: 8px 0; border-bottom: 1px solid rgba(255, 255, 255, 0.07); font-size: 0.86rem; }
  .metric-row:last-child { border-bottom: none; }
  .metric-sub { display: block; font-size: 0.74rem; color: var(--muted); }
  .metric-val { white-space: nowrap; font-weight: 800; text-align: right; }
  .progress-bg { height: 8px; background: #1e293b; border-radius: 4px; overflow: hidden; margin: 6px 0 10px; }
  .progress-fill { height: 100%; background: var(--orange); width: 0%; transition: width 0.2s; }
  .sensor-row { display: flex; flex-wrap: wrap; gap: 8px; align-items: center; margin: 6px 0; }
  .note { font-size: 0.8rem; color: var(--muted); margin-top: 14px; }
  .fade { animation: fade 0.25s ease-out; }
  @keyframes fade { from { opacity: 0.4; } to { opacity: 1; } }
  @media (prefers-reduced-motion: reduce) { .fade { animation: none; } .cta:active { transform: none; } .battery-fill, .progress-fill { transition: none; } }
  .visually-hidden { position: absolute; width: 1px; height: 1px; overflow: hidden; clip: rect(0 0 0 0); white-space: nowrap; }
</style>
</head>
<body>
<div class="wrap">
  <header>
    <div class="brand">
      <h1>PetRand <span class="dot" aria-hidden="true">●</span> <small>Random number console</small></h1>
      <span id="source-badge" class="badge b-orange">SOURCE: …</span>
      <span id="health-badge" class="badge b-orange">CONNECTING…</span>
    </div>
    <a id="btn-open-zoomies" class="btn btn-game" href="/zoomies">Play Zoomies! →</a>
  </header>

  <div id="feed-error-banner" class="error-banner" role="alert">Can't reach the PetRand server. Readings are paused until it's back.</div>

  <p id="announce" class="visually-hidden" aria-live="polite"></p>
  <main class="wrap">
    <section class="card hero" aria-labelledby="hero-title">
      <div>
        <h2 id="hero-title" class="title">Random numbers from a bouncing ball</h2>
        <p id="mode-line" class="mode-line">Connecting…</p>
        <button id="btn-extract-seed" class="btn cta" onclick="extractSeed()">New random number</button>
        <p id="progress-line" class="progress-line" hidden></p>
        <div id="stopped-panel" class="stopped" hidden>
          <p><strong>Stopped:</strong> a safety check failed, so no numbers are made until the checks restart. Details are under the hood.</p>
          <button id="btn-reset" class="btn btn-small" onclick="resetChecks()">Restart the checks</button>
        </div>
        <div class="seed-wrap">
          <div class="label"><span>Your random number</span><button class="btn btn-small" onclick="copyText('seed-box', this)" aria-label="Copy the number">Copy</button></div>
          <div id="seed-box" class="seed-box mono msg">No number yet.</div>
        </div>
      </div>
      <div class="tiles">
        <div class="tile">
          <div class="k"><span>Passphrase</span><span class="verb" aria-hidden="true">Unlock!</span></div>
          <div id="tk-passphrase" class="big mono">—</div>
          <div class="sub row"><span>PIN <strong id="tk-pin" class="mono">—</strong></span><button class="btn btn-small" onclick="copyText('tk-passphrase', this)" aria-label="Copy the passphrase">Copy</button></div>
        </div>
        <div class="tile">
          <div class="k"><span>Dice and coin</span></div>
          <div class="rolls">
            <div><div class="v mono" id="tk-coin">—</div><div class="n">Coin</div></div>
            <div><div class="v mono" id="tk-d20">—</div><div class="n">D20</div></div>
            <div><div class="v mono" id="tk-d100">—</div><div class="n">D100</div></div>
          </div>
        </div>
        <div class="tile">
          <div class="k"><span>Lottery 6/49</span></div>
          <div id="tk-lottery" class="big mono">—</div>
        </div>
        <div class="tile">
          <div class="k"><span class="no-caps">π from your number</span></div>
          <div id="pi-est" class="big mono">—</div>
          <div class="sub">Monte Carlo, 20,000 points</div>
        </div>
      </div>
    </section>

    <section class="card" aria-labelledby="raffle-title">
      <div class="card-head"><h2 id="raffle-title">Pick a winner</h2><span class="verb" aria-hidden="true">Shuffle!</span></div>
      <label for="raffle-input" class="field-label">Names, separated by commas or new lines</label>
      <textarea id="raffle-input" class="raffle-input" rows="2">Alice, Bob, Charlie, Diana, Echo, Frank</textarea>
      <div class="raffle-row" style="margin-top: 8px;">
        <button id="btn-run-raffle" class="btn" onclick="runEverydayRaffle()">Draw</button>
        <a id="btn-download-csv" class="btn" href="/api/v1/export.csv?rows=500" download="petrand_random_numbers.csv">Download 500 rows (.csv)</a>
      </div>
      <p class="winner-line">Winner: <strong id="tk-winner">—</strong></p>
      <p class="sub">Order: <span id="tk-shuffled">—</span></p>
      <p class="sub">Receipt <span id="tk-receipt" class="mono" style="color: var(--yellow);">—</span>. Anyone with the number and the same list can redo the draw below and get this receipt. It can't show whether someone drew more than once.</p>
      <details class="check-draw">
        <summary>Check a draw</summary>
        <label for="check-seed" class="field-label">Random number (64 hex digits)</label>
        <input id="check-seed" class="raffle-input mono" type="text" spellcheck="false" autocomplete="off">
        <label for="check-names" class="field-label">Names, in the same order</label>
        <textarea id="check-names" class="raffle-input" rows="2"></textarea>
        <label for="check-receipt" class="field-label">Receipt (optional)</label>
        <input id="check-receipt" class="raffle-input mono" type="text" spellcheck="false" autocomplete="off" placeholder="VERIFY-…">
        <div class="check-row"><button id="btn-check-draw" class="btn btn-small" onclick="checkDraw()">Check</button><button id="btn-use-last-draw" class="btn btn-small" onclick="useLastDraw()">Use the last draw</button></div>
        <p id="check-result" class="sub" aria-live="polite">Paste a number, its names and its receipt, then press Check.</p>
      </details>
    </section>

    <section class="card" aria-labelledby="how-title">
      <h2 id="how-title">How a number gets made</h2>
      <ol class="how">
        <li id="step-measure" class="step"><div class="s-head"><span class="num">1</span>Measure</div><p id="how-measure">—</p></li>
        <li id="step-check" class="step"><div class="s-head"><span class="num">2</span>Check</div><p id="how-check">—</p></li>
        <li id="step-count" class="step"><div class="s-head"><span class="num">3</span>Count</div><p id="how-count">—</p></li>
        <li id="step-mix" class="step"><div class="s-head"><span class="num">4</span>Mix</div><p id="how-mix">—</p></li>
      </ol>
    </section>

    <div class="lower">
      <section class="card" aria-labelledby="arena-title">
        <div class="card-head"><h2 id="arena-title">Arena</h2><span id="active-ball-name" class="muted-note">—</span></div>
        <div class="canvas-wrap"><canvas id="arena-canvas" role="img" aria-label="Overhead view of the ball's recent path"></canvas></div>
        <div class="ball-row">
          <div id="card-ball-a" class="ball-card active">
            <div class="ball-name"><span id="name-ball-a">Ball 1</span><span id="status-ball-a" class="badge b-orange">ACTIVE</span></div>
            <div class="battery-bar"><div id="bat-fill-a" class="battery-fill"></div></div>
            <div class="row" style="color: var(--muted);"><span>Battery <span id="bat-text-a">—</span></span><span>Impacts <span id="ev-text-a">0</span></span></div>
            <div class="row" style="margin-top: 6px;">
              <label for="select-color-a" style="color: var(--muted);">Ball 1 color</label>
              <select id="select-color-a" class="color-select" onchange="updateBallColors('a')">
                <option value="orange" selected>Orange</option><option value="blue">Blue</option><option value="yellow">Yellow</option><option value="green">Green</option><option value="pink">Pink</option>
              </select>
            </div>
          </div>
          <div id="card-ball-b" class="ball-card">
            <div class="ball-name"><span id="name-ball-b">Ball 2</span><span id="status-ball-b" class="badge b-blue">CHARGING</span></div>
            <div class="battery-bar"><div id="bat-fill-b" class="battery-fill" style="background: var(--blue);"></div></div>
            <div class="row" style="color: var(--muted);"><span>Battery <span id="bat-text-b">—</span></span><span>Impacts <span id="ev-text-b">0</span></span></div>
            <div class="row" style="margin-top: 6px;">
              <label for="select-color-b" style="color: var(--muted);">Ball 2 color</label>
              <select id="select-color-b" class="color-select" onchange="updateBallColors('b')">
                <option value="orange">Orange</option><option value="blue" selected>Blue</option><option value="yellow">Yellow</option><option value="green">Green</option><option value="pink">Pink</option>
              </select>
            </div>
          </div>
        </div>
        <div class="row" style="margin-top: 12px;">
          <span id="swap-note" class="sub" style="margin-top: 0;">Swap when a battery runs low. Batteries are estimated from run time.</span>
          <button id="btn-swap-ball" class="btn btn-small" onclick="triggerSwap()">Swap ball</button>
        </div>
      </section>

      <section class="card" aria-labelledby="hood-title">
        <details class="hood" id="hood">
          <summary id="hood-title">Under the hood <span class="hood-toggle" aria-hidden="true"></span></summary>
          <p class="hood-hint">Health tests, entropy credit and the motion game's checks.</p>
          <div class="hood-section">
            <h3>Health tests</h3>
            <div class="metric-row"><span>Repeat guard<span class="metric-sub">Stops if one reading repeats too often in a row</span></span><span class="metric-val mono" id="rct-val">—</span></div>
            <div class="metric-row"><span>Balance guard<span class="metric-sub">Stops if one reading fills too much of a 512-sample window</span></span><span class="metric-val mono" id="apt-val">—</span></div>
            <div class="metric-row"><span>Start-up check<span class="metric-sub">1,024 clean samples before anything counts</span></span><span class="metric-val" id="startup-val">—</span></div>
            <div class="metric-row"><span>Self-tests</span><span id="selftest-badge" class="badge b-blue">…</span></div>
            <div class="metric-row"><span>Last alarm</span><span class="mono" id="halt-detail" style="font-size: 0.78rem; text-align: right;">none</span></div>
          </div>
          <div class="hood-section">
            <h3>Entropy credit</h3>
            <div class="metric-row"><span>Credit per sample<span class="metric-sub">A cautious placeholder until the source is measured</span></span><span class="metric-val mono" id="h-inf-val">—</span></div>
            <div class="row" style="margin-top: 8px;"><span>Toward the next seed</span><span id="credit-text" class="mono">—</span></div>
            <div class="progress-bg"><div id="credit-fill" class="progress-fill"></div></div>
            <div class="metric-row"><span>Seeds made this session</span><span class="metric-val mono" id="seed-count">0</span></div>
          </div>
          <div class="hood-section">
            <h3>Motion game</h3>
            <div class="metric-row"><span>Rhythm detector<span class="metric-sub">Scripted test motions it should reject</span></span><span class="metric-val" id="bot-audit">—</span></div>
          </div>
          <p class="note">The health tests follow the designs in NIST SP 800-90B. Numbers are mixed with HMAC-SHA-256.</p>
        </details>
      </section>
    </div>
  </main>
  <footer class="note" style="text-align: center; margin-top: 0;">Experimental and not certified. Health tests follow the NIST SP 800-90B designs; credit per sample is a placeholder until the source is measured.</footer>
</div>

<script>
  const $ = (id) => document.getElementById(id);
  const canvas = $('arena-canvas');
  const ctx = canvas.getContext('2d');
  const trail = [];
  let cssW = 400, cssH = 240;
  let ballX = 200, ballY = 120;
  let activeColorHex = '#ff8c1a';
  // Every number on this page comes from a seed this tab asked for; nothing is synced from elsewhere
  const TILE_IDS = ['tk-passphrase', 'tk-pin', 'tk-coin', 'tk-d20', 'tk-d100', 'tk-lottery', 'pi-est', 'tk-winner', 'tk-shuffled', 'tk-receipt'];
  function clearTiles() { TILE_IDS.forEach((id) => { $(id).innerText = '—'; }); }
  let lastReport = null;
  let sourceKind = 'unknown';
  let pendingRoll = true;      // the first number is made as soon as the engine is ready
  let rolling = false;
  let lastAttempt = 0;
  let lastTelemetry = null;
  let lastDraw = null;
  let colorsSynced = false;

  const COLOR_MAP = { orange: '#ff8c1a', blue: '#38bdf8', yellow: '#eab308', green: '#34d399', pink: '#f472b6' };

  function resizeArenaCanvas() {
    const rect = canvas.parentElement.getBoundingClientRect();
    const dpr = Math.max(2, window.devicePixelRatio || 1);
    cssW = Math.max(200, rect.width);
    cssH = Math.max(180, rect.height);
    canvas.width = Math.round(cssW * dpr);
    canvas.height = Math.round(cssH * dpr);
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  }
  window.addEventListener('resize', resizeArenaCanvas);
  resizeArenaCanvas();

  function flash(id) {
    const el = $(id);
    el.classList.remove('fade');
    void el.offsetWidth;
    el.classList.add('fade');
  }

  // Small seeded generator (sfc32) so the π demo is driven by your seed, not by the browser
  function sfc32(a, b, c, d) {
    return function () {
      a >>>= 0; b >>>= 0; c >>>= 0; d >>>= 0;
      let t = (a + b) | 0;
      a = b ^ (b >>> 9);
      b = (c + (c << 3)) | 0;
      c = (c << 21) | (c >>> 11);
      d = (d + 1) | 0;
      t = (t + d) | 0;
      c = (c + t) | 0;
      return (t >>> 0) / 4294967296;
    };
  }
  function piFromSeed(hex) {
    const w = [0, 8, 16, 24].map((i) => parseInt(hex.slice(i, i + 8), 16) >>> 0);
    const rnd = sfc32(w[0], w[1], w[2], w[3]);
    for (let i = 0; i < 16; i++) rnd();
    let inside = 0;
    const n = 20000;
    for (let i = 0; i < n; i++) {
      const x = rnd(), y = rnd();
      if (x * x + y * y <= 1) inside++;
    }
    return (4 * inside / n).toFixed(4);
  }

  function announce(msg) {
    const el = $('announce');
    el.textContent = '';
    setTimeout(() => { el.textContent = msg; }, 30);
  }

  function setSeedBox(text, isNumber) {
    const box = $('seed-box');
    box.innerText = text;
    box.classList.toggle('msg', !isNumber);
  }

  function parseNames(raw) {
    return String(raw || '').split(/[,\\n]/).map((s) => s.trim()).filter(Boolean).slice(0, 1000);
  }

  async function copyText(id, btn) {
    const text = $(id).innerText.trim();
    // Copy only a real number or passphrase, never a message
    const ok = id === 'seed-box' ? /^[0-9a-f]{64}$/i.test(text) : /^[a-z]+(-[a-z]+)+$/.test(text);
    if (!ok) { announce('Nothing to copy yet.'); return; }
    const old = btn.innerText;
    try {
      await navigator.clipboard.writeText(text);
      btn.innerText = 'Copied';
      announce('Copied.');
    } catch (e) {
      btn.innerText = 'Copy failed';
      announce('Copy failed.');
    }
    setTimeout(() => { btn.innerText = old; }, 1200);
  }

  function describeSource(src) {
    const kind = String(src.kind || 'unknown');
    if (src.physical) {
      return {
        badge: 'SOURCE: ' + kind.toUpperCase() + ' (LIVE)', cls: 'b-green', noun: 'the ' + kind,
        line: 'Live arena: numbers come from the ' + kind + '. How much randomness it gives has not been measured yet.'
      };
    }
    if (kind === 'simulator') {
      return {
        badge: 'SOURCE: SIMULATOR', cls: 'b-orange', noun: 'the simulator',
        line: "Simulator mode: no camera is watching a real ball, so the randomness comes from your computer's own generator. Great for games, dice and raffles."
      };
    }
    const device = kind.replace('-simulator', '');
    return {
      badge: 'SOURCE: ' + kind.toUpperCase(), cls: 'b-orange', noun: 'the simulator',
      line: "No real " + device + " found, so a simulator stands in. The randomness comes from your computer's own generator."
    };
  }

  async function refreshToolkit(seedHex) {
    try {
      let items = parseNames($('raffle-input').value);
      if (items.length < 2) items = ['(no names)', '(no names)'];
      const res = await fetch('/api/v1/toolkit', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ seed_hex: seedHex || null, items: items, winners_count: 1 })
      });
      if (!res.ok) { clearTiles(); return; }
      const data = await res.json();
      $('tk-passphrase').innerText = data.passphrase_bundle.passphrase;
      $('tk-pin').innerText = data.passphrase_bundle.pin_6_digit;
      $('tk-coin').innerText = data.tabletop_bundle.coin_flip;
      $('tk-d20').innerText = data.tabletop_bundle.d20;
      $('tk-d100').innerText = data.tabletop_bundle.d100;
      $('tk-lottery').innerText = data.tabletop_bundle.lottery_6_of_49.join(' · ');
      $('tk-winner').innerText = data.raffle_bundle.winner_primary;
      $('tk-shuffled').innerText = data.raffle_bundle.shuffled_order.join(' → ');
      $('tk-receipt').innerText = data.raffle_bundle.verification_receipt;
      lastDraw = { seed: seedHex, names: items, receipt: data.raffle_bundle.verification_receipt };
      if (data.seed_hex && /^[0-9a-f]{64}$/i.test(data.seed_hex)) {
        $('pi-est').innerText = piFromSeed(data.seed_hex);
      }
      ['tk-passphrase', 'tk-lottery', 'tk-winner', 'pi-est'].forEach(flash);
    } catch (e) { clearTiles(); }
  }

  async function loadBotAudit() {
    const el = $('bot-audit');
    try {
      const res = await fetch('/api/v1/antibot');
      if (!res.ok) throw new Error('HTTP ' + res.status);
      const data = await res.json();
      const rows = data.results || [];
      const bots = rows.filter((r) => r.is_adversarial_bot);
      const blocked = bots.filter((r) => !r.unlocked).length;
      el.innerText = blocked + '/' + bots.length + ' rejected';
    } catch (e) {
      el.innerText = 'not available';
    }
  }

  function setStep(id, state, html) {
    const li = $('step-' + id);
    li.className = 'step' + (state ? ' ' + state : '');
    $('how-' + id).innerHTML = html;
  }

  function renderHow(data, src) {
    const h = data.health, e = data.engine;
    const H = Number(h.min_entropy_h);
    if (lastReport) {
      // Everything here describes the number on screen, from its own report
      $('how-title').innerText = 'How your number was made';
      const extra = lastReport.uncredited_extra_inputs || 0;
      setStep('measure', 'ok', '<strong>' + lastReport.samples + '</strong> readings from ' + src.noun + '.' +
        (extra ? ' Plus ' + extra + ' Zoomies! round' + (extra === 1 ? '' : 's') + ' mixed in, not counted.' : ''));
      setStep('check', 'ok', 'Safety checks <strong>passed</strong>.');
      setStep('count', 'ok', 'Enough collected: <strong>' + Math.round(lastReport.credit_bits) + ' bits</strong>, at a cautious ' + lastReport.h_bits_per_sample + ' bit per reading.');
      setStep('mix', 'ok', 'Scrambled into number <strong>#' + lastReport.seed_number + '</strong>.');
      $('how-mix').title = 'Mixed with ' + (lastReport.conditioner || 'HMAC-SHA-256');
      return;
    }
    $('how-title').innerText = 'How a number gets made';
    let checkState = 'ok', checkHtml = 'Safety checks <strong>passing</strong>.';
    if (e.halted) { checkState = 'bad'; checkHtml = '<strong>Stopped</strong> by a safety check. Use Restart the checks above.'; }
    else if (!h.is_operational) { checkState = 'warn'; checkHtml = 'Warming up: <strong>' + h.startup_remaining + '</strong> start-up readings to go.'; }
    setStep('check', checkState, checkHtml);
    setStep('measure', '', '<strong>' + (e.events_in_pool || 0) + '</strong> readings collected from ' + src.noun + ' so far.');
    setStep('count', '', '<strong>' + e.entropy_credit_bits.toFixed(0) + ' of ' + e.required_entropy_bits.toFixed(0) + ' bits</strong> collected, at a cautious ' + H + ' bit per reading.');
    setStep('mix', '', 'When there are enough, the readings are scrambled into a 256-bit number.');
  }

  function showProgress(data) {
    const el = $('progress-line');
    const e = data.engine, h = data.health;
    let text = '';
    if (pendingRoll && !e.halted && !e.ready) {
      text = !h.is_operational
        ? 'Warming up: ' + h.startup_remaining + ' start-up readings to go. Your number will appear by itself.'
        : 'Collecting readings: ' + Math.floor(e.entropy_credit_bits) + ' of ' + Math.round(e.required_entropy_bits) + ' bits. Your number will appear by itself.';
    }
    el.innerText = text;
    el.hidden = !text;
  }

  async function updateTelemetry() {
    const errBanner = $('feed-error-banner');
    const healthBadge = $('health-badge');
    try {
      const res = await fetch('/api/v1/status');
      if (!res.ok) throw new Error('HTTP ' + res.status);
      const data = await res.json();
      errBanner.style.display = 'none';

      lastTelemetry = data;
      $('stopped-panel').hidden = !data.engine.halted;
      $('halt-detail').innerText = data.engine.halted ? String(data.engine.halt_reason || 'unknown') : 'none';
      showProgress(data);
      if (!colorsSynced && data.lifecycle) {
        $('select-color-a').value = data.lifecycle.ball_a_color || 'orange';
        $('select-color-b').value = data.lifecycle.ball_b_color || 'blue';
        colorsSynced = true;
      }
      if (data.engine.halted) { healthBadge.className = 'badge b-red'; healthBadge.innerText = 'STOPPED: HEALTH ALARM'; }
      else if (data.health.is_operational) { healthBadge.className = 'badge b-green'; healthBadge.innerText = 'HEALTHY'; }
      else { healthBadge.className = 'badge b-orange'; healthBadge.innerText = 'WARMING UP'; }

      const src = describeSource(data.source || {});
      sourceKind = String((data.source || {}).kind || 'unknown');
      $('source-badge').className = 'badge ' + src.cls;
      $('source-badge').innerText = src.badge;
      $('mode-line').innerText = src.line;
      $('hero-title').innerText = (data.source || {}).physical ? 'Random numbers from a bouncing ball' : 'Random numbers from a simulated ball';

      $('rct-val').innerText = data.health.rct_consecutive + ' / ' + data.health.rct_cutoff;
      $('apt-val').innerText = data.health.apt_matches + ' / ' + data.health.apt_cutoff;
      const startupEl = $('startup-val');
      startupEl.innerText = data.health.is_operational ? 'Done' : data.health.startup_remaining + ' to go';
      startupEl.style.color = data.health.is_operational ? 'var(--green)' : 'var(--orange)';
      $('h-inf-val').innerText = data.health.min_entropy_h + ' bit';
      const post = data.engine.post_self_test || 'not run';
      $('selftest-badge').className = 'badge ' + (post === 'passed' ? 'b-blue' : 'b-orange');
      $('selftest-badge').innerText = post === 'passed' ? 'PASSED AT START' : 'NOT RUN';

      const activeA = data.lifecycle.swap_count % 2 === 0;
      const activeColor = data.lifecycle.active_ball_color || 'orange';
      activeColorHex = COLOR_MAP[activeColor] || '#ff8c1a';
      $('active-ball-name').innerText = 'Active: Ball ' + (activeA ? '1' : '2') + ' (' + activeColor + ')';
      const simulated = !(data.source || {}).physical;
      $('arena-title').innerText = simulated ? 'Arena (simulated)' : 'Arena';
      $('swap-note').innerText = simulated
        ? 'Simulated balls: the battery is a run-time estimate.'
        : 'Swap when a battery runs low. Batteries are estimated from run time.';
      $('card-ball-a').className = 'ball-card' + (activeA ? ' active' : '');
      $('card-ball-b').className = 'ball-card' + (!activeA ? ' active blue' : '');
      $('status-ball-a').innerText = activeA ? 'ACTIVE' : 'CHARGING';
      $('status-ball-b').innerText = !activeA ? 'ACTIVE' : 'CHARGING';
      // Only the active ball has a run-time estimate; the idle one shows no made-up number
      const bat = Math.round(data.lifecycle.estimated_battery_percent);
      $('bat-fill-a').style.width = (activeA ? bat : 0) + '%';
      $('bat-fill-b').style.width = (!activeA ? bat : 0) + '%';
      $('bat-text-a').innerText = activeA ? '~' + bat + '% (estimated)' : 'charging';
      $('bat-text-b').innerText = !activeA ? '~' + bat + '% (estimated)' : 'charging';
      $('ev-text-a').innerText = data.lifecycle.ball_a_events || 0;
      $('ev-text-b').innerText = data.lifecycle.ball_b_events || 0;

      const credit = data.engine.entropy_credit_bits;
      const req = data.engine.required_entropy_bits;
      $('credit-text').innerText = credit.toFixed(1) + ' / ' + req.toFixed(1) + ' bits';
      $('credit-fill').style.width = Math.min(100, (credit / Math.max(1, req)) * 100) + '%';
      $('seed-count').innerText = data.engine.counter;

      if (data.ball_coords) {
        ballX = data.ball_coords.x * (cssW / 1.2);
        ballY = data.ball_coords.y * (cssH / 1.2);
      }
      renderHow(data, src);
      if (pendingRoll && !rolling && !data.engine.halted && (data.engine.ready || Date.now() - lastAttempt > 3000)) {
        pendingRoll = false;
        setTimeout(extractSeed, 0);
      }
      return true;
    } catch (e) {
      errBanner.style.display = 'block';
      healthBadge.className = 'badge b-red';
      healthBadge.innerText = 'OFFLINE';
      return false;
    }
  }

  function renderArena() {
    ctx.fillStyle = '#06090f';
    ctx.fillRect(0, 0, cssW, cssH);
    if (sourceKind === 'simulator') {
      // The simulator's three circular scatterers (positions in a 1.2 m arena)
      ctx.fillStyle = '#1e293b';
      for (const [ox, oy, orad] of [[0.4, 0.4, 0.08], [0.8, 0.7, 0.10], [0.3, 0.9, 0.06]]) {
        ctx.beginPath();
        ctx.arc(ox / 1.2 * cssW, oy / 1.2 * cssH, orad / 1.2 * Math.min(cssW, cssH), 0, Math.PI * 2);
        ctx.fill();
      }
    }
    trail.push({ x: ballX, y: ballY });
    if (trail.length > 28) trail.shift();
    ctx.strokeStyle = activeColorHex;
    ctx.lineWidth = 2;
    ctx.beginPath();
    trail.forEach((p, i) => { if (i === 0) ctx.moveTo(p.x, p.y); else ctx.lineTo(p.x, p.y); });
    ctx.stroke();
    ctx.fillStyle = activeColorHex;
    ctx.beginPath();
    ctx.arc(ballX, ballY, 7, 0, Math.PI * 2);
    ctx.fill();
    requestAnimationFrame(renderArena);
  }

  async function extractSeed() {
    if (rolling) return;   // the button stays focusable; a second press during a roll does nothing
    const btn = $('btn-extract-seed');
    if (lastTelemetry && lastTelemetry.engine.halted) {
      $('stopped-panel').hidden = false;
      announce('Stopped: restart the checks first.');
      return;
    }
    // Ask even if the background collection isn't done: the server tops up for up to 1.5 s,
    // and if that is still not enough the page retries by itself
    rolling = true;
    lastAttempt = Date.now();
    btn.setAttribute('aria-disabled', 'true');
    btn.innerText = 'Rolling…';
    try {
      // At least 0.4 s per roll: new values (and their fade-in) never change more than 2.5 times a second
      const [res] = await Promise.all([fetch('/api/v1/seed'), new Promise((r) => setTimeout(r, 400))]);
      const data = await res.json();
      if (data.seed) {
        lastReport = data.report || null;
        pendingRoll = false;
        setSeedBox(data.seed, true);
        flash('seed-box');
        $('pi-est').innerText = piFromSeed(data.seed);
        await refreshToolkit(data.seed);
        announce('New number ready.');
        await updateTelemetry();
      } else {
        lastReport = null;
        clearTiles();
        if (data.halted) { $('stopped-panel').hidden = false; setSeedBox('Stopped: restart the checks to make numbers again.', false); }
        else if (res.status === 503) {
          pendingRoll = true;
          setSeedBox('Collecting readings: your number will appear by itself.', false);
          if (lastTelemetry) showProgress(lastTelemetry);
        }
        else setSeedBox("Couldn't get a number: " + (data.error || 'unknown error') + '.', false);
      }
    } catch (e) {
      lastReport = null;
      clearTiles();
      setSeedBox("Couldn't get a number: the PetRand server isn't responding.", false);
      announce("Couldn't get a number.");
    } finally {
      rolling = false;
      btn.removeAttribute('aria-disabled');
      btn.innerText = 'New random number';
    }
  }

  async function runEverydayRaffle() {
    if (parseNames($('raffle-input').value).length < 2) {
      $('tk-winner').innerText = 'type at least two names';
      $('tk-shuffled').innerText = '—';
      $('tk-receipt').innerText = '—';
      return;
    }
    await extractSeed();
  }

  function useLastDraw() {
    if (!lastDraw) { $('check-result').innerText = 'Make a draw first.'; return; }
    $('check-seed').value = lastDraw.seed;
    $('check-names').value = lastDraw.names.join(', ');
    $('check-receipt').value = lastDraw.receipt;
  }

  async function checkDraw() {
    const out = $('check-result');
    const seed = $('check-seed').value.trim().toLowerCase();
    const names = parseNames($('check-names').value);
    if (!/^[0-9a-f]{64}$/.test(seed)) { out.innerText = 'The number must be 64 hex digits (0-9, a-f).'; return; }
    if (names.length < 2) { out.innerText = 'Type at least two names, in the order used for the draw.'; return; }
    try {
      const res = await fetch('/api/v1/toolkit', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ seed_hex: seed, items: names, winners_count: 1 })
      });
      if (!res.ok) throw new Error('HTTP ' + res.status);
      const r = (await res.json()).raffle_bundle;
      let given = $('check-receipt').value.trim().toUpperCase();
      if (given && !given.startsWith('VERIFY-')) given = 'VERIFY-' + given;
      const verdict = !given ? '' : (given === r.verification_receipt ? ' Matches the receipt.' : ' Does NOT match the receipt.');
      out.innerText = 'Order: ' + r.shuffled_order.join(' → ') + '. Receipt ' + r.verification_receipt + '.' + verdict;
    } catch (e) {
      out.innerText = "Couldn't check: the PetRand server isn't responding.";
    }
  }

  async function resetChecks() {
    try {
      const res = await fetch('/api/v1/reset', { method: 'POST' });
      if (!res.ok) throw new Error('HTTP ' + res.status);
      pendingRoll = true;
      lastReport = null;
      setSeedBox('Checks restarted: warming up. Your number will appear by itself.', false);
      announce('Checks restarted.');
      await updateTelemetry();
    } catch (e) {
      announce("Couldn't restart the checks.");
    }
  }

  async function triggerSwap() {
    try {
      const res = await fetch('/api/v1/swap', { method: 'POST' });
      const data = await res.json();
      await updateTelemetry();
    } catch (e) {
      $('active-ball-name').innerText = 'Swap failed: server offline';
    }
  }

  async function updateBallColors(which) {
    // Send only the ball that changed, so another tab's choice for the other ball stays
    const body = which === 'b' ? { color_b: $('select-color-b').value } : { color_a: $('select-color-a').value };
    try {
      await fetch('/api/v1/config/colors', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body)
      });
      await updateTelemetry();
    } catch (e) {}
  }

  renderArena();
  loadBotAudit();
  updateTelemetry();   // the first number follows as soon as the engine is ready
  setInterval(updateTelemetry, 500);
</script>
</body>
</html>
"""


class PetRandServerHandler(BaseHTTPRequestHandler):
    """Handles REST API and static UI requests for the PetRand console and the Zoomies! game."""

    engine: PetRandEngine
    ball_mgr: TwoBallManager
    sensor: BaseSensor
    spooler: Optional[SeedSpooler] = None
    latest_seed: Optional[str] = None  # only for the status fingerprint; never served or reused
    timeout = REQUEST_TIMEOUT_S         # drop idle or half-sent connections
    antibot_cache: Optional[dict] = None

    def _send_json(self, status_code: int, payload: dict) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status_code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    LOOPBACK_NAMES = ("127.0.0.1", "localhost", "[::1]", "::1")

    def _bound_to_loopback(self) -> bool:
        try:
            return self.server.server_address[0] in ("127.0.0.1", "::1", "localhost")
        except Exception:
            return True

    def _request_allowed(self, is_post: bool) -> bool:
        """Apply petrand.http_guard: Host check on loopback, other-site API requests and POSTs refused."""
        refusal = request_refusal(self, is_post)
        if refusal:
            self._send_json(refusal[0], {"error": refusal[1]})
            return False
        return True

    @staticmethod
    def _valid_seed_hex(value) -> Optional[str]:
        if isinstance(value, str) and 32 <= len(value) <= 128 and all(c in "0123456789abcdefABCDEF" for c in value):
            return value.lower()
        return None

    def do_GET(self):
        if not self._request_allowed(is_post=False):
            return
        url = urlparse(self.path)

        if url.path in ["/", "/index.html"]:
            body = DASHBOARD_HTML.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        elif url.path == "/zoomies":
            body = render_zoomies_html(console_link=True).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        elif url.path == "/api/v1/spool/status":
            # Read-only: looking at the spool never makes seeds
            if self.spooler is None:
                self._send_json(200, {"available_seeds": 0, "capacity": 0, "note": "no spool yet; /api/v1/seed/instant creates one"})
            else:
                self._send_json(200, self.spooler.get_status())

        elif url.path == "/api/v1/seed/instant":
            try:
                if self.spooler is None:
                    PetRandServerHandler.spooler = SeedSpooler(engine=self.engine, sensor=self.sensor, capacity=16)
                    PetRandServerHandler.spooler.fill_to_capacity()
                self._send_json(200, self.spooler.pop_seed())
            except Exception:
                self._send_json(503, {"error": "no seeds available yet; try again shortly"})

        elif url.path == "/api/v1/zoomies/cards":
            self._send_json(200, {"cards": FORM_CARDS})

        elif url.path == "/favicon.ico":
            self.send_response(204)
            self.end_headers()

        elif url.path == "/api/v1/status":
            telem = self.engine.get_telemetry()
            ball_telem = self.ball_mgr.get_telemetry()
            monitor_telem = self.engine.monitor.get_telemetry()

            ball_coords = {"x": 0.6, "y": 0.6}
            if isinstance(self.sensor, SyntheticArenaSimulator):
                ball_coords = {"x": self.sensor.x, "y": self.sensor.y}
            elif isinstance(self.sensor, CameraMotionSensor):
                tr = self.sensor.tracker
                if tr.prev_x is not None and tr.prev_y is not None:
                    ball_coords = {
                        "x": (tr.prev_x / max(1.0, float(tr.width))) * 1.2,
                        "y": (tr.prev_y / max(1.0, float(tr.height))) * 1.2,
                    }

            payload = {
                "engine": telem,
                "lifecycle": ball_telem,
                "health": {
                    "rct_consecutive": self.engine.monitor.rct_consecutive,
                    "rct_cutoff": self.engine.monitor.rct_cutoff,
                    "apt_matches": self.engine.monitor.apt_matches,
                    "apt_window_progress": len(self.engine.monitor.apt_window),
                    "apt_cutoff": self.engine.monitor.apt_cutoff,
                    "apt_window": self.engine.monitor.W,
                    "min_entropy_h": self.engine.monitor.H,
                    "is_operational": self.engine.monitor.is_operational,
                    "startup_remaining": monitor_telem.get("startup_samples_remaining", 0),
                },
                "ball_coords": ball_coords,
                "latest_seed_fingerprint": (
                    hashlib.sha256(PetRandServerHandler.latest_seed.encode()).hexdigest()[:16]
                    if PetRandServerHandler.latest_seed else None
                ),
                "source": self.sensor.describe() if hasattr(self.sensor, "describe") else {"kind": "unknown", "physical": False},
                "credit": {
                    "h_bits_per_sample": self.engine.monitor.H,
                    "assessed": False,
                    "note": "placeholder until the source is assessed with NIST's SP 800-90B tool",
                },
            }
            self._send_json(200, payload)

        elif url.path == "/api/v1/seed":
            try:
                # A short budget keeps one request from freezing this single-threaded server;
                # the background harvester does the steady filling.
                if not fill_until_ready(self.engine, self.sensor, timeout_s=1.5):
                    telem = self.engine.get_telemetry()
                    self._send_json(503, {
                        "error": ("stopped by a health test: " + str(telem["halt_reason"])) if telem["halted"]
                                 else "not enough entropy credit yet; try again shortly",
                        "entropy_credit_bits": round(telem["entropy_credit_bits"], 2),
                        "required_entropy_bits": telem["required_entropy_bits"],
                        "halted": bool(telem["halted"]),
                        "startup_remaining": telem["startup_remaining"],
                    })
                    return
                made = self.engine.extract_seed_with_report(seed_bytes=32)
                seed = made["seed"]
                PetRandServerHandler.latest_seed = seed.hex()
                resp = {
                    "seed": seed.hex(),
                    "seed_bytes": len(seed),
                    "source": self.sensor.describe() if hasattr(self.sensor, "describe") else {"kind": "unknown", "physical": False},
                    "report": made["report"],
                    "fingerprint": hashlib.sha256(seed.hex().encode()).hexdigest()[:16],
                    "active_ball": self.ball_mgr.current_ball.ball_id,
                    "timestamp_ns": time.time_ns(),
                }
                self._send_json(200, resp)
            except EngineHaltedError:
                self._send_json(503, {"error": "stopped by a safety check; restart the checks", "halted": True})
            except Exception:
                self._send_json(500, {"error": "internal error"})

        elif url.path == "/api/v1/export.csv":
            qs = parse_qs(url.query)
            try:
                rows = max(1, min(10000, int(qs.get("rows", ["500"])[0])))
            except ValueError:
                self._send_json(400, {"error": "rows must be a whole number"})
                return
            # An explicit ?seed= is used as given; otherwise every download gets its own fresh seed
            seed_hex = self._valid_seed_hex(qs.get("seed", [None])[0])
            if not seed_hex:
                if not fill_until_ready(self.engine, self.sensor, timeout_s=1.5):
                    self._send_json(503, {"error": "not enough entropy credit yet; try again shortly"})
                    return
                seed_hex = self.engine.extract_seed(seed_bytes=32).hex()
            csv_text = EverydaySeedToolkit.generate_csv_spreadsheet(seed_hex=seed_hex, rows=rows)
            body = csv_text.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/csv; charset=utf-8")
            self.send_header("Content-Disposition", 'attachment; filename="petrand_random_numbers.csv"')
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        elif url.path == "/api/v1/antibot":
            # Anti-bot audit (synthetic bots vs. the chaos meter and rhythm detector); cached
            # after the first run; reads no files.
            if PetRandServerHandler.antibot_cache is None:
                from petrand.antibot import ZoomiesBotAudit
                PetRandServerHandler.antibot_cache = ZoomiesBotAudit.run_adversarial_audit()
            self._send_json(200, PetRandServerHandler.antibot_cache)

        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self):
        if not self._request_allowed(is_post=True):
            return
        url = urlparse(self.path)
        raw_body, err = read_body(self)
        if err:
            self._send_json(err[0], {"error": err[1]})
            return
        try:
            self._handle_post(url, raw_body)
        except (EngineHaltedError, HealthError):
            self._send_json(503, {"error": "stopped by a safety check; restart the checks", "halted": True})
        except Exception:
            self._send_json(500, {"error": "internal error"})

    def _handle_post(self, url, raw_body):
        if url.path == "/api/v1/reset":
            # Restart the safety checks after an alarm: the pool is wiped and the start-up check runs again
            self.engine.reset_health()
            self._send_json(200, {"status": "CHECKS_RESTARTED",
                                  "startup_remaining": self.engine.get_telemetry()["startup_remaining"]})

        elif url.path == "/api/v1/swap":
            self.engine.pause_for_swap()
            self.ball_mgr.initiate_swap()
            new_ball = self.ball_mgr.complete_swap()
            if hasattr(self.sensor, "ball_id"):
                self.sensor.ball_id = new_ball.ball_id
            self.engine.resume_after_swap(new_ball.ball_id)
            self._send_json(200, {"status": "SWAP_COMPLETED", "active_ball": new_ball.ball_id})

        elif url.path == "/api/v1/config/colors":
            try:
                data = json.loads(raw_body) if raw_body else {}
            except json.JSONDecodeError:
                self._send_json(400, {"error": "Invalid JSON"})
                return
            updated = self.ball_mgr.set_ball_colors(
                color_a=data.get("color_a"),
                color_b=data.get("color_b"),
            )
            self._send_json(200, {"status": "COLORS_UPDATED", **updated})

        elif url.path == "/api/v1/toolkit":
            try:
                data = json.loads(raw_body) if raw_body else {}
            except json.JSONDecodeError:
                self._send_json(400, {"error": "Invalid JSON"})
                return
            # Use only the seed this request names; never another viewer's or a game's seed.
            seed_hex = self._valid_seed_hex(data.get("seed_hex"))
            if data.get("seed_hex") and not seed_hex:
                self._send_json(400, {"error": "seed_hex must be 32-128 hex digits"})
                return
            if not seed_hex:
                if not fill_until_ready(self.engine, self.sensor, timeout_s=1.5):
                    self._send_json(503, {"error": "not enough entropy credit yet; try again shortly"})
                    return
                seed_hex = self.engine.extract_seed(seed_bytes=32).hex()
            items = data.get("items") or ["Alice", "Bob", "Charlie", "Diana", "Echo"]
            if not isinstance(items, list) or len(items) > 1000:
                self._send_json(400, {"error": "items must be a list of at most 1,000 names"})
                return
            items = [str(i)[:100] for i in items]
            try:
                winners_count = max(1, min(len(items), int(data.get("winners_count", 1))))
            except (TypeError, ValueError):
                self._send_json(400, {"error": "winners_count must be a whole number"})
                return
            try:
                bundle = EverydaySeedToolkit.evaluate_all(seed_hex=seed_hex, items=items, winners_count=winners_count)
            except ValueError as exc:
                self._send_json(400, {"error": str(exc)})
                return
            self._send_json(200, bundle)

        elif url.path == "/api/v1/zoomies/evaluate":
            try:
                data = json.loads(raw_body) if raw_body else {}
            except json.JSONDecodeError:
                self._send_json(400, {"error": "Invalid JSON"})
                return
            try:
                res = ZoomiesScorer.evaluate_request(data)
            except ValueError as exc:
                self._send_json(400, {"error": str(exc)})
                return
            # An unlocked round is mixed into the pool without credit: human motion has not
            # been assessed, so it can add unpredictability but never counts toward a seed.
            if self.engine is not None and res["unlocked"] and res["seed_hex"]:
                self.engine.mix_uncredited(bytes.fromhex(res["seed_hex"]), label=f"zoomies_{res['form_id']}")
                res["mixed_into_console_pool"] = True
            self._send_json(200, res)

        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, format, *args):
        return


def start_server(
    engine: PetRandEngine,
    ball_mgr: TwoBallManager,
    sensor: BaseSensor,
    host: str = "127.0.0.1",
    port: int = 8080,
) -> HTTPServer:
    """Builds the PetRand console server (console, everyday studio and the Zoomies! game)."""
    PetRandServerHandler.engine = engine
    PetRandServerHandler.ball_mgr = ball_mgr
    PetRandServerHandler.sensor = sensor

    # Threaded, so one slow or idle connection cannot freeze the page; the engine has its own lock
    server = ThreadingHTTPServer((host, port), PetRandServerHandler)
    server.daemon_threads = True
    return server
