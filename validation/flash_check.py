#!/usr/bin/env python3
"""validation/flash_check.py: measure flicker on PetRand's two pages with the WCAG 2.2 general flash rule.

A flash is a pair of opposing changes of at least 10% of maximum relative luminance, where the darker
image is below 0.80. A page fails if more than 3 flashes happen in some one-second span over more than
25% of a 10 degree visual field, or if the whole frame's mean luminance flashes more than 3 times a second.

How it measures (rebuilt 2026-11-08 after a review showed the first version had blind spots):
  - Frames come from Chrome's own screencast, so page and canvas alike are measured as drawn. The
    screencast sends a frame whenever something changes, so a still screen is not a gap.
  - Each frame is reduced to 2x2-pixel patches of linear luminance, and every patch is tracked on its
    own, so fine flashing stripes are seen (the old version averaged 22x25-pixel cells).
  - A window the size of a 10 degree field slides across the screen in 2-pixel steps (desktop: 341x256,
    WCAG's own estimate; phone: half the width by a quarter of the height, a stricter stand-in), and the
    worst window counts. The old version used fixed tiles, so a patch on a tile corner could pass.
  - Patches count as flashing if they ever pass 3 flashes in a second, whether or not at the same moment
    as their neighbors, which overstates the area: a strict screen.
  - `--self-test` runs pages that flash on purpose, plus controls, and checks each verdict.
A strict screen, not a certificate: it has no red-flash test, and only a person can judge comfort.

Brutal cases: the game driven by synthetic pointer events every 8 ms at random spots (the most particles it
will draw) with key mashing, through the unlock and the next round; a rhythmic loop that trips the
"Too rhythmic" warning (its toggles per second are counted separately); the console with "New random
number" clicked as fast as it will take, and with Under the hood open.

Run: python3 validation/flash_check.py [--self-test]   (exit 1 if any case fails; JSON lines to stdout)
"""
import asyncio
import os
import base64
import io
import json
import sys
import threading
import time
from pathlib import Path

import numpy as np
from PIL import Image
from playwright.async_api import async_playwright

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from petrand.engine import PetRandEngine  # noqa: E402
from petrand.lifecycle import TwoBallManager  # noqa: E402
from petrand.sensors import SyntheticArenaSimulator  # noqa: E402
from petrand.server import start_server  # noqa: E402
from petrand.zoomies import start_zoomies_server  # noqa: E402

RECORD_MS = 9000
MIN_FPS = 20             # below this the recording could miss a 3-per-second flash
# Chrome's screencast sends a frame only when something on screen changes, so a low rate on a still
# screen is not a gap. The rate is checked only in cases where something on screen moves every frame.
PATCH = 2                # analysis patch size in CSS pixels
FIELD_DESKTOP = (341, 256)   # a 10 degree field at 1024 x 768, WCAG's own estimate
FIELD_PHONE = (188, 203)     # half a 375 px phone screen by a quarter of its height (stricter)
_LIN = np.array([(c / 255) / 12.92 if (c / 255) <= 0.04045 else (((c / 255) + 0.055) / 1.055) ** 2.4
                 for c in range(256)], dtype=np.float32)


def to_lum(jpeg_b64):
    """Decode one frame to linear relative luminance averaged over PATCH x PATCH pixel patches."""
    img = np.asarray(Image.open(io.BytesIO(base64.b64decode(jpeg_b64))).convert('RGB'))
    lin = _LIN[img]
    lum = lin[..., 0] * 0.2126 + lin[..., 1] * 0.7152 + lin[..., 2] * 0.0722
    h, w = (lum.shape[0] // PATCH) * PATCH, (lum.shape[1] // PATCH) * PATCH
    return lum[:h, :w].reshape(h // PATCH, PATCH, w // PATCH, PATCH).mean(axis=(1, 3))


def flashes_per_second(series, times):
    """Worst one-second count of opposing 10% pairs in one luminance series (times in ms)."""
    turns, ref, direction = [], series[0], 0
    for v, t in zip(series, times):
        if direction >= 0 and v - ref >= 0.1 and min(v, ref) < 0.8:
            turns.append(t); direction = -1; ref = v
        elif direction <= 0 and ref - v >= 0.1 and min(v, ref) < 0.8:
            turns.append(t); direction = 1; ref = v
        else:
            ref = max(ref, v) if direction == -1 else min(ref, v) if direction == 1 else ref
    best = 0
    for i, t in enumerate(turns):
        n = sum(1 for x in turns[i:] if x < t + 1000)
        best = max(best, n // 2)  # two opposing changes make one flash
    return best


def flash_map(frames):
    """Per patch: True if it ever makes more than 3 flashes (8 opposing changes) within one second.
    The same rule as flashes_per_second, run on every patch at once."""
    t0, first = frames[0]
    ref = first.copy()
    direction = np.zeros(first.shape, np.int8)
    ring = np.full((8,) + first.shape, -1e18)       # times of each patch's last 8 changes
    idx = np.zeros(first.shape, np.int64)
    flagged = np.zeros(first.shape, bool)
    for t, v in frames[1:]:
        lo = np.minimum(v, ref)
        up = (direction >= 0) & (v - ref >= 0.1) & (lo < 0.8)
        down = ~up & (direction <= 0) & (ref - v >= 0.1) & (lo < 0.8)
        turn = up | down
        ref = np.where(~turn & (direction == -1), np.maximum(ref, v), ref)
        ref = np.where(~turn & (direction == 1), np.minimum(ref, v), ref)
        if turn.any():
            ys, xs = np.nonzero(turn)
            slot = idx[ys, xs]
            ring[slot, ys, xs] = t
            idx[ys, xs] = (slot + 1) % 8
            oldest = ring[idx[ys, xs], ys, xs]       # after the write, the next slot holds the oldest
            flagged[ys, xs] |= (t - oldest) < 1000.0
            direction = np.where(up, -1, np.where(down, 1, direction)).astype(np.int8)
            ref = np.where(turn, v, ref)
    return flagged


def worst_window(flagged, win_w, win_h):
    """Largest share of flashing patches in any win_w x win_h (patch units) window."""
    h, w = flagged.shape
    win_w, win_h = max(1, min(win_w, w)), max(1, min(win_h, h))
    ii = np.zeros((h + 1, w + 1))
    ii[1:, 1:] = flagged.astype(np.float64).cumsum(0).cumsum(1)
    sums = ii[win_h:, win_w:] - ii[:-win_h, win_w:] - ii[win_h:, :-win_w] + ii[:-win_h, :-win_w]
    return float(sums.max()) / (win_w * win_h)


def analyze(frames, field):
    """frames: list of (t_ms, patch luminance). field: (width, height) of a 10 degree field in CSS px."""
    times = [t for t, _ in frames]
    flagged = flash_map(frames)
    worst = worst_window(flagged, field[0] // PATCH, field[1] // PATCH)
    means = [float(g.mean()) for _, g in frames]
    span_s = (times[-1] - times[0]) / 1000 if len(times) > 1 else 0
    return {
        'frames': len(frames),
        'fps': round(len(frames) / span_s, 1) if span_s else 0,
        'worstWindowOver3PerSec': round(worst, 3),
        'flashingPatchesPct': round(100.0 * float(flagged.mean()), 2),
        'wholeFrame': flashes_per_second(means, times),
        'meanLuminancePct': [round(min(means) * 100, 1), round(max(means) * 100, 1)],
    }


async def record(page, ms, during=None):
    """Record the page with Chrome's screencast for ms milliseconds; run `during` (a coroutine) meanwhile."""
    cdp = await page.context.new_cdp_session(page)
    raw = []

    async def ack(session_id):
        try:
            await cdp.send('Page.screencastFrameAck', {'sessionId': session_id})
        except Exception:
            pass  # a late frame after the recording stopped

    def on_frame(p):
        raw.append((p['metadata']['timestamp'] * 1000.0, p['data']))
        asyncio.ensure_future(ack(p['sessionId']))

    cdp.on('Page.screencastFrame', on_frame)
    vp = page.viewport_size
    await cdp.send('Page.startScreencast', {'format': 'jpeg', 'quality': 80, 'maxWidth': vp['width'],
                                             'maxHeight': vp['height'], 'everyNthFrame': 1})
    task = asyncio.ensure_future(during()) if during else None
    await page.wait_for_timeout(ms)
    await cdp.send('Page.stopScreencast')
    if task:
        task.cancel()
        try:
            await task
        except (asyncio.CancelledError, Exception):
            pass
    try:
        await cdp.detach()
    except Exception:
        pass
    return [(t, to_lum(d)) for t, d in raw]


# In-page drivers -----------------------------------------------------------------------------------------
BRUTAL_JS = """() => {
  const cv = document.querySelector('canvas');
  const r = cv.getBoundingClientRect();
  let k = 0;
  window.__brutal = setInterval(() => {
    for (let n = 0; n < 2; n++) {
      cv.dispatchEvent(new MouseEvent('mousemove', { bubbles: true, buttons: 1,
        clientX: r.left + Math.random() * r.width, clientY: r.top + Math.random() * r.height }));
    }
    window.dispatchEvent(new KeyboardEvent('keydown', { keyCode: 65 + (k++ % 26) }));
  }, 8);
}"""

RHYTHM_JS = """() => {
  const cv = document.querySelector('canvas');
  const r = cv.getBoundingClientRect();
  let a = 0;
  window.__rhythm = setInterval(() => {
    a += 0.35;   // a steady circle, the pattern the rhythm detector is there to catch
    cv.dispatchEvent(new MouseEvent('mousemove', { bubbles: true, buttons: 1,
      clientX: r.left + r.width / 2 + Math.cos(a) * r.width * 0.3,
      clientY: r.top + r.height / 2 + Math.sin(a) * r.height * 0.3 }));
  }, 16);
  const w = document.getElementById('warningToast');
  window.__toggles = [];
  new MutationObserver(() => window.__toggles.push([performance.now(), w.style.display])).observe(w, { attributes: true });
}"""


async def game_cases(page):
    runs = []
    runs.append(('idle', await record(page, 4000), {}, True))

    async def brutal_round():
        await page.evaluate(BRUTAL_JS)
        while True:  # keep starting rounds: the jump, its unlock, the next move, the next jump
            btn = page.locator('#btnStartJump')
            if await btn.is_enabled():
                await btn.click()
            await page.wait_for_timeout(4600)
            nxt = page.get_by_role('button', name='Next move')
            if await nxt.count() and await nxt.first.is_enabled():
                await nxt.first.click()

    runs.append(('brutal motion, rounds back to back', await record(page, RECORD_MS, brutal_round), {}, True))
    await page.evaluate('clearInterval(window.__brutal)')
    await page.wait_for_timeout(1500)

    async def rhythmic():
        await page.evaluate(RHYTHM_JS)
        await page.locator('#btnStartJump').click()
        await page.wait_for_timeout(RECORD_MS)

    frames = await record(page, 4800, rhythmic)
    toggles = await page.evaluate('window.__toggles || []')
    await page.evaluate('clearInterval(window.__rhythm)')
    shows = [t for t, d in toggles if d == 'block']
    worst = max((sum(1 for x in shows if s <= x < s + 1000) for s in shows), default=0)
    runs.append(('rhythmic loop', frames, {'warningShownCount': len(shows), 'warningShowsPerSecondWorst': worst}, True))
    return runs


async def console_cases(page):
    runs = []
    await page.locator('#arena-canvas').scroll_into_view_if_needed()
    runs.append(('idle, arena in view', await record(page, 6000), {}, True))
    await page.evaluate('window.scrollTo(0, 0)')

    async def rapid():
        btn = page.locator('#btn-extract-seed')
        while True:
            if await btn.is_enabled():
                await btn.click(no_wait_after=True)
            await page.wait_for_timeout(120)

    runs.append(('New random number, as fast as it takes', await record(page, RECORD_MS, rapid), {}, False))
    await page.evaluate("document.getElementById('hood').open = true")
    await page.locator('#hood').scroll_into_view_if_needed()
    runs.append(('Under the hood open', await record(page, 6000), {}, False))
    return runs


def start_servers():
    engine = PetRandEngine(required_entropy_bits=64.0, require_startup_test=False)
    mgr = TwoBallManager(ball_a_id='Ball_1', ball_b_id='Ball_2')
    sim = SyntheticArenaSimulator(ball_id='Ball_1', ball_manager=mgr)
    while not engine.is_ready:
        ev = sim.poll()
        if ev:
            engine.ingest_event(ev)
    stop = threading.Event()

    def harvest():
        while not stop.is_set():
            ev = sim.poll()
            if ev:
                engine.ingest_event(ev)
            time.sleep(0.005)

    threading.Thread(target=harvest, daemon=True).start()
    dash = start_server(engine, mgr, sim, host='127.0.0.1', port=0)
    threading.Thread(target=dash.serve_forever, daemon=True).start()
    vj = start_zoomies_server(host='127.0.0.1', port=0,
                                engine=PetRandEngine(required_entropy_bits=64.0, require_startup_test=False))
    threading.Thread(target=vj.serve_forever, daemon=True).start()
    return stop, dash, vj


async def main():
    stop, dash, vj = start_servers()
    pages = [('console', f'http://127.0.0.1:{dash.server_address[1]}/', console_cases),
             ('game', f'http://127.0.0.1:{vj.server_address[1]}/', game_cases)]
    only = os.environ.get('FLASH_PAGES')   # e.g. FLASH_PAGES=game for a quicker rerun of one page
    if only:
        pages = [pg for pg in pages if pg[0] in only.split(',')]
    views = [('desktop', {'width': 1440, 'height': 900}, FIELD_DESKTOP),
             ('phone', {'width': 375, 'height': 812}, FIELD_PHONE)]
    bad, low_fps = [], []
    async with async_playwright() as p:
        try:
            b = await p.chromium.launch(channel='chrome', headless=True,
                                        args=['--enable-gpu', '--use-angle=metal', '--ignore-gpu-blocklist'])
        except Exception:
            b = await p.chromium.launch(headless=True)
        for name, url, cases in pages:
            for view, vp, field in views:
                for motion in ('no-preference', 'reduce'):
                    ctx = await b.new_context(viewport=vp, device_scale_factor=1, reduced_motion=motion,
                                              is_mobile=(view == 'phone'), has_touch=(view == 'phone'))
                    page = await ctx.new_page()
                    await page.goto(url)
                    await page.wait_for_timeout(1500)
                    for run in await cases(page):
                        label, frames, extra, animating = run
                        if len(frames) < 2:
                            # the screencast only sends a frame when something changes: none means nothing moved
                            res = {'frames': len(frames), 'note': 'no change on screen during the recording'}
                        else:
                            res = analyze(frames, field)
                            if animating and res['fps'] < MIN_FPS:
                                low_fps.append(f'{name} {view} {motion} {label}: {res["fps"]} fps')
                        res.update(extra)
                        print(json.dumps({'page': name, 'view': view, 'motion': motion, 'case': label, **res}))
                        tag = f'{name} {view} {motion} "{label}"'
                        if res.get('worstWindowOver3PerSec', 0) > 0.25:
                            bad.append(f'{tag}: {res["worstWindowOver3PerSec"]:.0%} of a 10-degree field flashes more than 3 times a second')
                        if res.get('wholeFrame', 0) > 3:
                            bad.append(f'{tag}: the whole frame flashes {res["wholeFrame"]} times in a second')
                        if extra.get('warningShowsPerSecondWorst', 0) > 3:
                            bad.append(f'{tag}: the warning appears {extra["warningShowsPerSecondWorst"]} times in a second')
                    await ctx.close()
        await b.close()
    stop.set()
    dash.shutdown()
    vj.shutdown()
    if low_fps:
        print('recorded below', MIN_FPS, 'fps (a flash could hide between frames):', '; '.join(low_fps))
    print('\n'.join(bad) or 'flicker check passed')
    return 1 if (bad or low_fps) else 0


# --- Self-test: pages that flash on purpose, and controls --------------------------------------------------
def _flash_page(hz, css_on, css_off, box):
    x, y, w, h = box
    return f"""<html><body style="margin:0;background:#000;width:1440px;height:900px;overflow:hidden">
<div id="f" style="position:absolute;left:{x}px;top:{y}px;width:{w}px;height:{h}px;background:{css_off}"></div>
<script>(() => {{ const el = document.getElementById('f'); const t0 = performance.now(); const half = 500 / {hz};
function tick(now) {{ el.style.background = (Math.floor((now - t0) / half) % 2) ? '{css_on}' : '{css_off}';
requestAnimationFrame(tick); }} requestAnimationFrame(tick); }})();</script></body></html>"""


SELF_TESTS = [
    # name, page, should fail
    ('square on an old tile corner, 5 Hz', _flash_page(5, '#fff', '#000', (310, 194, 340, 212)), True),
    ('fine stripes (10 px of 25), 6 Hz, luminance 0 to 0.24',
     _flash_page(6, 'repeating-linear-gradient(to bottom, rgb(134,134,134) 0 10px, #000 10px 25px)', '#000',
                 (400, 250, 600, 400)), True),
    ('whole screen, 5 Hz', _flash_page(5, '#fff', '#000', (0, 0, 1440, 900)), True),
    ('whole screen, 2 Hz (allowed)', _flash_page(2, '#fff', '#000', (0, 0, 1440, 900)), False),
    ('small square 100 x 100, 10 Hz (allowed: 11% of a field)', _flash_page(10, '#fff', '#000', (600, 400, 100, 100)), False),
]


async def self_test():
    wrong = []
    async with async_playwright() as p:
        try:
            b = await p.chromium.launch(channel='chrome', headless=True,
                                        args=['--enable-gpu', '--use-angle=metal', '--ignore-gpu-blocklist'])
        except Exception:
            b = await p.chromium.launch(headless=True)
        ctx = await b.new_context(viewport={'width': 1440, 'height': 900}, device_scale_factor=1)
        for name, html, should_fail in SELF_TESTS:
            page = await ctx.new_page()   # a fresh page per test, so no script state carries over
            await page.set_content(html)
            await page.wait_for_timeout(300)
            res = analyze(await record(page, 4000), FIELD_DESKTOP)
            await page.close()
            failed = res['worstWindowOver3PerSec'] > 0.25 or res['wholeFrame'] > 3
            print(json.dumps({'self_test': name, 'expected': 'fail' if should_fail else 'pass',
                              'got': 'fail' if failed else 'pass', **res}))
            if failed != should_fail:
                wrong.append(name)
        await b.close()
    print('self-test passed' if not wrong else 'self-test FAILED: ' + '; '.join(wrong))
    return 1 if wrong else 0


if __name__ == '__main__':
    sys.exit(asyncio.run(self_test() if '--self-test' in sys.argv else main()))
