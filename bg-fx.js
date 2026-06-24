/* ===========================================================================
   PARALLED · Smart Background FX
   A site-wide, persistent background treatment + floating control.
   - One "Blur + dim" slider (blends a backdrop blur with a frosted veil)
   - One "Grain" slider (film-grain opacity)
   - On/off toggle
   - Saved to localStorage so every page in the suite remembers the setting.

   How it stays "smart": a single fixed overlay is appended to <body> at
   z-index:0. Every page paints its animated background at z-index:0 and its
   real content at z-index:1+, so the overlay sits ABOVE the background but
   BELOW the content — only the background is blurred/dimmed/grained.
   =========================================================================== */
(function () {
  if (window.__paralledBgFx) { window.__paralledBgFx.mount(); return; }

  var LS_KEY = 'paralled-bgfx-v1';
  var DEFAULTS = { on: true, intensity: 30, grain: 14 }; // gentle by default

  var GRAIN_URI =
    "url(\"data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='160' height='160'%3E%3Cfilter id='n'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.9' numOctaves='2' stitchTiles='stitch'/%3E%3C/filter%3E%3Crect width='100%25' height='100%25' filter='url(%23n)'/%3E%3C/svg%3E\")";

  function load() {
    try { return Object.assign({}, DEFAULTS, JSON.parse(localStorage.getItem(LS_KEY) || '{}')); }
    catch (e) { return Object.assign({}, DEFAULTS); }
  }
  function save() { try { localStorage.setItem(LS_KEY, JSON.stringify(state)); } catch (e) {} }

  var state = load();
  var els = {}; // cached nodes

  function isLight() {
    try { return document.documentElement.dataset.theme === 'light'; } catch (e) { return false; }
  }

  function apply() {
    var on = state.on;
    var k = state.intensity / 100;
    var blurPx = on ? (k * 22) : 0;
    var bright = on ? (1 - k * 0.55) : 1;          // dim the background
    var grainOp = on ? (state.grain / 100) * 0.5 : 0;

    // Blur + dim are applied DIRECTLY to the page's background element, so
    // foreground content (nav, headers, sections at z-index:1) is never touched.
    if (els.bg) {
      try {
        els.bg.style.filter = on ? ('blur(' + blurPx.toFixed(2) + 'px) brightness(' + bright.toFixed(3) + ')') : '';
      } catch (e) {}
    }
    if (els.grain) {
      els.grain.style.opacity = grainOp.toFixed(3);
      els.grain.style.mixBlendMode = isLight() ? 'multiply' : 'overlay';
    }

    // reflect on control
    if (els.btn) {
      els.btn.style.boxShadow = on
        ? '0 6px 22px rgba(0,0,0,.4), 0 0 0 1px rgba(94,224,200,.55), 0 0 16px rgba(94,224,200,.35)'
        : '0 6px 22px rgba(0,0,0,.4), 0 0 0 1px rgba(255,255,255,.14)';
      els.btn.style.color = on ? '#5ee0c8' : 'rgba(255,255,255,.7)';
    }
    if (els.toggle) els.toggle.checked = on;
    if (els.sIntensity) els.sIntensity.value = state.intensity;
    if (els.sGrain) els.sGrain.value = state.grain;
    if (els.vIntensity) els.vIntensity.textContent = state.intensity;
    if (els.vGrain) els.vGrain.textContent = state.grain;
    var dimWhenOff = on ? '1' : '.4';
    if (els.rows) els.rows.style.opacity = dimWhenOff;
    if (els.rows) els.rows.style.pointerEvents = on ? 'auto' : 'none';
  }

  function set(patch) { Object.assign(state, patch); save(); apply(); }

  /* ----------------------------- build DOM ------------------------------ */
  function buildGrain() {
    var grain = document.createElement('div');
    grain.id = 'paralled-bgfx-grain';
    grain.style.cssText =
      'position:fixed; inset:0; z-index:0; pointer-events:none; ' +
      'background-image:' + GRAIN_URI + '; background-size:160px 160px; ' +
      'transition:opacity .35s ease;';
    els.grain = grain;
    return grain;
  }

  function slider(label, key) {
    var wrap = document.createElement('label');
    wrap.style.cssText = 'display:block; margin-top:14px;';
    var top = document.createElement('div');
    top.style.cssText =
      'display:flex; justify-content:space-between; align-items:baseline; margin-bottom:7px; ' +
      "font-family:'JetBrains Mono',ui-monospace,monospace; font-size:10.5px; letter-spacing:.14em; " +
      'text-transform:uppercase; color:rgba(255,255,255,.6);';
    var name = document.createElement('span'); name.textContent = label;
    var val = document.createElement('span');
    val.textContent = state[key]; val.style.color = '#5ee0c8'; val.style.fontWeight = '500';
    top.appendChild(name); top.appendChild(val);

    var input = document.createElement('input');
    input.type = 'range'; input.min = '0'; input.max = '100'; input.step = '1'; input.value = state[key];
    input.style.cssText = 'width:100%; display:block; accent-color:#5ee0c8; cursor:pointer; margin:0;';
    input.addEventListener('input', function () {
      var p = {}; p[key] = parseInt(input.value, 10); set(p);
    });

    wrap.appendChild(top); wrap.appendChild(input);
    return { wrap: wrap, input: input, val: val };
  }

  function buildControl() {
    var ctrl = document.createElement('div');
    ctrl.id = 'paralled-bgfx-ctrl';
    ctrl.style.cssText =
      'position:fixed; right:20px; bottom:22px; z-index:80; ' +
      "font-family:'Manrope',system-ui,sans-serif; display:flex; flex-direction:column; " +
      'align-items:flex-end; gap:12px; pointer-events:auto;';

    /* ---- panel ---- */
    var panel = document.createElement('div');
    panel.style.cssText =
      'width:248px; padding:18px 18px 16px; border-radius:18px; ' +
      'background:rgba(10,14,19,.82); -webkit-backdrop-filter:blur(20px); backdrop-filter:blur(20px); ' +
      'border:1px solid rgba(255,255,255,.13); box-shadow:0 20px 60px rgba(0,0,0,.5); ' +
      'color:#eef2f6; transform-origin:bottom right; ' +
      'opacity:0; transform:translateY(8px) scale(.96); pointer-events:none; ' +
      'transition:opacity .26s cubic-bezier(.16,1,.3,1), transform .26s cubic-bezier(.16,1,.3,1);';

    var head = document.createElement('div');
    head.style.cssText = 'display:flex; align-items:center; justify-content:space-between; gap:10px;';
    var title = document.createElement('div');
    title.style.cssText =
      "font-family:'Space Grotesk','Manrope',sans-serif; font-size:14px; font-weight:600; letter-spacing:.01em;";
    title.textContent = 'Background blur';
    head.appendChild(title);

    // on/off switch
    var sw = document.createElement('label');
    sw.style.cssText = 'position:relative; display:inline-flex; align-items:center; cursor:pointer;';
    var toggle = document.createElement('input');
    toggle.type = 'checkbox'; toggle.checked = state.on;
    toggle.style.cssText = 'position:absolute; opacity:0; width:0; height:0;';
    var track = document.createElement('span');
    track.style.cssText =
      'width:38px; height:21px; border-radius:999px; display:inline-block; position:relative; ' +
      'transition:background .25s ease; background:' + (state.on ? '#2f9e8e' : 'rgba(255,255,255,.18)') + ';';
    var knob = document.createElement('span');
    knob.style.cssText =
      'position:absolute; top:2.5px; left:' + (state.on ? '19px' : '2.5px') + '; width:16px; height:16px; ' +
      'border-radius:50%; background:#fff; transition:left .25s cubic-bezier(.16,1,.3,1); box-shadow:0 1px 3px rgba(0,0,0,.4);';
    track.appendChild(knob); sw.appendChild(toggle); sw.appendChild(track);
    toggle.addEventListener('change', function () {
      set({ on: toggle.checked });
      track.style.background = toggle.checked ? '#2f9e8e' : 'rgba(255,255,255,.18)';
      knob.style.left = toggle.checked ? '19px' : '2.5px';
    });
    head.appendChild(sw);

    var rows = document.createElement('div');
    var s1 = slider('Blur + dim', 'intensity');
    var s2 = slider('Grain', 'grain');
    rows.appendChild(s1.wrap); rows.appendChild(s2.wrap);

    var note = document.createElement('div');
    note.style.cssText =
      "margin-top:15px; padding-top:12px; border-top:1px solid rgba(255,255,255,.09); " +
      "font-family:'JetBrains Mono',ui-monospace,monospace; font-size:9.5px; letter-spacing:.12em; " +
      'text-transform:uppercase; color:rgba(255,255,255,.34); display:flex; align-items:center; gap:7px;';
    var dot = document.createElement('span');
    dot.style.cssText = 'width:5px; height:5px; border-radius:50%; background:#5ee0c8; flex:none;';
    var noteTxt = document.createElement('span'); noteTxt.textContent = 'Saved across every page';
    note.appendChild(dot); note.appendChild(noteTxt);

    panel.appendChild(head); panel.appendChild(rows); panel.appendChild(note);

    /* ---- trigger button ---- */
    var btn = document.createElement('button');
    btn.type = 'button';
    btn.setAttribute('aria-label', 'Background blur settings');
    btn.style.cssText =
      'width:46px; height:46px; border-radius:50%; cursor:pointer; border:none; ' +
      'background:rgba(10,14,19,.82); -webkit-backdrop-filter:blur(20px); backdrop-filter:blur(20px); ' +
      'display:flex; align-items:center; justify-content:center; transition:transform .2s ease, box-shadow .3s ease, color .3s ease;';
    btn.innerHTML =
      '<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7">' +
      '<circle cx="12" cy="12" r="8.5"/><circle cx="12" cy="12" r="3.2" fill="currentColor" stroke="none"/>' +
      '<circle cx="12" cy="12" r="5.6" stroke-dasharray="1.2 3.4"/></svg>';
    btn.addEventListener('mouseenter', function () { btn.style.transform = 'scale(1.06)'; });
    btn.addEventListener('mouseleave', function () { btn.style.transform = 'scale(1)'; });

    var open = false;
    function setOpen(v) {
      open = v;
      panel.style.opacity = v ? '1' : '0';
      panel.style.transform = v ? 'translateY(0) scale(1)' : 'translateY(8px) scale(.96)';
      panel.style.pointerEvents = v ? 'auto' : 'none';
    }
    btn.addEventListener('click', function (e) { e.stopPropagation(); setOpen(!open); });
    panel.addEventListener('click', function (e) { e.stopPropagation(); });
    document.addEventListener('click', function () { if (open) setOpen(false); });

    ctrl.appendChild(panel); ctrl.appendChild(btn);

    els.ctrl = ctrl; els.btn = btn; els.toggle = toggle;
    els.sIntensity = s1.input; els.sGrain = s2.input;
    els.vIntensity = s1.val; els.vGrain = s2.val; els.rows = rows;
    return ctrl;
  }

  /* ----------------------------- mount ---------------------------------- */
  // The runtime renders the page into a [data-screen-label] root. Inside it
  // the first position:fixed/absolute child (z-index:0) is the animated
  // background; everything else (nav, headers, sections) sits at z-index>=1.
  function rootEl() {
    return document.querySelector('[data-screen-label]')
        || (document.querySelector('.sc-host') && document.querySelector('.sc-host').firstElementChild)
        || null;
  }

  // Find the page's background element: the EARLIEST fixed/absolute child of
  // the root that sits at the lowest z-index. Using strict < (not <=) keeps the
  // first such element — the real background — rather than letting a later
  // same-z content wrapper override it (which would blur/dim the whole page).
  function findBg(root) {
    var kids = root.children, best = null, bestZ = Infinity;
    for (var i = 0; i < kids.length; i++) {
      var el = kids[i];
      if (el.id === 'paralled-bgfx-grain' || el.id === 'paralled-bgfx-ctrl') continue;
      var s = getComputedStyle(el);
      if (s.position === 'fixed' || s.position === 'absolute') {
        var z = parseInt(s.zIndex, 10); if (isNaN(z)) z = 0;
        if (z < bestZ) { bestZ = z; best = el; }
      }
    }
    // Only treat it as a background if it actually sits behind content
    // (z-index <= 0). Otherwise there is no dedicated background layer and we
    // skip the blur entirely rather than risk dimming real content.
    if (best && bestZ > 0) return null;
    return best;
  }

  function mount() {
    if (!document.body) return;
    try {
      var root = rootEl();
      if (root) {
        var bg = findBg(root);
        if (bg && bg !== els.bg) {
          if (els.bg) { try { els.bg.style.filter = ''; } catch (e) {} }
          els.bg = bg;
        }
        if (!els.grain) buildGrain();
        // Grain is appended as the LAST child of the root (a sibling of the
        // background + content). Appending a TRAILING node is safe with React
        // reconciliation; inserting BETWEEN React siblings is what crashes it.
        // z-index:0 keeps grain above the background (also z0, earlier in DOM)
        // but below all content (z-index >= 1).
        if (els.grain.parentNode !== root || els.grain !== root.lastElementChild) {
          root.appendChild(els.grain);
        }
      }
      if (!document.getElementById('paralled-bgfx-ctrl')) {
        document.body.appendChild(buildControl());
      }
    } catch (e) { /* never let the effect break the page */ }
    apply();
  }

  // The page root streams in after this script runs (helmet closes early).
  // Poll briefly until the background element exists so the effect lands.
  function scheduleMount() {
    var tries = 0;
    (function tick() {
      mount();
      tries++;
      if ((!els.bg) && tries < 90) requestAnimationFrame(tick);
    })();
  }

  // keep multiple tabs / re-applied themes in sync
  window.addEventListener('storage', function (e) {
    if (e.key === LS_KEY) { state = load(); apply(); }
  });
  // themes can switch at runtime (Cosmos/Light/Alien) — re-tint grain + re-assert
  try {
    var mo = new MutationObserver(function () { mount(); });
    mo.observe(document.documentElement, { attributes: true, attributeFilter: ['data-theme'] });
  } catch (e) {}

  window.__paralledBgFx = { mount: mount, apply: apply, set: set, get state() { return state; } };

  if (document.body) scheduleMount();
  else document.addEventListener('DOMContentLoaded', scheduleMount);
  window.addEventListener('load', scheduleMount);

  // Re-assert gently after the page settles: React re-renders can replace the
  // background node or drop our trailing grain. A slow idempotent interval
  // re-finds the bg and re-appends grain if missing — no aggressive subtree
  // observer that would fight React's reconciliation mid-render.
  var ticks = 0;
  var iv = setInterval(function () {
    ticks++;
    try {
      var root = rootEl();
      if (root) {
        var bg = findBg(root);
        if (bg && bg !== els.bg) { els.bg = bg; }
        if (els.grain && (els.grain.parentNode !== root || els.grain !== root.lastElementChild)) {
          root.appendChild(els.grain);
        }
        apply();
      }
    } catch (e) {}
    if (ticks > 40) clearInterval(iv); // ~20s of settling, then stop
  }, 500);
})();
