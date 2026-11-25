// Coded graphics. Each <div class="figure" data-figure="name"> is filled with an
// SVG drawn by FIGURES[name](svg, w, h, div). Coordinates are canvas px: the
// default figure box is 1700 × 750 (see .figure in theme.css); data-w / data-h
// change it. Groups made with frag() step in with the arrow keys. A figure that
// returns { start, stop } is animated only while its slide is showing.

(function () {
  const NS = 'http://www.w3.org/2000/svg';
  const W = 1700, H = 750;

  function el(tag, attrs = {}, parent) {
    const n = document.createElementNS(NS, tag);
    for (const [k, v] of Object.entries(attrs)) n.setAttribute(k, v);
    if (parent) parent.append(n);
    return n;
  }
  function text(parent, x, y, str, attrs = {}) {
    const t = el('text', { x, y, ...attrs }, parent);
    t.textContent = str;
    return t;
  }
  function frag(parent, index) {
    const g = el('g', { class: 'fragment' }, parent);
    if (index !== undefined) g.setAttribute('data-fragment-index', index);
    return g;
  }
  const polyline = (parent, pts, attrs) =>
    el('polyline', { points: pts.map(p => p.join(',')).join(' '), fill: 'none', ...attrs }, parent);

  // Run step(dt) every frame while the figure's slide is showing
  function animate(step) {
    let raf = 0, last = 0;
    const tick = now => { step(Math.min(0.05, (now - last) / 1000)); last = now; raf = requestAnimationFrame(tick); };
    return {
      start() { if (!raf) { last = performance.now(); raf = requestAnimationFrame(tick); } },
      stop() { cancelAnimationFrame(raf); raf = 0; },
    };
  }
  // n click-steps owned by this figure: onStep(k) gets the number of steps shown so far
  function steps(svg, n, onStep) {
    const marks = Array.from({ length: n }, () => frag(svg));
    const update = () => onStep(marks.filter(m => m.classList.contains('visible')).length);
    Reveal.on('fragmentshown', update); Reveal.on('fragmenthidden', update); Reveal.on('ready', update);
    onStep(0);
  }

  const FIGURES = {};

  // Wide → narrow → wide, four bands of half-width hw and height dy, labels to the right
  function funnel(svg, labels, hw, dy, size, lit) {
    const widths = [1, 0.42, 0.14, 0.42, 1];
    labels.forEach((label, i) => {
      const y0 = i * dy, y1 = y0 + dy - size / 6, a = widths[i] * hw, b = widths[i + 1] * hw, on = i === lit;
      el('path', { d: `M${hw - a},${y0} H${hw + a} L${hw + b},${y1} H${hw - b} Z`,
        fill: on ? 'var(--warm)' : 'none', stroke: on ? 'var(--warm)' : 'var(--ink-faint)', 'stroke-width': 3 }, svg);
      text(svg, 2 * hw + size, (y0 + y1) / 2 + size / 3, label, { 'font-size': size, fill: on ? 'var(--warm)' : 'var(--ink-dim)' });
    });
  }

  // "You are here" map, with data-at's section lit
  FIGURES['hourglass'] = (svg, w, h, div) => {
    const parts = ['Motivation', 'Theory', 'Results', 'Conclusion'];
    funnel(svg, parts, 55, h / 4, 26, parts.findIndex(p => p.toLowerCase() === div.dataset.at));
  };

  // The shape of a talk
  FIGURES['funnel'] = (svg, w, h) =>
    funnel(svg, ['a hook the room cares about', 'the problem', 'your contribution', 'what it changes'], 190, h / 4, 36);

  // Example to copy: a curve that draws itself, then a label that steps in on a click
  FIGURES['example'] = (svg, w, h) => {
    const n = 200, pts = Array.from({ length: n + 1 }, (_, i) => [40 + i / n * (w - 80), h / 2 - 0.35 * h * Math.sin(i / n * 4 * Math.PI)]);
    const line = polyline(svg, [], { stroke: 'var(--cool)', 'stroke-width': 5 });
    text(frag(svg), 40, 60, 'a label on the first click', { 'font-size': 44, class: 'warm' });
    let k = 0;
    return animate(dt => {
      k = Math.min(n + 1, k + 120 * dt);
      line.setAttribute('points', pts.slice(0, Math.ceil(k)).map(p => p.join(',')).join(' '));
    });
  };

  // Icons for the four CFD headaches (slide 2), each 360 × 190
  const fish = (svg, cx, cy, rx, ry, cls = 'hull') => el('ellipse', { cx, cy, rx, ry, class: cls }, svg);
  // A body-fitted triangle mesh grown out from a nearly closed jelly bell: the layers
  // from the two lips and the cavity walls run into each other and tangle
  FIGURES['pain-mesh'] = (svg, w, h) => {
    const cx = w / 2, cy = h / 2 + 8, a = 44, b = 52, t = 7, φm = 2.8, n = 30;
    // a prolate bell, open at the bottom, its margin curling in towards the axis
    const arc = (s, φ) => { const curl = 1 - 0.25 * Math.max(0, Math.abs(φ) - 2) ** 2;
      return [cx + s * a * curl * Math.sin(φ), cy - s * b * Math.cos(φ)]; };
    const outer = Array.from({ length: n + 1 }, (_, i) => arc(1, -φm + 2 * φm * i / n));
    const inner = Array.from({ length: n + 1 }, (_, i) => arc(1 - t / a, φm - 2 * φm * i / n));
    const body = [...outer, ...inner];   // closed outline, clockwise on screen
    const m = body.length;
    const normal = i => {
      const [x0, y0] = body[(i - 1 + m) % m], [x1, y1] = body[(i + 1) % m];
      const dx = x1 - x0, dy = y1 - y0, l = Math.hypot(dx, dy);
      return [dy / l, -dx / l];
    };
    const inside = ([x, y]) => {   // ray casting
      let c = false;
      for (let i = 0, j = m - 1; i < m; j = i++) {
        const [xi, yi] = body[i], [xj, yj] = body[j];
        if ((yi > y) !== (yj > y) && x < (xj - xi) * (y - yi) / (yj - yi) + xi) c = !c;
      }
      return c;
    };
    // Points: the body, two thin layers off it, then a fill that coarsens away from the bell
    const pts = [...body];
    [2.5, 6].forEach(d => body.forEach((p, i) => { const [nx, ny] = normal(i); pts.push([p[0] + d * nx, p[1] + d * ny]); }));
    let seed = 7; const rnd = () => (seed = (seed * 16807) % 2147483647) / 2147483647;
    for (let k = 0; k < 2600 && pts.length < 520; k++) {
      const p = [rnd() * w, rnd() * h], r = Math.hypot((p[0] - cx) / a, (p[1] - cy) / b);
      if (rnd() < Math.min(1, 0.5 / (r * r))) pts.push(p);
    }
    for (let i = 0; i <= 12; i++) pts.push([i * w / 12, 0], [i * w / 12, h]);
    for (let j = 1; j < 6; j++) pts.push([0, j * h / 6], [w, j * h / 6]);
    const keep = pts.filter((p, i) => i < m || (!inside(p) && p[0] >= 0 && p[0] <= w && p[1] >= 0 && p[1] <= h));
    const g = el('g', { stroke: 'var(--ink-dim)', 'stroke-width': 0.8, 'stroke-opacity': 0.8, fill: 'none' }, svg);
    delaunay(keep).forEach(([p, q, r]) => {
      const c = [(p[0] + q[0] + r[0]) / 3, (p[1] + q[1] + r[1]) / 3];
      if (!inside(c)) el('polygon', { points: [p, q, r].map(v => v.join(',')).join(' ') }, g);
    });
    el('polygon', { points: body.map(v => v.join(',')).join(' '), class: 'hull' }, svg);
  };
  // Bowyer–Watson Delaunay triangulation of a few hundred points
  function delaunay(pts) {
    const big = 1e5, sup = [[-big, -big], [big, -big], [0, big]];
    let tris = [[...sup]];
    const circ = ([a, b, c]) => {
      const d = 2 * (a[0] * (b[1] - c[1]) + b[0] * (c[1] - a[1]) + c[0] * (a[1] - b[1]));
      const s = p => p[0] * p[0] + p[1] * p[1];
      const x = (s(a) * (b[1] - c[1]) + s(b) * (c[1] - a[1]) + s(c) * (a[1] - b[1])) / d;
      const y = (s(a) * (c[0] - b[0]) + s(b) * (a[0] - c[0]) + s(c) * (b[0] - a[0])) / d;
      return [x, y, (a[0] - x) ** 2 + (a[1] - y) ** 2];
    };
    pts.forEach(p => {
      const bad = tris.filter(t => { const [x, y, r2] = circ(t); return (p[0] - x) ** 2 + (p[1] - y) ** 2 < r2; });
      const edges = [];
      bad.forEach(t => [[t[0], t[1]], [t[1], t[2]], [t[2], t[0]]].forEach(e => {
        const k = edges.findIndex(f => (f[0] === e[1] && f[1] === e[0]) || (f[0] === e[0] && f[1] === e[1]));
        k < 0 ? edges.push(e) : edges.splice(k, 1);
      }));
      tris = tris.filter(t => !bad.includes(t));
      edges.forEach(([u, v]) => tris.push([u, v, p]));
    });
    return tris.filter(t => !t.some(v => sup.includes(v)));
  }
  FIGURES['pain-model'] = (svg, w, h) => {   // which closure?
    [['k–ε', 40, 60], ['k–ω SST', 170, 40], ['Spalart–Allmaras', 30, 140], ['DES?', 250, 120]].forEach(([s, x, y]) =>
      text(svg, x, y, s, { 'font-size': 30, class: 'dim', 'font-style': 'italic' }));
    text(svg, w / 2 - 14, h / 2 + 24, '?', { 'font-size': 90, class: 'warm' });
  };
  FIGURES['pain-domain'] = (svg, w, h) => {   // a huge box around a tiny body
    el('rect', { x: 4, y: 4, width: w - 8, height: h - 8, class: 'axis' }, svg);
    fish(svg, 70, h / 2, 12, 5);
    el('line', { x1: 100, y1: h / 2, x2: w - 14, y2: h / 2, class: 'dim-line', 'stroke-dasharray': '8 6' }, svg);
    text(svg, 140, h / 2 - 14, '≈ 50 body lengths', { 'font-size': 26, class: 'dim' });
  };
  FIGURES['pain-time'] = (svg, w, h) => {   // an hourglass
    const cx = w / 2;
    el('path', { d: `M${cx - 55},12 H${cx + 55} L${cx + 6},${h / 2} L${cx + 55},${h - 12} H${cx - 55} L${cx - 6},${h / 2} Z`, class: 'axis' }, svg);
    el('path', { d: `M${cx - 32},${h - 14} H${cx + 32} L${cx},${h - 60} Z`, class: 'warm' }, svg);
    text(svg, cx + 80, h / 2 + 12, 'days', { 'font-size': 34, class: 'dim' });
  };

  // <ol class="titles">: the slide titles marked data-story, so it stays in step with edits
  document.querySelectorAll('.titles').forEach(ol => document.querySelectorAll('[data-story]').forEach(h => {
    const c = h.cloneNode(true);
    c.querySelectorAll('.source, .opt').forEach(n => n.remove());
    ol.append(Object.assign(document.createElement('li'), { innerHTML: c.innerHTML.trim() }));
  }));

  const controls = [];
  document.querySelectorAll('.figure[data-figure]').forEach(div => {
    const draw = FIGURES[div.dataset.figure];
    if (!draw) { div.textContent = `unknown figure: ${div.dataset.figure}`; return; }
    const w = +div.dataset.w || W, h = +div.dataset.h || H;
    const svg = el('svg', { viewBox: `0 0 ${w} ${h}`, 'aria-hidden': 'true' }, div);
    const ctl = draw(svg, w, h, div);
    if (div.closest('.steps')) svg.querySelectorAll('.fragment').forEach(f => f.classList.remove('fragment'));   // thumbnails show every step
    if (ctl) controls.push([div, ctl]);
  });

  // Figures with script-driven animation run only while their slide is showing.
  function sync() {
    const cur = Reveal.getCurrentSlide();
    controls.forEach(([div, ctl]) => (cur && cur.contains(div) ? ctl.start() : ctl.stop()));
  }
  Reveal.on('ready', sync);
  Reveal.on('slidechanged', sync);
})();
