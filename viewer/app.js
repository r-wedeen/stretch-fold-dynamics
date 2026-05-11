(function () {
  "use strict";

  const data = window.TRAJECTORY;
  if (!data) {
    document.body.innerHTML = "<main class='app'><h1>No trajectory data</h1><p>Run scripts/train_export.py first.</p></main>";
    return;
  }

  const foldCanvas = document.getElementById("foldCanvas");
  const lossScrubCanvas = document.getElementById("lossScrubCanvas");
  const playButton = document.getElementById("playButton");
  const timeSlider = document.getElementById("timeSlider");
  const stepReadout = document.getElementById("stepReadout");
  const lossReadout = document.getElementById("lossReadout");

  const snapshots = data.snapshots;
  const x = data.x;
  const target = data.target;
  const targetDomain = paddedExtent(target);
  const xDomain = horizontalAxisDomain();
  const outputDomain = paddedExtent(target.concat(snapshots.flatMap((snap) => snap.pred)));
  const playbackStartStep = 0;
  const playbackEndStep = Math.min(300, data.meta.steps);
  const playbackStepMs = 13 * (1000 / playbackEndStep);
  let frame = 0;
  let playing = false;
  let lastTick = 0;
  let playbackStep = playbackStartStep;

  timeSlider.max = String(playbackEndStep);

  function paddedExtent(values) {
    let min = Infinity;
    let max = -Infinity;
    for (const value of values) {
      if (value < min) min = value;
      if (value > max) max = value;
    }
    if (!Number.isFinite(min) || !Number.isFinite(max)) return [-1, 1];
    const pad = Math.max((max - min) * 0.08, 0.1);
    return [min - pad, max + pad];
  }

  function tightExtent(values) {
    let min = Infinity;
    let max = -Infinity;
    for (const value of values) {
      if (value < min) min = value;
      if (value > max) max = value;
    }
    if (!Number.isFinite(min) || !Number.isFinite(max)) return [-1, 1];
    if (Math.abs(max - min) < 1e-8) return [min - 1, max + 1];
    return [min, max];
  }

  function horizontalAxisDomain() {
    let min = Infinity;
    let max = -Infinity;
    function observe(values) {
      for (const value of values) {
        if (value < min) min = value;
        if (value > max) max = value;
      }
    }

    observe(x);
    observe(target);
    for (const snap of snapshots) {
      for (const layer of snap.layers) observe(layer);
      observe(snap.pred);
    }

    if (!Number.isFinite(min) || !Number.isFinite(max)) return tightExtent(x);
    const pad = Math.max((max - min) * 0.04, 0.1);
    return [min - pad, max + pad];
  }

  function fitCanvas(canvas) {
    const rect = canvas.getBoundingClientRect();
    const scale = window.devicePixelRatio || 1;
    const width = Math.max(1, Math.round(rect.width * scale));
    const height = Math.max(1, Math.round(rect.height * scale));
    if (canvas.width !== width || canvas.height !== height) {
      canvas.width = width;
      canvas.height = height;
    }
    const ctx = canvas.getContext("2d");
    ctx.setTransform(scale, 0, 0, scale, 0, 0);
    return { ctx, width: rect.width, height: rect.height };
  }

  function lerp(a, b, t) {
    return a + (b - a) * t;
  }

  function mapValue(value, domain, range) {
    const t = (value - domain[0]) / (domain[1] - domain[0]);
    return lerp(range[0], range[1], t);
  }

  function paletteColor(value, domain, saturation) {
    const t = Math.max(0, Math.min(1, (value - domain[0]) / (domain[1] - domain[0])));
    const hue = lerp(218, 14, t);
    const light = lerp(36, 56, Math.sin(t * Math.PI));
    return `hsl(${hue} ${saturation}% ${light}%)`;
  }

  function targetColorFor(index) {
    return paletteColor(target[index], targetDomain, 74);
  }

  function colorFor(index) {
    return targetColorFor(index);
  }

  function rowColor(row, index, snap) {
    if (row.kind === "identity") return paletteColor(row.values[index], targetDomain, 74);
    return colorFor(index);
  }

  function formatNumber(value) {
    const abs = Math.abs(value);
    if (abs >= 10) return value.toFixed(1);
    if (abs >= 1) return value.toFixed(2);
    return value.toFixed(3);
  }

  function formatFoldLabel(a, b) {
    const displayA = a < 0 ? -a : a;
    const displayB = a < 0 ? -b : b;
    const sign = displayB < 0 ? "-" : "+";
    return `|${formatNumber(displayA)}x ${sign} ${formatNumber(Math.abs(displayB))}|`;
  }

  function formatAffineLabel(c, d) {
    const sign = d < 0 ? "-" : "+";
    return `${formatNumber(c)}x ${sign} ${formatNumber(Math.abs(d))}`;
  }

  function compactRowLabel(label) {
    const hidden = label.match(/^hidden layer (\d+)$/);
    if (hidden) return `h${hidden[1]}`;
    if (label === "identity colors") return "identity";
    return label;
  }

  function rowsForSnapshot(snap) {
    const rows = [{ label: "input", values: snap.layers[0] }];
    for (let idx = 0; idx < snap.layers.length - 1; idx += 1) {
      rows.push({
        label: `hidden layer ${idx + 1}`,
        values: snap.layers[idx + 1],
      });
    }
    rows.push({ label: "output", values: snap.pred });
    rows.push({
      kind: "identity",
      label: "identity colors",
      values: x.map((_, index) => mapValue(index, [0, x.length - 1], targetDomain)),
    });
    return rows;
  }

  function transformLabelsForSnapshot(snap) {
    const labels = [];
    for (let idx = 0; idx < snap.params.a.length; idx += 1) {
      labels.push({ row: idx + 0.5, label: formatFoldLabel(snap.params.a[idx], snap.params.b[idx]) });
    }
    labels.push({ row: snap.params.a.length + 0.5, label: formatAffineLabel(snap.params.c[0], snap.params.d[0]) });
    return labels;
  }

  function drawGrid(ctx, width, height, left, right, top, bottom, rows, snap) {
    ctx.save();
    ctx.strokeStyle = "#e1e4ea";
    ctx.lineWidth = 1;
    ctx.fillStyle = "#62666f";
    ctx.font = "12px ui-sans-serif, system-ui";
    ctx.textAlign = "left";
    for (let l = 0; l < rows.length; l += 1) {
      const yPos = mapValue(l, [0, rows.length - 1], [top, bottom]);
      ctx.beginPath();
      ctx.moveTo(left, yPos);
      ctx.lineTo(right, yPos);
      ctx.stroke();
      ctx.fillText(compactRowLabel(rows[l].label), 8, yPos + 4);
    }
    const transforms = transformLabelsForSnapshot(snap);
    const arrowX = 16;
    ctx.strokeStyle = "rgba(155, 63, 79, 0.28)";
    ctx.fillStyle = "rgba(155, 63, 79, 0.28)";
    ctx.lineWidth = 1.2;
    for (const transform of transforms) {
      const yStart = mapValue(transform.row - 0.5, [0, rows.length - 1], [top, bottom]) + 13;
      const yEnd = mapValue(transform.row + 0.5, [0, rows.length - 1], [top, bottom]) - 13;
      ctx.beginPath();
      ctx.moveTo(arrowX, yStart);
      ctx.lineTo(arrowX, yEnd);
      ctx.stroke();
      ctx.beginPath();
      ctx.moveTo(arrowX, yEnd);
      ctx.lineTo(arrowX - 4, yEnd - 5);
      ctx.lineTo(arrowX + 4, yEnd - 5);
      ctx.closePath();
      ctx.fill();
    }
    ctx.fillStyle = "#9b3f4f";
    ctx.font = "10.5px ui-sans-serif, system-ui";
    ctx.textAlign = "left";
    for (const transform of transforms) {
      const yPos = mapValue(transform.row, [0, rows.length - 1], [top, bottom]);
      const labelWidth = ctx.measureText(transform.label).width;
      ctx.fillStyle = "#ffffff";
      ctx.fillRect(6, yPos - 8, labelWidth + 6, 14);
      ctx.fillStyle = "#9b3f4f";
      ctx.fillText(transform.label, 8, yPos + 3);
    }
    ctx.restore();
  }

  function drawFold() {
    const { ctx, width, height } = fitCanvas(foldCanvas);
    const snap = snapshots[frame];
    const rows = rowsForSnapshot(snap);
    const left = 164;
    const right = width - 18;
    const graphTop = 22;
    const graphBottom = 136;
    const top = 176;
    const bottom = height - 34;
    const stride = 1;

    ctx.clearRect(0, 0, width, height);
    ctx.fillStyle = "#ffffff";
    ctx.fillRect(0, 0, width, height);
    drawTargetCurve(ctx, snap, left, right, graphTop, graphBottom);
    drawGrid(ctx, width, height, left, right, top, bottom, rows, snap);

    ctx.save();
    ctx.lineCap = "round";
    ctx.lineJoin = "round";
    drawMaterialCurves(ctx, snap, rows, left, right, top, bottom, stride);
    ctx.restore();

  }

  function drawTargetCurve(ctx, snap, left, right, top, bottom) {
    ctx.save();
    ctx.fillStyle = "#62666f";
    ctx.font = "12px ui-sans-serif, system-ui";
    ctx.textAlign = "center";
    ctx.fillText("target / output", (left + right) / 2, top - 8);

    ctx.strokeStyle = "#e1e4ea";
    ctx.fillStyle = "#62666f";
    ctx.lineWidth = 1;
    ctx.font = "11px ui-sans-serif, system-ui";
    ctx.textAlign = "left";
    for (let tick = 0; tick <= 4; tick += 1) {
      const t = tick / 4;
      const yPos = lerp(bottom, top, t);
      const value = lerp(outputDomain[0], outputDomain[1], t);
      ctx.beginPath();
      ctx.moveTo(left, yPos);
      ctx.lineTo(right, yPos);
      ctx.stroke();
      ctx.fillText(value.toFixed(2), 8, yPos + 4);
    }

    ctx.strokeStyle = "#161719";
    ctx.lineWidth = 2.2;
    ctx.beginPath();
    for (let i = 0; i < x.length; i += 1) {
      const px = mapValue(x[i], xDomain, [left, right]);
      const py = mapValue(snap.pred[i], outputDomain, [bottom, top]);
      if (i === 0) ctx.moveTo(px, py);
      else ctx.lineTo(px, py);
    }
    ctx.stroke();

    ctx.globalAlpha = 0.95;
    ctx.lineWidth = 4;
    const stride = Math.max(2, Math.floor(x.length / 180));
    for (let i = 0; i < x.length - 1; i += stride) {
      const j = Math.min(i + stride, x.length - 1);
      ctx.strokeStyle = targetColorFor(Math.floor((i + j) / 2));
      ctx.beginPath();
      ctx.moveTo(mapValue(x[i], xDomain, [left, right]), mapValue(target[i], outputDomain, [bottom, top]));
      ctx.lineTo(mapValue(x[j], xDomain, [left, right]), mapValue(target[j], outputDomain, [bottom, top]));
      ctx.stroke();
    }
    ctx.restore();
  }

  function materialPoint(rows, layerIndex, sampleIndex, left, right, top, bottom) {
    return {
      x: mapValue(rows[layerIndex].values[sampleIndex], xDomain, [left, right]),
      y: mapValue(layerIndex, [0, rows.length - 1], [top, bottom]),
    };
  }

  function monotoneSegments(values) {
    let localMin = Infinity;
    let localMax = -Infinity;
    for (const value of values) {
      if (value < localMin) localMin = value;
      if (value > localMax) localMax = value;
    }
    const localRange = Math.max(localMax - localMin, 1e-7);
    const eps = Math.max(localRange * 1e-4, 1e-7);
    const turns = [0];
    let prevSign = 0;
    for (let i = 1; i < values.length; i += 1) {
      const diff = values[i] - values[i - 1];
      const sign = diff > eps ? 1 : diff < -eps ? -1 : prevSign;
      if (prevSign !== 0 && sign !== 0 && sign !== prevSign) {
        const turn = i - 1;
        if (turn - turns[turns.length - 1] > 1) turns.push(turn);
      }
      if (sign !== 0) prevSign = sign;
    }
    if (turns[turns.length - 1] !== values.length - 1) turns.push(values.length - 1);

    const segments = [];
    for (let i = 0; i < turns.length - 1; i += 1) {
      if (turns[i + 1] > turns[i]) segments.push({ start: turns[i], end: turns[i + 1] });
    }
    return segments;
  }

  function laneY(rowIndex, laneIndex, laneCount, rows, top, bottom) {
    const baseline = mapValue(rowIndex, [0, rows.length - 1], [top, bottom]);
    if (laneCount <= 1) return baseline;
    const rowGap = (bottom - top) / Math.max(1, rows.length - 1);
    const laneSpan = Math.min(rowGap * 0.76, 42);
    const laneStep = laneSpan / Math.max(1, laneCount - 1);
    return baseline - laneSpan / 2 + laneIndex * laneStep;
  }

  function laneIndexForSample(segments, sampleIndex) {
    for (let lane = 0; lane < segments.length; lane += 1) {
      const segment = segments[lane];
      if (sampleIndex >= segment.start && sampleIndex <= segment.end) return lane;
    }
    return Math.max(0, segments.length - 1);
  }

  function drawMaterialCurves(ctx, snap, rows, left, right, top, bottom, stride) {
    const rowSegments = rows.map((row) => monotoneSegments(row.values));
    ctx.save();
    ctx.lineCap = "round";
    ctx.lineJoin = "round";
    const drawStride = 1;
    ctx.lineWidth = 3.6;
    ctx.globalAlpha = 0.96;

    for (let l = 0; l < rows.length; l += 1) {
      const values = rows[l].values;
      const segments = rowSegments[l];
      for (let lane = 0; lane < segments.length; lane += 1) {
        const segment = segments[lane];
        const yPos = laneY(l, lane, segments.length, rows, top, bottom);
        for (let i = segment.start; i < segment.end; i += drawStride) {
          const j = Math.min(i + drawStride, segment.end);
          const p0x = mapValue(values[i], xDomain, [left, right]);
          const p1x = mapValue(values[j], xDomain, [left, right]);
          ctx.strokeStyle = rowColor(rows[l], Math.floor((i + j) / 2), snap);
          ctx.beginPath();
          ctx.moveTo(p0x, yPos);
          ctx.lineTo(p1x, yPos);
          ctx.stroke();
        }
      }

      ctx.lineWidth = 3.0;
      for (let lane = 0; lane < segments.length - 1; lane += 1) {
        const turnIndex = segments[lane].end;
        const turnValue = values[turnIndex];
        const xPos = mapValue(turnValue, xDomain, [left, right]);
        const y0 = laneY(l, lane, segments.length, rows, top, bottom);
        const y1 = laneY(l, lane + 1, segments.length, rows, top, bottom);
        ctx.strokeStyle = rowColor(rows[l], turnIndex, snap);
        ctx.beginPath();
        ctx.moveTo(xPos, y0);
        ctx.lineTo(xPos, y1);
        ctx.stroke();
      }
      ctx.lineWidth = 3.6;
    }

    ctx.globalAlpha = 0.18;
    ctx.lineWidth = 1.25;
    for (let l = 1; l < rows.length - 1; l += 1) {
      for (let i = 0; i < x.length; i += Math.max(stride * 13, 26)) {
        const priorLane = laneIndexForSample(rowSegments[l - 1], i);
        const nextLane = laneIndexForSample(rowSegments[l], i);
        const priorY = laneY(l - 1, priorLane, rowSegments[l - 1].length, rows, top, bottom);
        const nextY = laneY(l, nextLane, rowSegments[l].length, rows, top, bottom);
        const p0x = mapValue(rows[l - 1].values[i], xDomain, [left, right]);
        const p1x = mapValue(rows[l].values[i], xDomain, [left, right]);
        ctx.strokeStyle = colorFor(i);
        ctx.beginPath();
        ctx.moveTo(p0x, priorY);
        ctx.lineTo(p1x, nextY);
        ctx.stroke();
      }
    }

    ctx.restore();
  }

  function drawPastry(ctx, snap, rows, left, right, top, bottom, stride) {
    ctx.save();
    ctx.lineJoin = "round";

    ctx.globalAlpha = 0.24;
    for (let l = 0; l < rows.length - 1; l += 1) {
      for (let i = 0; i < x.length - 1; i += stride) {
        const j = Math.min(i + stride, x.length - 1);
        const p0 = materialPoint(rows, l, i, left, right, top, bottom);
        const p1 = materialPoint(rows, l, j, left, right, top, bottom);
        const p2 = materialPoint(rows, l + 1, j, left, right, top, bottom);
        const p3 = materialPoint(rows, l + 1, i, left, right, top, bottom);
        ctx.fillStyle = rowColor(rows[l], Math.floor((i + j) / 2), snap);
        ctx.beginPath();
        ctx.moveTo(p0.x, p0.y);
        ctx.lineTo(p1.x, p1.y);
        ctx.lineTo(p2.x, p2.y);
        ctx.lineTo(p3.x, p3.y);
        ctx.closePath();
        ctx.fill();
      }
    }

    ctx.globalAlpha = 0.9;
    ctx.lineCap = "round";
    ctx.lineWidth = 5;
    for (let l = 0; l < rows.length; l += 1) {
      for (let i = 0; i < x.length - 1; i += stride) {
        const j = Math.min(i + stride, x.length - 1);
        const p0 = materialPoint(rows, l, i, left, right, top, bottom);
        const p1 = materialPoint(rows, l, j, left, right, top, bottom);
        ctx.strokeStyle = rowColor(rows[l], Math.floor((i + j) / 2), snap);
        ctx.beginPath();
        ctx.moveTo(p0.x, p0.y);
        ctx.lineTo(p1.x, p1.y);
        ctx.stroke();
      }
    }

    ctx.restore();
  }

  function nearestSnapshotIndex(step) {
    const clampedStep = Math.max(playbackStartStep, Math.min(playbackEndStep, step));
    let lo = 0;
    let hi = snapshots.length - 1;
    while (lo < hi) {
      const mid = Math.floor((lo + hi) / 2);
      if (snapshots[mid].step < clampedStep) lo = mid + 1;
      else hi = mid;
    }
    if (lo > 0 && Math.abs(snapshots[lo - 1].step - clampedStep) < Math.abs(snapshots[lo].step - clampedStep)) {
      return lo - 1;
    }
    return lo;
  }

  function drawLossScrubber() {
    const { ctx, width, height } = fitCanvas(lossScrubCanvas);
    const pad = { left: 58, right: 18, top: 16, bottom: 20 };
    const losses = data.losses.map((d) => d.loss);
    const domain = paddedExtent(losses);
    domain[0] = Math.max(0, domain[0]);
    const xRange = [pad.left, width - pad.right];
    const yRange = [height - pad.bottom, pad.top];

    ctx.clearRect(0, 0, width, height);
    ctx.fillStyle = "#fffefa";
    ctx.fillRect(0, 0, width, height);

    ctx.save();
    ctx.strokeStyle = "#e1e4ea";
    ctx.fillStyle = "#62666f";
    ctx.lineWidth = 1;
    ctx.font = "11px ui-sans-serif, system-ui";
    ctx.textBaseline = "alphabetic";
    for (let i = 0; i <= 2; i += 1) {
      const t = i / 2;
      const yPos = lerp(yRange[0], yRange[1], t);
      const value = lerp(domain[0], domain[1], t);
      ctx.beginPath();
      ctx.moveTo(xRange[0], yPos);
      ctx.lineTo(xRange[1], yPos);
      ctx.stroke();
      ctx.fillText(value.toFixed(2), 8, yPos + 4);
    }
    ctx.font = "12px ui-sans-serif, system-ui";
    ctx.fillText("Loss", 8, 11);
    ctx.font = "11px ui-sans-serif, system-ui";
    ctx.fillText("step", xRange[1] - 24, height - 3);
    ctx.restore();

    ctx.save();
    ctx.strokeStyle = "#c28d2c";
    ctx.lineWidth = 2;
    ctx.beginPath();
    const visibleLosses = data.losses.filter((lossPoint) => lossPoint.step <= playbackEndStep);
    for (let i = 0; i < visibleLosses.length; i += 1) {
      const px = mapValue(visibleLosses[i].step, [playbackStartStep, playbackEndStep], xRange);
      const py = mapValue(visibleLosses[i].loss, domain, yRange);
      if (i === 0) ctx.moveTo(px, py);
      else ctx.lineTo(px, py);
    }
    ctx.stroke();

    const markerX = mapValue(snapshots[frame].step, [playbackStartStep, playbackEndStep], xRange);
    ctx.strokeStyle = "#9b3f4f";
    ctx.lineWidth = 1.4;
    ctx.beginPath();
    ctx.moveTo(markerX, pad.top);
    ctx.lineTo(markerX, height - pad.bottom);
    ctx.stroke();
    ctx.restore();
  }

  function drawAxes(ctx, width, height, pad, yDomain) {
    ctx.save();
    ctx.strokeStyle = "#d8dbe2";
    ctx.fillStyle = "#62666f";
    ctx.lineWidth = 1;
    ctx.font = "12px ui-sans-serif, system-ui";
    const left = pad.left;
    const right = width - pad.right;
    const top = pad.top;
    const bottom = height - pad.bottom;
    for (let i = 0; i <= 4; i += 1) {
      const t = i / 4;
      const yPos = lerp(bottom, top, t);
      const value = lerp(yDomain[0], yDomain[1], t);
      ctx.beginPath();
      ctx.moveTo(left, yPos);
      ctx.lineTo(right, yPos);
      ctx.stroke();
      ctx.fillText(value.toFixed(2), 8, yPos + 4);
    }
    ctx.strokeStyle = "#b7bdc8";
    ctx.beginPath();
    ctx.moveTo(left, bottom);
    ctx.lineTo(right, bottom);
    ctx.stroke();
    ctx.restore();
  }

  function drawSeries(ctx, values, stroke, lineWidth, xRange, yRange, yDomain) {
    ctx.save();
    ctx.strokeStyle = stroke;
    ctx.lineWidth = lineWidth;
    ctx.lineJoin = "round";
    ctx.beginPath();
    for (let i = 0; i < values.length; i += 1) {
      const px = mapValue(i, [0, values.length - 1], xRange);
      const py = mapValue(values[i], yDomain, yRange);
      if (i === 0) ctx.moveTo(px, py);
      else ctx.lineTo(px, py);
    }
    ctx.stroke();
    ctx.restore();
  }

  function render() {
    const snap = snapshots[frame];
    timeSlider.value = String(snap.step);
    stepReadout.textContent = `step ${snap.step}`;
    lossReadout.textContent = `loss ${snap.loss.toFixed(5)}`;
    drawFold();
    drawLossScrubber();
  }

  function tick(now) {
    if (!playing) return;
    if (!lastTick) lastTick = now;
    const elapsed = now - lastTick;
    lastTick = now;
    playbackStep += elapsed / playbackStepMs;
    if (playbackStep >= playbackEndStep) {
      playbackStep = playbackStartStep + ((playbackStep - playbackStartStep) % (playbackEndStep - playbackStartStep));
    }
    const nextFrame = nearestSnapshotIndex(playbackStep);
    if (nextFrame !== frame) {
      frame = nextFrame;
      render();
    }
    requestAnimationFrame(tick);
  }

  playButton.addEventListener("click", () => {
    playing = !playing;
    playButton.textContent = playing ? "Pause" : "Play";
    lastTick = 0;
    if (playing) {
      if (snapshots[frame].step >= playbackEndStep) {
        playbackStep = playbackStartStep;
        frame = nearestSnapshotIndex(playbackStep);
        render();
      } else {
        playbackStep = snapshots[frame].step;
      }
      requestAnimationFrame(tick);
    }
  });

  timeSlider.addEventListener("input", () => {
    playbackStep = Number(timeSlider.value);
    frame = nearestSnapshotIndex(playbackStep);
    render();
  });

  window.addEventListener("resize", render);

  frame = nearestSnapshotIndex(playbackStartStep);
  render();
})();
