import React, { useEffect, useMemo, useRef, useState } from "react";

// Source image URL (will be drawn into a fixed N×N canvas for DCT processing).
const IMAGE_URL =
  "https://chandak1299.github.io/Teaching/ENGR76_Winter_2026/Visualizations/image_example.jpg";

const N_DEFAULT = 256;

function clamp01(x: number) {
  return Math.max(0, Math.min(1, x));
}

function idx(r: number, c: number, n: number) {
  return r * n + c;
}

// Build orthonormal DCT-II matrix C (NxN): X = C * x * C^T
// With this definition, the inverse is: x = C^T * X * C
function buildDCTMatrix(n: number): Float32Array {
  const C = new Float32Array(n * n);
  const s0 = Math.sqrt(1 / n);
  const s = Math.sqrt(2 / n);
  for (let k = 0; k < n; k++) {
    const alpha = k === 0 ? s0 : s;
    for (let i = 0; i < n; i++) {
      C[idx(k, i, n)] =
        alpha * Math.cos(((Math.PI * (2 * i + 1)) / (2 * n)) * k);
    }
  }
  return C;
}

function transposeSquare(A: Float32Array, n: number): Float32Array {
  const out = new Float32Array(n * n);
  for (let r = 0; r < n; r++) {
    for (let c = 0; c < n; c++) {
      out[idx(c, r, n)] = A[idx(r, c, n)];
    }
  }
  return out;
}

// A (m x n) times B (n x p) => out (m x p)
function matMul(
  A: Float32Array,
  B: Float32Array,
  m: number,
  n: number,
  p: number
): Float32Array {
  const out = new Float32Array(m * p);
  for (let i = 0; i < m; i++) {
    const aRow = i * n;
    const outRow = i * p;
    for (let k = 0; k < n; k++) {
      const a = A[aRow + k];
      const bRow = k * p;
      for (let j = 0; j < p; j++) {
        out[outRow + j] += a * B[bRow + j];
      }
    }
  }
  return out;
}

// 2D DCT using separability: X = C * x * C^T
function dct2(x: Float32Array, C: Float32Array, Ct: Float32Array, n: number) {
  const tmp = matMul(C, x, n, n, n); // (n x n)
  const X = matMul(tmp, Ct, n, n, n);
  return X;
}

// Inverse 2D DCT for orthonormal C: x = C^T * X * C
function idct2(X: Float32Array, C: Float32Array, Ct: Float32Array, n: number) {
  const tmp = matMul(Ct, X, n, n, n);
  const x = matMul(tmp, C, n, n, n);
  return x;
}

function maskTopLeft(
  X: Float32Array,
  n: number,
  k: number
): { Xm: Float32Array; coeffRatio: number; energyRatio: number } {
  const kk = Math.max(1, Math.min(n, Math.floor(k)));
  const Xm = new Float32Array(n * n);
  let totalEnergy = 0;
  let keptEnergy = 0;
  for (let r = 0; r < n; r++) {
    for (let c = 0; c < n; c++) {
      const v = X[idx(r, c, n)];
      const e = v * v;
      totalEnergy += e;
      if (r < kk && c < kk) {
        Xm[idx(r, c, n)] = v;
        keptEnergy += e;
      }
    }
  }
  const coeffRatio = (kk * kk) / (n * n);
  const energyRatio = totalEnergy > 0 ? keptEnergy / totalEnergy : 0;
  return { Xm, coeffRatio, energyRatio };
}

function drawGrayToCanvas(
  canvas: HTMLCanvasElement,
  x: Float32Array,
  n: number
) {
  const ctx = canvas.getContext("2d", { willReadFrequently: true });
  if (!ctx) return;
  const imgData = ctx.createImageData(n, n);
  for (let i = 0; i < n * n; i++) {
    const v = Math.round(clamp01(x[i]) * 255);
    const j = i * 4;
    imgData.data[j + 0] = v;
    imgData.data[j + 1] = v;
    imgData.data[j + 2] = v;
    imgData.data[j + 3] = 255;
  }
  ctx.putImageData(imgData, 0, 0);
}

function drawImageToCanvas(
  canvas: HTMLCanvasElement,
  image: HTMLImageElement,
  n: number
) {
  const ctx = canvas.getContext("2d", { willReadFrequently: true });
  if (!ctx) return;
  ctx.clearRect(0, 0, n, n);
  ctx.drawImage(image, 0, 0, n, n);
}

function readGrayFromCanvas(canvas: HTMLCanvasElement, n: number): Float32Array {
  const ctx = canvas.getContext("2d", { willReadFrequently: true });
  if (!ctx) return new Float32Array(n * n);
  const imgData = ctx.getImageData(0, 0, n, n);
  const out = new Float32Array(n * n);
  for (let i = 0; i < n * n; i++) {
    const j = i * 4;
    // Use luminance instead of assuming grayscale.
    const r = imgData.data[j] / 255;
    const g = imgData.data[j + 1] / 255;
    const b = imgData.data[j + 2] / 255;
    out[i] = 0.2126 * r + 0.7152 * g + 0.0722 * b;
  }
  return out;
}

// ========== Lightweight self-tests (run once in dev) ==========
function approxEqual(a: number, b: number, tol: number) {
  return Math.abs(a - b) <= tol;
}

function runSelfTests() {
  // clamp01
  console.assert(clamp01(-1) === 0, "clamp01(-1) should be 0");
  console.assert(clamp01(2) === 1, "clamp01(2) should be 1");
  console.assert(clamp01(0.25) === 0.25, "clamp01(0.25) should be 0.25");

  // Orthonormality sanity + reconstruction sanity on small N
  const n = 8;
  const C = buildDCTMatrix(n);
  const Ct = transposeSquare(C, n);

  // Check that C * C^T ≈ I
  const CCt = matMul(C, Ct, n, n, n);
  for (let r = 0; r < n; r++) {
    for (let c = 0; c < n; c++) {
      const v = CCt[idx(r, c, n)];
      const target = r === c ? 1 : 0;
      console.assert(
        approxEqual(v, target, 1e-4),
        `Orthonormality failed at (${r},${c}): got ${v}, expected ${target}`
      );
    }
  }

  // Check idct2(dct2(x)) ≈ x
  const x = new Float32Array(n * n);
  for (let i = 0; i < x.length; i++) x[i] = (i % 7) / 7; // deterministic
  const X = dct2(x, C, Ct, n);
  const xr = idct2(X, C, Ct, n);
  let maxErr = 0;
  for (let i = 0; i < x.length; i++) maxErr = Math.max(maxErr, Math.abs(x[i] - xr[i]));
  console.assert(maxErr < 1e-3, `Reconstruction error too high: ${maxErr}`);

  // Mask ratios
  const { coeffRatio } = maskTopLeft(X, n, 2);
  console.assert(
    approxEqual(coeffRatio, (2 * 2) / (n * n), 1e-12),
    "coeffRatio should be k^2 / n^2"
  );
}

if (typeof window !== "undefined") {
  const w = window as any;
  if (!w.__DCT_SELF_TESTS_RAN__) {
    w.__DCT_SELF_TESTS_RAN__ = true;
    try {
      runSelfTests();
    } catch (e) {
      // Don’t crash UI because of tests; just log.
      console.error("DCT self-tests failed:", e);
    }
  }
}

export default function DCTLowPassCanvas() {
  const n = N_DEFAULT;

  const originalCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const reconCanvasRef = useRef<HTMLCanvasElement | null>(null);

  const [cutoff, setCutoff] = useState(Math.floor(n / 8));
  const [ready, setReady] = useState(false);
  const [processing, setProcessing] = useState(false);

  const [coeffPct, setCoeffPct] = useState(0);
  // Energy percentage removed for simplicity
  const [_energyPct, _setEnergyPct] = useState(0);
  const [statusMsg, setStatusMsg] = useState<string | null>(null);

  const imageEl = useMemo(() => {
    const im = new Image();
    im.crossOrigin = "anonymous";
    im.src = IMAGE_URL;
    return im;
  }, []);

  const C = useMemo(() => buildDCTMatrix(n), [n]);
  const Ct = useMemo(() => transposeSquare(C, n), [C, n]);

  // Cache full DCT so the slider only does masking + inverse.
  const fullDCTRef = useRef<Float32Array | null>(null);

  useEffect(() => {
    let cancelled = false;
    const init = async () => {
      setStatusMsg(null);
      const orig = originalCanvasRef.current;
      const recon = reconCanvasRef.current;
      if (!orig || !recon) return;

      // Load the image
      await new Promise<void>((resolve, reject) => {
        imageEl.onload = () => resolve();
        imageEl.onerror = () =>
          reject(new Error("Failed to load image from IMAGE_URL (possible CORS issue)."));
      });

      drawImageToCanvas(orig, imageEl, n);
      // read grayscale 0..1
      const x = readGrayFromCanvas(orig, n);

      // Compute DCT once
      setProcessing(true);
      await new Promise((r) => setTimeout(r, 0));
      const X = dct2(x, C, Ct, n);
      if (cancelled) return;
      fullDCTRef.current = X;
      setProcessing(false);
      setReady(true);
    };

    init().catch((e) => {
      console.error(e);
      setStatusMsg(String(e?.message ?? e));
      setProcessing(false);
      setReady(false);
    });

    return () => {
      cancelled = true;
    };
  }, [C, Ct, imageEl, n]);

  useEffect(() => {
    if (!ready) return;
    const X = fullDCTRef.current;
    const recon = reconCanvasRef.current;
    if (!X || !recon) return;

    let cancelled = false;

    const run = async () => {
      setProcessing(true);
      await new Promise((r) => setTimeout(r, 0));

      const { Xm, coeffRatio, energyRatio } = maskTopLeft(X, n, cutoff);
      const xhat = idct2(Xm, C, Ct, n);

      if (cancelled) return;

      drawGrayToCanvas(recon, xhat, n);
      setCoeffPct(coeffRatio * 100);
      // energy percentage intentionally not displayed
      _setEnergyPct(energyRatio * 100);
      setProcessing(false);
    };

    run();
    return () => {
      cancelled = true;
    };
  }, [cutoff, ready, C, Ct, n]);

  return (
    <div className="min-h-screen w-full bg-white text-zinc-900 p-6">
      <div className="max-w-6xl mx-auto">
        <div className="flex items-end justify-between gap-4 flex-wrap">
          <div>
            <div className="text-2xl font-semibold">2D DCT Low‑Pass Reconstruction</div>
            <div className="text-sm text-zinc-600 mt-1">
              Slider keeps the top‑left <span className="font-medium">k×k</span> DCT coefficients (same cutoff in both directions).
            </div>
          </div>
          <div className="text-sm text-zinc-700">
            {processing ? (
              <span className="inline-flex items-center gap-2">
                <span className="h-2 w-2 rounded-full bg-zinc-900 animate-pulse" /> Processing…
              </span>
            ) : ready ? (
              <span>Ready</span>
            ) : (
              <span>Loading…</span>
            )}
          </div>
        </div>

        {statusMsg ? (
          <div className="mt-4 rounded-xl border border-red-200 bg-red-50 p-4 text-sm text-red-800">
            <div className="font-semibold">Couldn’t load or process the image</div>
            <div className="mt-1 break-words">{statusMsg}</div>
            <div className="mt-2 text-xs text-red-700">
              If this is a CORS issue, hosting the image with permissive CORS headers (or serving it from the same origin) will fix it.
            </div>
          </div>
        ) : null}

        <div className="mt-6 grid grid-cols-1 lg:grid-cols-2 gap-6">
          <div className="rounded-2xl border border-zinc-200 shadow-sm p-4">
            <div className="text-sm font-medium text-zinc-800 mb-3">Original</div>
            <canvas
              ref={originalCanvasRef}
              width={n}
              height={n}
              className="w-full h-auto rounded-xl border border-zinc-100"
            />
          </div>

          <div className="rounded-2xl border border-zinc-200 shadow-sm p-4">
            <div className="text-sm font-medium text-zinc-800 mb-3">Reconstruction (after low‑pass filter)</div>
            <canvas
              ref={reconCanvasRef}
              width={n}
              height={n}
              className="w-full h-auto rounded-xl border border-zinc-100"
            />
          </div>
        </div>

        <div className="mt-6 rounded-2xl border border-zinc-200 shadow-sm p-5">
          <div className="flex items-center justify-between flex-wrap gap-4">
            <div>
              <div className="text-sm font-medium">Cutoff (k×k coefficients)</div>
              <div className="text-xs text-zinc-600">k = {cutoff} (out of {n})</div>
            </div>
            <div className="text-lg text-zinc-800">
              <span className="font-semibold">Coefficients kept:</span> {coeffPct.toFixed(2)}%
            </div>
          </div>

          <div className="mt-4">
            <input
              type="range"
              min={1}
              max={n}
              value={cutoff}
              onChange={(e) => setCutoff(parseInt(e.target.value, 10))}
              className="w-full"
              disabled={!ready}
            />
            <div className="mt-2 flex justify-between text-xs text-zinc-500">
              <span>1</span>
              <span>{Math.floor(n / 4)}</span>
              <span>{Math.floor(n / 2)}</span>
              <span>{n}</span>
            </div>
          </div>

          
        </div>
      </div>
    </div>
  );
}
