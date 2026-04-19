// Lightweight matrix operations for nanoGPT
// All matrices are stored as Float32Array (row-major) with shape [rows, cols]

export interface Matrix {
  data: Float32Array;
  rows: number;
  cols: number;
}

export function zeros(rows: number, cols: number): Matrix {
  return { data: new Float32Array(rows * cols), rows, cols };
}

export function randn(rows: number, cols: number, scale = 0.02): Matrix {
  const data = new Float32Array(rows * cols);
  for (let i = 0; i < data.length; i++) {
    // Box-Muller transform
    const u1 = Math.random() || 1e-9;
    const u2 = Math.random();
    data[i] = Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2) * scale;
  }
  return { data, rows, cols };
}

export function clone(m: Matrix): Matrix {
  return { data: new Float32Array(m.data), rows: m.rows, cols: m.cols };
}

// C = A @ B  with shapes [m,k] @ [k,n] -> [m,n]
export function matmul(A: Matrix, B: Matrix): Matrix {
  if (A.cols !== B.rows) throw new Error(`matmul shape mismatch ${A.rows}x${A.cols} @ ${B.rows}x${B.cols}`);
  const m = A.rows, k = A.cols, n = B.cols;
  const C = new Float32Array(m * n);
  const a = A.data, b = B.data;
  for (let i = 0; i < m; i++) {
    for (let p = 0; p < k; p++) {
      const aip = a[i * k + p];
      if (aip === 0) continue;
      const aRow = i * n;
      const bRow = p * n;
      for (let j = 0; j < n; j++) {
        C[aRow + j] += aip * b[bRow + j];
      }
    }
  }
  return { data: C, rows: m, cols: n };
}

// Add bias vector (length cols) to each row of matrix
export function addBias(M: Matrix, b: Matrix): Matrix {
  const out = new Float32Array(M.data.length);
  for (let i = 0; i < M.rows; i++) {
    for (let j = 0; j < M.cols; j++) {
      out[i * M.cols + j] = M.data[i * M.cols + j] + b.data[j];
    }
  }
  return { data: out, rows: M.rows, cols: M.cols };
}

export function add(A: Matrix, B: Matrix): Matrix {
  const out = new Float32Array(A.data.length);
  for (let i = 0; i < A.data.length; i++) out[i] = A.data[i] + B.data[i];
  return { data: out, rows: A.rows, cols: A.cols };
}

export function transpose(A: Matrix): Matrix {
  const out = new Float32Array(A.rows * A.cols);
  for (let i = 0; i < A.rows; i++)
    for (let j = 0; j < A.cols; j++)
      out[j * A.rows + i] = A.data[i * A.cols + j];
  return { data: out, rows: A.cols, cols: A.rows };
}

// GELU activation
export function gelu(M: Matrix): Matrix {
  const out = new Float32Array(M.data.length);
  for (let i = 0; i < M.data.length; i++) {
    const x = M.data[i];
    out[i] = 0.5 * x * (1 + Math.tanh(Math.sqrt(2 / Math.PI) * (x + 0.044715 * x * x * x)));
  }
  return { data: out, rows: M.rows, cols: M.cols };
}

// Layer Norm applied per-row
export function layerNorm(M: Matrix, gamma: Matrix, beta: Matrix, eps = 1e-5): Matrix {
  const out = new Float32Array(M.data.length);
  const C = M.cols;
  for (let i = 0; i < M.rows; i++) {
    let mean = 0;
    for (let j = 0; j < C; j++) mean += M.data[i * C + j];
    mean /= C;
    let varc = 0;
    for (let j = 0; j < C; j++) {
      const d = M.data[i * C + j] - mean;
      varc += d * d;
    }
    varc /= C;
    const denom = 1 / Math.sqrt(varc + eps);
    for (let j = 0; j < C; j++) {
      const norm = (M.data[i * C + j] - mean) * denom;
      out[i * C + j] = norm * gamma.data[j] + beta.data[j];
    }
  }
  return { data: out, rows: M.rows, cols: M.cols };
}

// Softmax per row, with optional causal mask. mask: only positions <= row index allowed
export function softmaxCausal(M: Matrix): Matrix {
  const out = new Float32Array(M.data.length);
  const R = M.rows, C = M.cols;
  for (let i = 0; i < R; i++) {
    const limit = Math.min(i + 1, C);
    let maxv = -Infinity;
    for (let j = 0; j < limit; j++) {
      const v = M.data[i * C + j];
      if (v > maxv) maxv = v;
    }
    let sum = 0;
    for (let j = 0; j < limit; j++) {
      const e = Math.exp(M.data[i * C + j] - maxv);
      out[i * C + j] = e;
      sum += e;
    }
    const inv = 1 / sum;
    for (let j = 0; j < limit; j++) out[i * C + j] *= inv;
    // positions > i remain 0
  }
  return { data: out, rows: R, cols: C };
}

// Softmax per row (no mask)
export function softmax(M: Matrix): Matrix {
  const out = new Float32Array(M.data.length);
  const R = M.rows, C = M.cols;
  for (let i = 0; i < R; i++) {
    let maxv = -Infinity;
    for (let j = 0; j < C; j++) {
      const v = M.data[i * C + j];
      if (v > maxv) maxv = v;
    }
    let sum = 0;
    for (let j = 0; j < C; j++) {
      const e = Math.exp(M.data[i * C + j] - maxv);
      out[i * C + j] = e;
      sum += e;
    }
    const inv = 1 / sum;
    for (let j = 0; j < C; j++) out[i * C + j] *= inv;
  }
  return { data: out, rows: R, cols: C };
}

export function scale(M: Matrix, s: number): Matrix {
  const out = new Float32Array(M.data.length);
  for (let i = 0; i < M.data.length; i++) out[i] = M.data[i] * s;
  return { data: out, rows: M.rows, cols: M.cols };
}
