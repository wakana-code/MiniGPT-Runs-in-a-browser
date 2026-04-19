// A minimal, trainable GPT-like Transformer (single block, single head by default).
// Implements: token+position embeddings, causal self-attention, MLP, layernorm, lm_head (weight tied).
// All forward + backward passes are implemented by hand for educational transparency.

import {
  Matrix,
  zeros,
  randn,
  matmul,
  transpose,
  addBias,
  add,
  gelu,
  softmaxCausal,
} from "./matrix";

export interface GPTConfig {
  vocabSize: number;
  blockSize: number; // context length
  nEmbd: number;
  nHead: number;
  nLayer: number;
}

interface AttnCache {
  x: Matrix;        // input to attention (after ln1)
  q: Matrix; k: Matrix; v: Matrix; // [T, nEmbd]
  // per head
  attnWeights: Matrix[]; // [T,T] each
  headOuts: Matrix[];    // [T, headDim]
  concat: Matrix;        // [T, nEmbd]
  proj: Matrix;          // [T, nEmbd]
}

interface BlockCache {
  inp: Matrix;          // residual input
  ln1Out: Matrix;
  ln1Mean: Float32Array; ln1Var: Float32Array;
  attn: AttnCache;
  afterAttn: Matrix;    // inp + attn.proj
  ln2Out: Matrix;
  ln2Mean: Float32Array; ln2Var: Float32Array;
  fc1: Matrix; fc1Act: Matrix; fc2: Matrix;
  out: Matrix;          // afterAttn + fc2
}

interface ForwardCache {
  ids: number[];
  tokEmb: Matrix;       // [T, nEmbd]
  posEmb: Matrix;       // [T, nEmbd]
  x0: Matrix;           // tokEmb + posEmb
  blocks: BlockCache[];
  lnFOut: Matrix;
  lnFMean: Float32Array; lnFVar: Float32Array;
  logits: Matrix;       // [T, vocabSize]
  probs: Matrix;        // [T, vocabSize]
}

interface BlockParams {
  // attention
  Wq: Matrix; bq: Matrix;
  Wk: Matrix; bk: Matrix;
  Wv: Matrix; bv: Matrix;
  Wo: Matrix; bo: Matrix;
  // mlp
  W1: Matrix; b1: Matrix;
  W2: Matrix; b2: Matrix;
  // layer norms
  ln1g: Matrix; ln1b: Matrix;
  ln2g: Matrix; ln2b: Matrix;
}

export class GPT {
  cfg: GPTConfig;
  // embeddings
  wte: Matrix; // [vocab, nEmbd]
  wpe: Matrix; // [blockSize, nEmbd]
  blocks: BlockParams[];
  lnFg: Matrix; lnFb: Matrix;
  // Adam state
  step = 0;
  private mState = new WeakMap<Matrix, Float32Array>();
  private vState = new WeakMap<Matrix, Float32Array>();

  constructor(cfg: GPTConfig) {
    this.cfg = cfg;
    this.wte = randn(cfg.vocabSize, cfg.nEmbd, 0.02);
    this.wpe = randn(cfg.blockSize, cfg.nEmbd, 0.02);
    this.blocks = [];
    for (let i = 0; i < cfg.nLayer; i++) {
      this.blocks.push(this.makeBlock());
    }
    this.lnFg = onesVec(cfg.nEmbd);
    this.lnFb = zeros(1, cfg.nEmbd);
  }

  private makeBlock(): BlockParams {
    const C = this.cfg.nEmbd;
    return {
      Wq: randn(C, C, 0.02), bq: zeros(1, C),
      Wk: randn(C, C, 0.02), bk: zeros(1, C),
      Wv: randn(C, C, 0.02), bv: zeros(1, C),
      Wo: randn(C, C, 0.02), bo: zeros(1, C),
      W1: randn(C, 4 * C, 0.02), b1: zeros(1, 4 * C),
      W2: randn(4 * C, C, 0.02), b2: zeros(1, C),
      ln1g: onesVec(C), ln1b: zeros(1, C),
      ln2g: onesVec(C), ln2b: zeros(1, C),
    };
  }

  // ===== Forward =====
  forward(ids: number[]): ForwardCache {
    const T = ids.length;
    const C = this.cfg.nEmbd;
    // Embedding lookup
    const tokEmb = zeros(T, C);
    for (let t = 0; t < T; t++) {
      const id = ids[t];
      for (let j = 0; j < C; j++) tokEmb.data[t * C + j] = this.wte.data[id * C + j];
    }
    const posEmb = zeros(T, C);
    for (let t = 0; t < T; t++) {
      for (let j = 0; j < C; j++) posEmb.data[t * C + j] = this.wpe.data[t * C + j];
    }
    const x0 = add(tokEmb, posEmb);

    let x = x0;
    const blockCaches: BlockCache[] = [];
    for (let l = 0; l < this.blocks.length; l++) {
      const bc = this.forwardBlock(x, this.blocks[l]);
      blockCaches.push(bc);
      x = bc.out;
    }
    const lnF = layerNormFull(x, this.lnFg, this.lnFb);
    // logits = lnFOut @ wte^T  (weight tying)
    const logits = matmul(lnF.out, transpose(this.wte));
    // softmax
    const probs = rowSoftmax(logits);
    return {
      ids, tokEmb, posEmb, x0, blocks: blockCaches,
      lnFOut: lnF.out, lnFMean: lnF.mean, lnFVar: lnF.varc,
      logits, probs,
    };
  }

  private forwardBlock(inp: Matrix, p: BlockParams): BlockCache {
    const ln1 = layerNormFull(inp, p.ln1g, p.ln1b);
    const attn = this.forwardAttn(ln1.out, p);
    const afterAttn = add(inp, attn.proj);
    const ln2 = layerNormFull(afterAttn, p.ln2g, p.ln2b);
    const fc1 = addBias(matmul(ln2.out, p.W1), p.b1);
    const fc1Act = gelu(fc1);
    const fc2 = addBias(matmul(fc1Act, p.W2), p.b2);
    const out = add(afterAttn, fc2);
    return {
      inp, ln1Out: ln1.out, ln1Mean: ln1.mean, ln1Var: ln1.varc,
      attn, afterAttn,
      ln2Out: ln2.out, ln2Mean: ln2.mean, ln2Var: ln2.varc,
      fc1, fc1Act, fc2, out,
    };
  }

  private forwardAttn(x: Matrix, p: BlockParams): AttnCache {
    const T = x.rows;
    const C = this.cfg.nEmbd;
    const H = this.cfg.nHead;
    const HD = C / H;
    const q = addBias(matmul(x, p.Wq), p.bq);
    const k = addBias(matmul(x, p.Wk), p.bk);
    const v = addBias(matmul(x, p.Wv), p.bv);
    const attnWeights: Matrix[] = [];
    const headOuts: Matrix[] = [];
    const concat = zeros(T, C);
    const scale = 1 / Math.sqrt(HD);
    for (let h = 0; h < H; h++) {
      const qh = sliceCols(q, h * HD, HD);
      const kh = sliceCols(k, h * HD, HD);
      const vh = sliceCols(v, h * HD, HD);
      const scores = matmul(qh, transpose(kh));
      // scale
      for (let i = 0; i < scores.data.length; i++) scores.data[i] *= scale;
      const w = softmaxCausal(scores);
      const o = matmul(w, vh); // [T, HD]
      attnWeights.push(w);
      headOuts.push(o);
      // place into concat
      for (let t = 0; t < T; t++)
        for (let d = 0; d < HD; d++)
          concat.data[t * C + h * HD + d] = o.data[t * HD + d];
    }
    const proj = addBias(matmul(concat, p.Wo), p.bo);
    return { x, q, k, v, attnWeights, headOuts, concat, proj };
  }

  // ===== Loss =====
  // Cross-entropy on next-token prediction. targets[t] is the correct token after position t.
  computeLoss(cache: ForwardCache, targets: number[]): number {
    const T = cache.probs.rows;
    const V = cache.probs.cols;
    let loss = 0;
    for (let t = 0; t < T; t++) {
      const p = Math.max(cache.probs.data[t * V + targets[t]], 1e-12);
      loss += -Math.log(p);
    }
    return loss / T;
  }

  // ===== Backward =====
  backward(cache: ForwardCache, targets: number[]): { grads: Map<Matrix, Matrix> } {
    const T = cache.probs.rows;
    const V = cache.probs.cols;
    const C = this.cfg.nEmbd;
    const grads = new Map<Matrix, Matrix>();
    const accum = (param: Matrix, g: Matrix) => {
      const existing = grads.get(param);
      if (!existing) { grads.set(param, g); return; }
      for (let i = 0; i < g.data.length; i++) existing.data[i] += g.data[i];
    };

    // dL/dlogits = (probs - onehot) / T
    const dLogits = zeros(T, V);
    for (let t = 0; t < T; t++) {
      for (let j = 0; j < V; j++) dLogits.data[t * V + j] = cache.probs.data[t * V + j] / T;
      dLogits.data[t * V + targets[t]] -= 1 / T;
    }
    // logits = lnFOut @ wte^T
    // dLnF = dLogits @ wte
    const dLnFOut = matmul(dLogits, this.wte);
    // dWte += dLogits^T @ lnFOut
    const dWteFromHead = matmul(transpose(dLogits), cache.lnFOut);
    accum(this.wte, dWteFromHead);

    // backward through final layer norm
    const dXAfterFinal = layerNormBackward(
      dLnFOut,
      // input to lnF is the output of last block
      cache.blocks[cache.blocks.length - 1].out,
      cache.lnFMean, cache.lnFVar,
      this.lnFg
    );
    accum(this.lnFg, lnGammaGrad(dLnFOut, cache.blocks[cache.blocks.length - 1].out, cache.lnFMean, cache.lnFVar));
    accum(this.lnFb, lnBetaGrad(dLnFOut));

    // backward through blocks (reverse)
    let dOut = dXAfterFinal;
    for (let l = this.blocks.length - 1; l >= 0; l--) {
      dOut = this.backwardBlock(dOut, cache.blocks[l], this.blocks[l], accum);
    }

    // dOut now is dx0 (gradient wrt token+pos embeddings sum)
    // accumulate into wte (rows indexed by tokens) and wpe
    const dWte = zeros(this.wte.rows, this.wte.cols);
    const dWpe = zeros(this.wpe.rows, this.wpe.cols);
    for (let t = 0; t < T; t++) {
      const id = cache.ids[t];
      for (let j = 0; j < C; j++) {
        dWte.data[id * C + j] += dOut.data[t * C + j];
        dWpe.data[t * C + j] += dOut.data[t * C + j];
      }
    }
    accum(this.wte, dWte);
    accum(this.wpe, dWpe);

    return { grads };
  }

  private backwardBlock(
    dOut: Matrix,
    bc: BlockCache,
    p: BlockParams,
    accum: (m: Matrix, g: Matrix) => void
  ): Matrix {
    // out = afterAttn + fc2  =>  dAfterAttn += dOut, dFc2 = dOut
    const dAfterAttnFromRes = dOut;
    const dFc2 = dOut;
    // fc2 = fc1Act @ W2 + b2
    accum(p.b2, biasGrad(dFc2));
    accum(p.W2, matmul(transpose(bc.fc1Act), dFc2));
    const dFc1Act = matmul(dFc2, transpose(p.W2));
    // gelu backward
    const dFc1 = geluBackward(dFc1Act, bc.fc1);
    // fc1 = ln2Out @ W1 + b1
    accum(p.b1, biasGrad(dFc1));
    accum(p.W1, matmul(transpose(bc.ln2Out), dFc1));
    const dLn2Out = matmul(dFc1, transpose(p.W1));
    // ln2 backward
    const dAfterAttnFromLn = layerNormBackward(dLn2Out, bc.afterAttn, bc.ln2Mean, bc.ln2Var, p.ln2g);
    accum(p.ln2g, lnGammaGrad(dLn2Out, bc.afterAttn, bc.ln2Mean, bc.ln2Var));
    accum(p.ln2b, lnBetaGrad(dLn2Out));
    // sum residual
    const dAfterAttn = addM(dAfterAttnFromRes, dAfterAttnFromLn);
    // afterAttn = inp + attn.proj
    const dInpFromRes = dAfterAttn;
    const dProj = dAfterAttn;
    // proj = concat @ Wo + bo
    accum(p.bo, biasGrad(dProj));
    accum(p.Wo, matmul(transpose(bc.attn.concat), dProj));
    const dConcat = matmul(dProj, transpose(p.Wo));
    // attention backward
    const dLn1Out = this.backwardAttn(dConcat, bc.attn, p, accum);
    // ln1 backward
    const dInpFromLn = layerNormBackward(dLn1Out, bc.inp, bc.ln1Mean, bc.ln1Var, p.ln1g);
    accum(p.ln1g, lnGammaGrad(dLn1Out, bc.inp, bc.ln1Mean, bc.ln1Var));
    accum(p.ln1b, lnBetaGrad(dLn1Out));
    return addM(dInpFromRes, dInpFromLn);
  }

  private backwardAttn(
    dConcat: Matrix,
    ac: AttnCache,
    p: BlockParams,
    accum: (m: Matrix, g: Matrix) => void
  ): Matrix {
    const C = this.cfg.nEmbd;
    const H = this.cfg.nHead;
    const HD = C / H;
    const T = dConcat.rows;
    const scaleFac = 1 / Math.sqrt(HD);
    const dQfull = zeros(T, C);
    const dKfull = zeros(T, C);
    const dVfull = zeros(T, C);
    for (let h = 0; h < H; h++) {
      // dHeadOut from dConcat
      const dHead = zeros(T, HD);
      for (let t = 0; t < T; t++)
        for (let d = 0; d < HD; d++)
          dHead.data[t * HD + d] = dConcat.data[t * C + h * HD + d];
      const w = ac.attnWeights[h];
      const qh = sliceCols(ac.q, h * HD, HD);
      const kh = sliceCols(ac.k, h * HD, HD);
      const vh = sliceCols(ac.v, h * HD, HD);
      // o = w @ vh => dW = dHead @ vh^T,  dVh = w^T @ dHead
      const dW = matmul(dHead, transpose(vh));
      const dVh = matmul(transpose(w), dHead);
      // softmax (causal) backward: dScores[i,j] = w[i,j]*(dW[i,j] - sum_k w[i,k]*dW[i,k])
      const dScores = zeros(T, T);
      for (let i = 0; i < T; i++) {
        let dot = 0;
        const limit = i + 1;
        for (let j = 0; j < limit; j++) dot += w.data[i * T + j] * dW.data[i * T + j];
        for (let j = 0; j < limit; j++) {
          dScores.data[i * T + j] = w.data[i * T + j] * (dW.data[i * T + j] - dot);
        }
      }
      // scale
      for (let i = 0; i < dScores.data.length; i++) dScores.data[i] *= scaleFac;
      // scores = qh @ kh^T => dQh = dScores @ kh, dKh = dScores^T @ qh
      const dQh = matmul(dScores, kh);
      const dKh = matmul(transpose(dScores), qh);
      // place back
      for (let t = 0; t < T; t++)
        for (let d = 0; d < HD; d++) {
          dQfull.data[t * C + h * HD + d] = dQh.data[t * HD + d];
          dKfull.data[t * C + h * HD + d] = dKh.data[t * HD + d];
          dVfull.data[t * C + h * HD + d] = dVh.data[t * HD + d];
        }
    }
    // q = x @ Wq + bq, etc
    accum(p.bq, biasGrad(dQfull));
    accum(p.bk, biasGrad(dKfull));
    accum(p.bv, biasGrad(dVfull));
    accum(p.Wq, matmul(transpose(ac.x), dQfull));
    accum(p.Wk, matmul(transpose(ac.x), dKfull));
    accum(p.Wv, matmul(transpose(ac.x), dVfull));
    const dx_q = matmul(dQfull, transpose(p.Wq));
    const dx_k = matmul(dKfull, transpose(p.Wk));
    const dx_v = matmul(dVfull, transpose(p.Wv));
    return addM(addM(dx_q, dx_k), dx_v);
  }

  // ===== Adam Optimizer =====
  applyGrads(grads: Map<Matrix, Matrix>, lr: number, beta1 = 0.9, beta2 = 0.999, eps = 1e-8, weightDecay = 0.01) {
    this.step += 1;
    const t = this.step;
    const bc1 = 1 - Math.pow(beta1, t);
    const bc2 = 1 - Math.pow(beta2, t);
    grads.forEach((g, param) => {
      let m = this.mState.get(param);
      let v = this.vState.get(param);
      if (!m) { m = new Float32Array(param.data.length); this.mState.set(param, m); }
      if (!v) { v = new Float32Array(param.data.length); this.vState.set(param, v); }
      for (let i = 0; i < param.data.length; i++) {
        const gi = g.data[i];
        m[i] = beta1 * m[i] + (1 - beta1) * gi;
        v[i] = beta2 * v[i] + (1 - beta2) * gi * gi;
        const mhat = m[i] / bc1;
        const vhat = v[i] / bc2;
        // weight decay (decoupled)
        param.data[i] -= lr * (mhat / (Math.sqrt(vhat) + eps) + weightDecay * param.data[i]);
      }
    });
  }

  // ===== Generation =====
  generate(prompt: number[], maxNewTokens: number, temperature = 1.0, topK = 0, onToken?: (id: number) => void): number[] {
    const out = [...prompt];
    for (let i = 0; i < maxNewTokens; i++) {
      const ctx = out.slice(Math.max(0, out.length - this.cfg.blockSize));
      const cache = this.forward(ctx);
      const V = cache.logits.cols;
      const last = cache.logits.data.subarray((ctx.length - 1) * V, ctx.length * V);
      const logits = new Float32Array(last);
      // temperature
      for (let j = 0; j < V; j++) logits[j] /= Math.max(temperature, 1e-6);
      // top-k filter
      if (topK > 0 && topK < V) {
        const sorted = Array.from(logits).map((v, idx) => ({v, idx})).sort((a, b) => b.v - a.v);
        const thresh = sorted[topK - 1].v;
        for (let j = 0; j < V; j++) if (logits[j] < thresh) logits[j] = -Infinity;
      }
      // softmax
      let maxv = -Infinity;
      for (let j = 0; j < V; j++) if (logits[j] > maxv) maxv = logits[j];
      let sum = 0;
      const probs = new Float32Array(V);
      for (let j = 0; j < V; j++) { probs[j] = Math.exp(logits[j] - maxv); sum += probs[j]; }
      for (let j = 0; j < V; j++) probs[j] /= sum;
      // sample
      const r = Math.random();
      let acc = 0;
      let pick = V - 1;
      for (let j = 0; j < V; j++) {
        acc += probs[j];
        if (r <= acc) { pick = j; break; }
      }
      out.push(pick);
      if (onToken) onToken(pick);
    }
    return out;
  }

  // count parameters
  numParams(): number {
    let n = this.wte.data.length + this.wpe.data.length;
    for (const b of this.blocks) {
      n += b.Wq.data.length + b.bq.data.length;
      n += b.Wk.data.length + b.bk.data.length;
      n += b.Wv.data.length + b.bv.data.length;
      n += b.Wo.data.length + b.bo.data.length;
      n += b.W1.data.length + b.b1.data.length;
      n += b.W2.data.length + b.b2.data.length;
      n += b.ln1g.data.length + b.ln1b.data.length;
      n += b.ln2g.data.length + b.ln2b.data.length;
    }
    n += this.lnFg.data.length + this.lnFb.data.length;
    return n;
  }
}

// ===== Helpers =====

function onesVec(C: number): Matrix {
  const data = new Float32Array(C);
  data.fill(1);
  return { data, rows: 1, cols: C };
}

function sliceCols(M: Matrix, start: number, count: number): Matrix {
  const out = new Float32Array(M.rows * count);
  for (let i = 0; i < M.rows; i++)
    for (let j = 0; j < count; j++)
      out[i * count + j] = M.data[i * M.cols + start + j];
  return { data: out, rows: M.rows, cols: count };
}

function rowSoftmax(M: Matrix): Matrix {
  const out = new Float32Array(M.data.length);
  const R = M.rows, C = M.cols;
  for (let i = 0; i < R; i++) {
    let maxv = -Infinity;
    for (let j = 0; j < C; j++) if (M.data[i * C + j] > maxv) maxv = M.data[i * C + j];
    let sum = 0;
    for (let j = 0; j < C; j++) { const e = Math.exp(M.data[i * C + j] - maxv); out[i * C + j] = e; sum += e; }
    const inv = 1 / sum;
    for (let j = 0; j < C; j++) out[i * C + j] *= inv;
  }
  return { data: out, rows: R, cols: C };
}

// Layer norm with caches for backward
function layerNormFull(M: Matrix, gamma: Matrix, beta: Matrix, eps = 1e-5):
  { out: Matrix; mean: Float32Array; varc: Float32Array } {
  const R = M.rows, C = M.cols;
  const out = new Float32Array(R * C);
  const mean = new Float32Array(R);
  const varc = new Float32Array(R);
  for (let i = 0; i < R; i++) {
    let mu = 0;
    for (let j = 0; j < C; j++) mu += M.data[i * C + j];
    mu /= C;
    let v = 0;
    for (let j = 0; j < C; j++) { const d = M.data[i * C + j] - mu; v += d * d; }
    v /= C;
    mean[i] = mu;
    varc[i] = v;
    const inv = 1 / Math.sqrt(v + eps);
    for (let j = 0; j < C; j++) {
      const norm = (M.data[i * C + j] - mu) * inv;
      out[i * C + j] = norm * gamma.data[j] + beta.data[j];
    }
  }
  return { out: { data: out, rows: R, cols: C }, mean, varc };
}

function layerNormBackward(dOut: Matrix, x: Matrix, mean: Float32Array, varc: Float32Array, gamma: Matrix, eps = 1e-5): Matrix {
  const R = x.rows, C = x.cols;
  const dx = new Float32Array(R * C);
  for (let i = 0; i < R; i++) {
    const inv = 1 / Math.sqrt(varc[i] + eps);
    // dxhat = dOut * gamma
    // We need: dx = (1/N) * inv * (N*dxhat - sum(dxhat) - xhat*sum(dxhat*xhat))
    let sumDxhat = 0, sumDxhatXhat = 0;
    const dxhat = new Float32Array(C);
    const xhat = new Float32Array(C);
    for (let j = 0; j < C; j++) {
      xhat[j] = (x.data[i * C + j] - mean[i]) * inv;
      dxhat[j] = dOut.data[i * C + j] * gamma.data[j];
      sumDxhat += dxhat[j];
      sumDxhatXhat += dxhat[j] * xhat[j];
    }
    for (let j = 0; j < C; j++) {
      dx[i * C + j] = (1 / C) * inv * (C * dxhat[j] - sumDxhat - xhat[j] * sumDxhatXhat);
    }
  }
  return { data: dx, rows: R, cols: C };
}

function lnGammaGrad(dOut: Matrix, x: Matrix, mean: Float32Array, varc: Float32Array, eps = 1e-5): Matrix {
  const R = x.rows, C = x.cols;
  const g = new Float32Array(C);
  for (let i = 0; i < R; i++) {
    const inv = 1 / Math.sqrt(varc[i] + eps);
    for (let j = 0; j < C; j++) g[j] += dOut.data[i * C + j] * (x.data[i * C + j] - mean[i]) * inv;
  }
  return { data: g, rows: 1, cols: C };
}

function lnBetaGrad(dOut: Matrix): Matrix {
  const R = dOut.rows, C = dOut.cols;
  const g = new Float32Array(C);
  for (let i = 0; i < R; i++)
    for (let j = 0; j < C; j++) g[j] += dOut.data[i * C + j];
  return { data: g, rows: 1, cols: C };
}

function biasGrad(dOut: Matrix): Matrix {
  const R = dOut.rows, C = dOut.cols;
  const g = new Float32Array(C);
  for (let i = 0; i < R; i++)
    for (let j = 0; j < C; j++) g[j] += dOut.data[i * C + j];
  return { data: g, rows: 1, cols: C };
}

function geluBackward(dOut: Matrix, x: Matrix): Matrix {
  const out = new Float32Array(dOut.data.length);
  const k = Math.sqrt(2 / Math.PI);
  for (let i = 0; i < x.data.length; i++) {
    const xi = x.data[i];
    const inner = k * (xi + 0.044715 * xi * xi * xi);
    const tanhInner = Math.tanh(inner);
    const sech2 = 1 - tanhInner * tanhInner;
    const dInner = k * (1 + 3 * 0.044715 * xi * xi);
    const dGelu = 0.5 * (1 + tanhInner) + 0.5 * xi * sech2 * dInner;
    out[i] = dOut.data[i] * dGelu;
  }
  return { data: out, rows: dOut.rows, cols: dOut.cols };
}

function addM(A: Matrix, B: Matrix): Matrix {
  const out = new Float32Array(A.data.length);
  for (let i = 0; i < A.data.length; i++) out[i] = A.data[i] + B.data[i];
  return { data: out, rows: A.rows, cols: A.cols };
}
