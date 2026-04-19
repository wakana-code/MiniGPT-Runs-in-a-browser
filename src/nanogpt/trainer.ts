import { GPT } from "./model";

// Get a random batch of (input, target) pairs from token stream
export function getBatch(data: number[], blockSize: number, batchSize: number): { x: number[][]; y: number[][] } {
  const x: number[][] = [];
  const y: number[][] = [];
  for (let b = 0; b < batchSize; b++) {
    const i = Math.floor(Math.random() * (data.length - blockSize - 1));
    x.push(data.slice(i, i + blockSize));
    y.push(data.slice(i + 1, i + blockSize + 1));
  }
  return { x, y };
}

// Train one step on a single sample (the "batch size" is achieved by gradient accumulation over multiple samples).
export async function trainStep(
  model: GPT,
  data: number[],
  blockSize: number,
  batchSize: number,
  lr: number
): Promise<number> {
  const { x, y } = getBatch(data, blockSize, batchSize);
  let totalLoss = 0;
  // Accumulate grads across batch
  const accGrads = new Map<any, any>();
  for (let b = 0; b < batchSize; b++) {
    const cache = model.forward(x[b]);
    totalLoss += model.computeLoss(cache, y[b]);
    const { grads } = model.backward(cache, y[b]);
    grads.forEach((g, p) => {
      const existing = accGrads.get(p);
      if (!existing) {
        // copy
        const cp = { data: new Float32Array(g.data), rows: g.rows, cols: g.cols };
        accGrads.set(p, cp);
      } else {
        for (let i = 0; i < g.data.length; i++) existing.data[i] += g.data[i];
      }
    });
  }
  // average grads
  accGrads.forEach((g) => {
    for (let i = 0; i < g.data.length; i++) g.data[i] /= batchSize;
  });
  // gradient clipping (norm 1.0)
  let totalNorm = 0;
  accGrads.forEach((g) => { for (let i = 0; i < g.data.length; i++) totalNorm += g.data[i] * g.data[i]; });
  totalNorm = Math.sqrt(totalNorm);
  const maxNorm = 1.0;
  if (totalNorm > maxNorm) {
    const factor = maxNorm / (totalNorm + 1e-6);
    accGrads.forEach((g) => { for (let i = 0; i < g.data.length; i++) g.data[i] *= factor; });
  }
  model.applyGrads(accGrads, lr);
  return totalLoss / batchSize;
}
