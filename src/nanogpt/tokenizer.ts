// Character-level tokenizer (like Karpathy's nanoGPT char-level)
export class CharTokenizer {
  vocab: string[];
  stoi: Map<string, number>;

  constructor(text: string) {
    const chars = Array.from(new Set(text)).sort();
    this.vocab = chars;
    this.stoi = new Map();
    chars.forEach((c, i) => this.stoi.set(c, i));
  }

  get vocabSize(): number {
    return this.vocab.length;
  }

  encode(text: string): number[] {
    const out: number[] = [];
    for (const c of text) {
      const id = this.stoi.get(c);
      if (id !== undefined) out.push(id);
    }
    return out;
  }

  decode(ids: number[]): string {
    return ids.map((i) => this.vocab[i] ?? "").join("");
  }
}
