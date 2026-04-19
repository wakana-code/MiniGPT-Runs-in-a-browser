import { useEffect, useMemo, useRef, useState } from "react";
import { GPT } from "./nanogpt/model";
import { CharTokenizer } from "./nanogpt/tokenizer";
import { trainStep } from "./nanogpt/trainer";
import { SAMPLE_DATASETS } from "./nanogpt/sample";
import { formatPrompt, STOP_SEQUENCE } from "./nanogpt/conversations";

type ChatMessage = { role: "user" | "assistant" | "system"; content: string };

export default function App() {
  // Dataset & tokenizer
  const [datasetName, setDatasetName] = useState<string>("💬 Sibugaki AI 会話データ");
  const [customText, setCustomText] = useState<string>("");
  const [text, setText] = useState<string>(SAMPLE_DATASETS["💬 Sibugaki AI 会話データ"]);

  // Hyperparameters - chat-tuned defaults
  const [nLayer, setNLayer] = useState(3);
  const [nHead, setNHead] = useState(4);
  const [nEmbd, setNEmbd] = useState(96);
  const [blockSize, setBlockSize] = useState(64);
  const [batchSize, setBatchSize] = useState(8);
  const [lr, setLr] = useState(0.003);

  // Sampling
  const [temperature, setTemperature] = useState(0.8);
  const [topK, setTopK] = useState(40);
  const [maxTokens, setMaxTokens] = useState(150);
  // Chat mode: format input as 人間：xxx\nSibugaki： and stop at next 人間：
  const [chatMode, setChatMode] = useState(true);

  // Training state
  const modelRef = useRef<GPT | null>(null);
  const tokenizerRef = useRef<CharTokenizer | null>(null);
  const dataIdsRef = useRef<number[]>([]);
  const [isTraining, setIsTraining] = useState(false);
  const trainingRef = useRef(false);
  const [step, setStep] = useState(0);
  const [losses, setLosses] = useState<number[]>([]);
  const [paramCount, setParamCount] = useState(0);
  const [vocabSize, setVocabSize] = useState(0);

  // Chat
  const [messages, setMessages] = useState<ChatMessage[]>([
    { role: "system", content: "👋 ようこそ！ブラウザで完全に動く nanoGPT（ChatGPT と同じ Transformer 構造）です。\n\n💬 デフォルトで「Sibugaki AI 会話データセット」を読み込んでいます。\n\n手順：\n  1. 左の「🔄 モデル初期化」を押す\n  2. 「▶ 学習開始」で訓練（loss が 1.5 以下になるまで放置推奨）\n  3. 下にメッセージを入力してSibugakiと会話！\n\n💡 チャットモード ON のとき、あなたの入力は裏で「人間：xxx\\nSibugaki：」に整形されてモデルに渡され、続きが Sibugaki の返答として生成されます。" }
  ]);
  const [input, setInput] = useState("");
  const [isGenerating, setIsGenerating] = useState(false);
  const chatEndRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  // Update text when dataset changes
  useEffect(() => {
    if (datasetName === "カスタム") {
      setText(customText);
    } else {
      setText(SAMPLE_DATASETS[datasetName] ?? "");
    }
  }, [datasetName, customText]);

  const datasetStats = useMemo(() => {
    const chars = new Set(text).size;
    return { chars: text.length, unique: chars };
  }, [text]);

  function initModel() {
    if (text.length < blockSize + 2) {
      alert("テキストが短すぎます。もっと長いテキストを使ってください。");
      return;
    }
    const tok = new CharTokenizer(text);
    const ids = tok.encode(text);
    tokenizerRef.current = tok;
    dataIdsRef.current = ids;
    setVocabSize(tok.vocabSize);

    // ensure nEmbd divisible by nHead
    let embd = nEmbd;
    if (embd % nHead !== 0) {
      embd = Math.ceil(embd / nHead) * nHead;
      setNEmbd(embd);
    }
    const model = new GPT({
      vocabSize: tok.vocabSize,
      blockSize,
      nEmbd: embd,
      nHead,
      nLayer,
    });
    modelRef.current = model;
    setParamCount(model.numParams());
    setStep(0);
    setLosses([]);
    setMessages([
      { role: "system", content: `✅ モデル初期化完了！\n\n📊 語彙数: ${tok.vocabSize} 文字\n🧮 パラメータ数: ${model.numParams().toLocaleString()}\n📚 学習データ: ${text.length.toLocaleString()} 文字\n\n「学習開始」ボタンを押して訓練してください。最初は意味のない文字列が出ますが、loss が下がるにつれて学習元データに似た文章が生成されるようになります。` }
    ]);
  }

  async function startTraining() {
    if (!modelRef.current) {
      alert("先にモデルを初期化してください。");
      return;
    }
    if (trainingRef.current) return;
    trainingRef.current = true;
    setIsTraining(true);

    const model = modelRef.current;
    const data = dataIdsRef.current;
    while (trainingRef.current) {
      const loss = await trainStep(model, data, blockSize, batchSize, lr);
      setStep((s) => s + 1);
      setLosses((arr) => {
        const next = [...arr, loss];
        return next.length > 200 ? next.slice(next.length - 200) : next;
      });
      // yield to UI
      await new Promise((r) => setTimeout(r, 0));
    }
    setIsTraining(false);
  }

  function stopTraining() {
    trainingRef.current = false;
  }

  async function handleSend() {
    const model = modelRef.current;
    const tok = tokenizerRef.current;
    if (!model || !tok) {
      alert("先にモデルを初期化してください。");
      return;
    }
    if (!input.trim()) return;
    const userMsg = input;
    setInput("");
    setMessages((m) => [...m, { role: "user", content: userMsg }]);
    setIsGenerating(true);

    // In chat mode: wrap as 「人間：xxx\nSibugaki：」 so the model continues an assistant turn.
    const fullPrompt = chatMode ? formatPrompt(userMsg) : userMsg;
    const promptIds = tok.encode(fullPrompt);
    if (promptIds.length === 0) {
      setMessages((m) => [...m, { role: "assistant", content: "(プロンプトが学習データの語彙に含まれていません)" }]);
      setIsGenerating(false);
      return;
    }

    // Add an empty assistant message we will fill in
    setMessages((m) => [...m, { role: "assistant", content: "" }]);

    // Generate token by token, updating UI
    const out = [...promptIds];
    let stopped = false;
    for (let i = 0; i < maxTokens && !stopped; i++) {
      const ctx = out.slice(Math.max(0, out.length - model.cfg.blockSize));
      const cache = model.forward(ctx);
      const V = cache.logits.cols;
      const last = new Float32Array(cache.logits.data.subarray((ctx.length - 1) * V, ctx.length * V));
      for (let j = 0; j < V; j++) last[j] /= Math.max(temperature, 1e-6);
      if (topK > 0 && topK < V) {
        const sorted = Array.from(last).map((v, idx) => ({ v, idx })).sort((a, b) => b.v - a.v);
        const thresh = sorted[Math.min(topK, sorted.length) - 1].v;
        for (let j = 0; j < V; j++) if (last[j] < thresh) last[j] = -Infinity;
      }
      let maxv = -Infinity;
      for (let j = 0; j < V; j++) if (last[j] > maxv) maxv = last[j];
      let sum = 0;
      const probs = new Float32Array(V);
      for (let j = 0; j < V; j++) { probs[j] = Math.exp(last[j] - maxv); sum += probs[j]; }
      for (let j = 0; j < V; j++) probs[j] /= sum;
      const r = Math.random();
      let acc = 0;
      let pick = V - 1;
      for (let j = 0; j < V; j++) { acc += probs[j]; if (r <= acc) { pick = j; break; } }
      out.push(pick);

      // Decode just the newly generated portion (after the prompt).
      let generatedText = tok.decode(out.slice(promptIds.length));

      // In chat mode, stop when the model starts a new "人間：" turn.
      if (chatMode) {
        const stopIdx = generatedText.indexOf(STOP_SEQUENCE);
        if (stopIdx >= 0) {
          generatedText = generatedText.slice(0, stopIdx);
          stopped = true;
        }
        // Also stop on a double newline (end of assistant turn).
        const dblIdx = generatedText.indexOf("\n\n");
        if (dblIdx >= 0) {
          generatedText = generatedText.slice(0, dblIdx);
          stopped = true;
        }
      }

      setMessages((m) => {
        const copy = [...m];
        copy[copy.length - 1] = { role: "assistant", content: generatedText || "…" };
        return copy;
      });
      if (i % 4 === 0) await new Promise((r) => setTimeout(r, 0));
    }
    setIsGenerating(false);
  }

  const lastLoss = losses.length > 0 ? losses[losses.length - 1] : null;
  const minLoss = losses.length > 0 ? Math.min(...losses) : null;

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-950 via-slate-900 to-indigo-950 text-slate-100">
      {/* Header */}
      <header className="border-b border-white/10 backdrop-blur sticky top-0 z-10 bg-slate-950/70">
        <div className="max-w-7xl mx-auto px-4 py-3 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="w-9 h-9 rounded-lg bg-gradient-to-br from-indigo-400 to-purple-600 flex items-center justify-center font-bold text-lg shadow-lg shadow-indigo-500/30">
              n
            </div>
            <div>
              <h1 className="text-lg font-bold tracking-tight">nanoGPT <span className="text-indigo-300 text-xs font-mono">in your browser</span></h1>
              <p className="text-xs text-slate-400">Transformer を一から JS で実装・学習・生成</p>
            </div>
          </div>
          <div className="hidden md:flex items-center gap-4 text-xs text-slate-300">
            <Stat label="vocab" value={vocabSize || "-"} />
            <Stat label="params" value={paramCount ? paramCount.toLocaleString() : "-"} />
            <Stat label="step" value={step} />
            <Stat label="loss" value={lastLoss ? lastLoss.toFixed(3) : "-"} />
          </div>
        </div>
      </header>

      <div className="max-w-7xl mx-auto px-4 py-4 grid grid-cols-1 lg:grid-cols-[360px_1fr] gap-4">
        {/* Sidebar */}
        <aside className="space-y-4">
          <Card title="📚 1. データセット">
            <div className="space-y-2">
              <select
                value={datasetName}
                onChange={(e) => setDatasetName(e.target.value)}
                className="w-full bg-slate-800/70 border border-white/10 rounded-lg px-3 py-2 text-sm"
              >
                {Object.keys(SAMPLE_DATASETS).map((k) => (
                  <option key={k} value={k}>{k}</option>
                ))}
                <option value="カスタム">カスタム（自分で入力）</option>
              </select>
              {datasetName === "カスタム" && (
                <textarea
                  value={customText}
                  onChange={(e) => setCustomText(e.target.value)}
                  placeholder="ここに学習させたい文章をペースト..."
                  rows={5}
                  className="w-full bg-slate-800/70 border border-white/10 rounded-lg px-3 py-2 text-xs font-mono"
                />
              )}
              <div className="text-xs text-slate-400">
                {datasetStats.chars.toLocaleString()} 文字 / ユニーク {datasetStats.unique} 種
              </div>
            </div>
          </Card>

          <Card title="🧠 2. モデル構造">
            <div className="grid grid-cols-2 gap-2">
              <Field label="層数 nLayer" value={nLayer} onChange={(v) => setNLayer(clamp(v, 1, 6))} min={1} max={6} />
              <Field label="ヘッド nHead" value={nHead} onChange={(v) => setNHead(clamp(v, 1, 8))} min={1} max={8} />
              <Field label="埋込次元 nEmbd" value={nEmbd} onChange={(v) => setNEmbd(clamp(v, 8, 256))} min={8} max={256} step={8} />
              <Field label="文脈長 blockSize" value={blockSize} onChange={(v) => setBlockSize(clamp(v, 8, 128))} min={8} max={128} />
            </div>
            <button
              onClick={initModel}
              disabled={isTraining}
              className="w-full mt-3 bg-gradient-to-r from-indigo-500 to-purple-600 hover:from-indigo-400 hover:to-purple-500 disabled:opacity-50 disabled:cursor-not-allowed rounded-lg py-2 text-sm font-medium shadow-lg shadow-indigo-500/20"
            >
              🔄 モデル初期化
            </button>
          </Card>

          <Card title="🏃 3. 学習">
            <div className="grid grid-cols-2 gap-2">
              <Field label="batchSize" value={batchSize} onChange={(v) => setBatchSize(clamp(v, 1, 16))} min={1} max={16} />
              <FieldFloat label="learning rate" value={lr} onChange={(v) => setLr(v)} step={0.0005} />
            </div>
            <div className="flex gap-2 mt-3">
              {!isTraining ? (
                <button
                  onClick={startTraining}
                  className="flex-1 bg-emerald-500 hover:bg-emerald-400 rounded-lg py-2 text-sm font-medium shadow-lg shadow-emerald-500/20"
                >
                  ▶ 学習開始
                </button>
              ) : (
                <button
                  onClick={stopTraining}
                  className="flex-1 bg-red-500 hover:bg-red-400 rounded-lg py-2 text-sm font-medium shadow-lg shadow-red-500/20"
                >
                  ⏸ 停止
                </button>
              )}
            </div>
            {losses.length > 0 && (
              <div className="mt-3">
                <div className="text-xs text-slate-400 mb-1 flex justify-between">
                  <span>loss curve (最後 {losses.length} step)</span>
                  <span>min: {minLoss?.toFixed(3)}</span>
                </div>
                <LossChart losses={losses} />
              </div>
            )}
          </Card>

          <Card title="🎲 4. 生成パラメータ">
            <label className="flex items-center gap-2 mb-3 text-xs cursor-pointer select-none p-2 bg-indigo-500/10 border border-indigo-400/20 rounded-lg">
              <input
                type="checkbox"
                checked={chatMode}
                onChange={(e) => setChatMode(e.target.checked)}
                className="accent-indigo-400"
              />
              <span>
                💬 <b>Sibugaki チャットモード</b>
                <div className="text-[10px] text-slate-400 mt-0.5">
                  入力を「人間：…\nSibugaki：」に整形して続きを生成
                </div>
              </span>
            </label>
            <FieldFloat label="temperature" value={temperature} onChange={(v) => setTemperature(Math.max(0.1, v))} step={0.1} />
            <Field label="top-k (0=off)" value={topK} onChange={(v) => setTopK(clamp(v, 0, 200))} min={0} max={200} />
            <Field label="max tokens" value={maxTokens} onChange={(v) => setMaxTokens(clamp(v, 10, 1000))} min={10} max={1000} step={10} />
          </Card>

          <div className="text-xs text-slate-500 leading-relaxed px-1">
            このページは外部 API を一切使いません。Transformer (Multi-Head Causal Self-Attention + LayerNorm + GELU MLP + 重み共有 LM Head) と Adam optimizer を素の TypeScript で実装し、ブラウザ内で順伝播・逆伝播・サンプリングを行っています。
          </div>
        </aside>

        {/* Chat */}
        <main className="bg-slate-900/40 border border-white/10 rounded-2xl flex flex-col min-h-[80vh]">
          <div className="flex-1 overflow-y-auto p-6 space-y-4">
            {messages.map((m, i) => (
              <MessageBubble key={i} msg={m} />
            ))}
            {isGenerating && (
              <div className="text-xs text-slate-400 animate-pulse">⚡ 生成中...</div>
            )}
            <div ref={chatEndRef} />
          </div>
          <div className="border-t border-white/10 p-3">
            <div className="flex gap-2">
              <input
                value={input}
                onChange={(e) => setInput(e.target.value)}
                onKeyDown={(e) => { if (e.key === "Enter" && !isGenerating) handleSend(); }}
                placeholder={modelRef.current ? (chatMode ? "Sibugakiに話しかける… (例: こんにちは)" : "プロンプトを入力 (続きを生成します)") : "← まず左で『モデル初期化』してください"}
                disabled={isGenerating || !modelRef.current}
                className="flex-1 bg-slate-800/70 border border-white/10 rounded-lg px-4 py-3 text-sm focus:outline-none focus:border-indigo-400/50 disabled:opacity-50"
              />
              <button
                onClick={handleSend}
                disabled={isGenerating || !modelRef.current || !input.trim()}
                className="bg-gradient-to-r from-indigo-500 to-purple-600 hover:from-indigo-400 hover:to-purple-500 disabled:opacity-50 disabled:cursor-not-allowed rounded-lg px-5 py-3 text-sm font-medium"
              >
                送信 →
              </button>
            </div>
            <div className="mt-2 text-[11px] text-slate-500">
              ヒント: loss が 1.8 以下になると Sibugaki が会話らしい返答を始めます。学習を続けながら何度でも生成できます。
            </div>
          </div>
        </main>
      </div>
    </div>
  );
}

function Card({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <div className="bg-slate-900/40 border border-white/10 rounded-2xl p-4">
      <div className="text-sm font-semibold mb-3">{title}</div>
      {children}
    </div>
  );
}

function Stat({ label, value }: { label: string; value: React.ReactNode }) {
  return (
    <div className="flex flex-col items-center px-2">
      <div className="text-[10px] uppercase tracking-wider text-slate-500">{label}</div>
      <div className="font-mono text-sm text-indigo-200">{value}</div>
    </div>
  );
}

function Field({ label, value, onChange, min, max, step = 1 }:
  { label: string; value: number; onChange: (v: number) => void; min?: number; max?: number; step?: number }) {
  return (
    <label className="block">
      <span className="text-[11px] text-slate-400">{label}</span>
      <input
        type="number"
        value={value}
        min={min} max={max} step={step}
        onChange={(e) => onChange(parseInt(e.target.value || "0"))}
        className="w-full bg-slate-800/70 border border-white/10 rounded-lg px-2 py-1.5 text-sm font-mono"
      />
    </label>
  );
}

function FieldFloat({ label, value, onChange, step = 0.1 }:
  { label: string; value: number; onChange: (v: number) => void; step?: number }) {
  return (
    <label className="block">
      <span className="text-[11px] text-slate-400">{label}</span>
      <input
        type="number"
        value={value}
        step={step}
        onChange={(e) => onChange(parseFloat(e.target.value || "0"))}
        className="w-full bg-slate-800/70 border border-white/10 rounded-lg px-2 py-1.5 text-sm font-mono"
      />
    </label>
  );
}

function MessageBubble({ msg }: { msg: ChatMessage }) {
  if (msg.role === "system") {
    return (
      <div className="bg-indigo-500/10 border border-indigo-400/20 rounded-xl p-4 text-sm whitespace-pre-wrap text-indigo-100">
        {msg.content}
      </div>
    );
  }
  const isUser = msg.role === "user";
  return (
    <div className={`flex ${isUser ? "justify-end" : "justify-start"}`}>
      <div className={`max-w-[85%] rounded-2xl px-4 py-3 text-sm whitespace-pre-wrap font-mono leading-relaxed ${
        isUser ? "bg-indigo-500 text-white" : "bg-slate-800/80 border border-white/5 text-slate-100"
      }`}>
        {msg.content || <span className="text-slate-500 italic">...</span>}
      </div>
    </div>
  );
}

function LossChart({ losses }: { losses: number[] }) {
  const w = 320, h = 80;
  if (losses.length < 2) return <div className="h-20 flex items-center justify-center text-xs text-slate-500">データ蓄積中...</div>;
  const min = Math.min(...losses);
  const max = Math.max(...losses);
  const range = max - min || 1;
  const pts = losses.map((l, i) => {
    const x = (i / (losses.length - 1)) * w;
    const y = h - ((l - min) / range) * (h - 4) - 2;
    return `${x.toFixed(1)},${y.toFixed(1)}`;
  }).join(" ");
  return (
    <svg viewBox={`0 0 ${w} ${h}`} className="w-full h-20 bg-slate-950/50 rounded">
      <polyline fill="none" stroke="#a78bfa" strokeWidth="1.5" points={pts} />
    </svg>
  );
}

function clamp(v: number, lo: number, hi: number) {
  if (isNaN(v)) return lo;
  return Math.max(lo, Math.min(hi, v));
}
