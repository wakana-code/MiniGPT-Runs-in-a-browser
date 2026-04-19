// A small built-in dataset (Tiny Shakespeare-style snippet) so users can train without uploading anything.
import { SIBUGAKI_DATASET } from "./conversations";

export const SAMPLE_DATASETS: Record<string, string> = {
  "💬 Sibugaki AI 会話データ": SIBUGAKI_DATASET,
  "Tiny Shakespeare": `First Citizen:
Before we proceed any further, hear me speak.

All:
Speak, speak.

First Citizen:
You are all resolved rather to die than to famish?

All:
Resolved. resolved.

First Citizen:
First, you know Caius Marcius is chief enemy to the people.

All:
We know't, we know't.

First Citizen:
Let us kill him, and we'll have corn at our own price.
Is't a verdict?

All:
No more talking on't; let it be done: away, away!

Second Citizen:
One word, good citizens.

First Citizen:
We are accounted poor citizens, the patricians good.
What authority surfeits on would relieve us: if they
would yield us but the superfluity, while it were
wholesome, we might guess they relieved us humanely;
but they think we are too dear: the leanness that
afflicts us, the object of our misery, is as an
inventory to particularise their abundance; our
sufferance is a gain to them. Let us revenge this with
our pikes, ere we become rakes: for the gods know I
speak this in hunger for bread, not in thirst for revenge.
`,
  "日本語俳句": `古池や　蛙飛び込む　水の音
夏草や　兵どもが　夢の跡
閑さや　岩にしみ入る　蝉の声
五月雨を　集めて早し　最上川
柿食えば　鐘が鳴るなり　法隆寺
雪とけて　村いっぱいの　子どもかな
痩蛙　負けるな一茶　これにあり
雀の子　そこのけそこのけ　御馬が通る
やれ打つな　蝿が手をすり　足をする
名月を　取ってくれろと　泣く子かな
菜の花や　月は東に　日は西に
春の海　ひねもすのたり　のたりかな
朝顔に　釣瓶取られて　もらい水
これがまあ　終の住処か　雪五尺
我と来て　遊べや親の　ない雀
`,
  "ABC nursery": `the quick brown fox jumps over the lazy dog.
the quick brown fox jumps over the lazy dog.
abcdefghijklmnopqrstuvwxyz
twinkle twinkle little star, how i wonder what you are.
up above the world so high, like a diamond in the sky.
mary had a little lamb, its fleece was white as snow.
and everywhere that mary went, the lamb was sure to go.
hickory dickory dock, the mouse ran up the clock.
the clock struck one, the mouse ran down, hickory dickory dock.
`,
};
