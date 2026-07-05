# Chapter 4: KV Cache

At the end of Chapter 3 I asked you to run the story demo and feel it slow
down as the text grows. This chapter removes that slowdown. Nothing about the
model changes: same weights, same math, same story down to the last token. We
only stop computing things we already computed. This is the single most
important inference optimization in every real LLM engine, and by the end of
this chapter you will have built it.

## Why the naive loop is slow, exactly

Look at what Chapter 3's Generate does to produce ONE new token:

```
logits = Forward(tokens)     the WHOLE sequence, from scratch
```

To predict token 51 it recomputes the forward pass over all 50 existing
tokens, uses the last column of logits, and throws the rest away. Then, to
predict token 52, it recomputes all 51. Token 3's keys and values get computed
again, and again, and again, hundreds of times over one story, and they come
out IDENTICAL every single time.

Why identical? Because of causality, the property you built in Chapter 1's
`MaskCausal`: a token only ever attends BACKWARD. Nothing a future token does
can change what token 3's key and value vectors are. The weights are frozen,
token 3's position is frozen, so its k and v at every layer are frozen too.
Recomputing them is pure waste.

The fix writes itself once you see it: compute each token's k and v ONCE, and
keep them.

## What the cache is

Two tensors, allocated per generation:

```
keyCache   = {kv_dim, seq_len, n_layers} = {288, 256, 6}
valueCache = {kv_dim, seq_len, n_layers} = {288, 256, 6}
```

Read the axes: column p of layer plane l holds the key (or value) vector that
token at position p produced at layer l. `kv_dim` is `headSize * n_kv_heads`,
which for stories15M is `48 * 6 = 288`, all heads concatenated, the same
side-by-side layout you sliced apart in Chapter 2.

Price of the whole optimization:
`288 * 256 * 6 floats * 4 bytes * 2 tensors = 3.5 MB`. That is the entire cost
of never recomputing attention history again.

And why is there no query cache? Ask what old queries would ever be used FOR.
A query exists to let a token look around at the moment it arrives. Once token
3 has looked, its question is answered and never asked again; only its k and v
are ever needed by others. The newest token is the only one asking.

## Attention against the cache

The new attention op processes ONE token at position `pos`:

1. Write the token's k and v into cache column `pos`.
2. Its query attends over cache columns `[0, pos]`, per head.

Two details in there are worth slowing down for.

**There is no causal mask anymore.** Chapter 2 computed a full
seqLen-by-seqLen score matrix and masked the upper triangle with negative
infinity. Here the cache only CONTAINS the past: columns beyond `pos` are
whatever uninitialized memory happens to hold, and we simply never slice them
in. `Slice(cache, 1, 0, pos + 1)` IS the causal mask, expressed as a shape.

**The double Slice.** To get head h's view of the usable cache:

```
Slice(Slice(kCache, 0, h * headDim, headDim), 1, 0, pos + 1)
```

The inner Slice cuts the head's 48 features (axis 0), the outer cuts positions
0 through pos (axis 1). Both are zero-copy views from Chapter 2, so "give me
head 2's keys for every token so far" moves not a single float. This is the
payoff of building tensors on strides back in Chapter 1: the cache trick costs
nothing but bookkeeping.

The same view mechanics work in the other direction: the block will hand the
attention op a `Slice` of the full cache (this layer's plane), and when the op
writes k and v through that view, it writes the real cache. A view is not a
copy, in both directions.

## RoPE finally uses positionOffset

Since Chapter 1, `RoPE` has carried a `positionOffset` argument that has been
0 every time. Now it matters. The cached forward pass processes one token as a
`{dim, 1}` tensor: one column, row index 0. Without help, RoPE would read that
as "position 0" and rotate every token like it was the first word. The token's
real position must be passed in:

```
q = ApplyRoPE(Project(h, wq, layer), n_heads, 10000.0f, pos)
```

Forgetting the offset is this chapter's classic bug, and it is a sneaky one:
nothing crashes, the shapes are all right, and the story comes out as word
soup. If your Task 2 numbers are wrong, check this first.

## Prefill, then decode

The generation loop gets two phases:

```
PREFILL   feed each prompt token through ForwardWithKVCache, positions 0, 1, 2...
          purpose: fill the cache; only the LAST logits are kept
DECODE    sample a token from the logits, feed it back in at the next position,
          repeat
```

Every real inference engine has these exact two phases (when a chatbot pauses
before its first word, that is prefill). One wrinkle to notice: the cached
forward returns logits of shape `{vocab_size, 1}`, a single column, so the
position argument to Sample is 0, not `tokens.size() - 1`.

## The contract: identical output

Greedy decoding is deterministic, so there is nowhere to hide: your
GenerateWithKVCache at temperature 0 must produce EXACTLY the token ids that
Chapter 3's naive Generate produces. The tests enforce this all the way up the
stack: Task 1 is checked column-by-column against Chapter 2's
MultiHeadAttention, Tasks 2 and 3 against the same golden checkpoint values
Chapter 2 used, and Task 4 runs BOTH generators and compares them token for
token, with a stopwatch on each.

## Your tasks

Task 1 in `TensorOps.h`, tasks 2-4 in `Model.h`. The exact recipe is in the
comment above each function.

1. `MultiHeadAttentionWithKVCache` (write k,v into the cache, attend one query
   over the cached history, per head)
2. `AttentionBlockWithKVCache` (the cached twin of Chapter 2's AttentionBlock;
   pass `pos` to RoPE)
3. `ForwardWithKVCache` (one token in, one column of logits out; FeedForwardBlock
   is reused untouched, it never looks at other tokens)
4. `GenerateWithKVCache` (prefill the prompt, then decode)

## Build and run

From this chapter's folder:

```bash
cmake -B build
cmake --build build --config Release
./build/ch4_tests               # Linux/macOS  (add -DCMAKE_BUILD_TYPE=Release to the first cmake)
# .\build\Release\ch4_tests.exe # Windows / MSVC
```

Build Release again: Task 4 times the naive path against yours, and the
comparison should be honest. When it passes you will see something like:

```
[       OK ] KVCache.GenerateWithKVCache
    same 8 tokens: naive 5000 ms, kv-cache 800 ms (6.3x)
```

The exact numbers depend on your machine; the shape of the result does not.
And the speedup GROWS with length: 8 tokens is a modest win, but at token 200
the naive path recomputes a 200-token forward pass per token while yours still
does one. Try it:

```
./build/ch4_story "One day a dragon" 0.9 200
```

then run Chapter 3's `ch3_story` with the same arguments and watch it crawl.

## Where you are now

This was the last chapter. What you built across the four of them, a strided
tensor library, the full Llama 2 forward pass, sampling, and a KV cache, is
the complete engine that lives in `header/` on `master`, the same code that
produced every reference value you tested against. The remaining piece you
took on faith, how `tokenizer.bin` and `stories15M.bin` are actually parsed,
lives in the commented sources themselves: `Tokenizer.h` and the `Model`
constructor are short reads whenever you want them.
