# Chapter 2: The Forward Pass

In Chapter 1 you wrote the math operations. In this chapter you arrange them into
the actual model. No toy data this time: you will load the real `stories15M`
weights, push a real prompt through your own code, and end with the model
predicting the next token of "Once upon a time".

## What is given, what you build

This chapter's folder contains:

- `Tensor.h`: unchanged from Chapter 1.
- `TensorOps.h`: the Chapter 1 ops, **completed** (they are the answers; do
  Chapter 1 first), plus five new empty ops for you to fill in.
- `Model.h`: the model config, the weight tensors, and the loading code, all
  **given**. The file format is plumbing, not transformer knowledge; the
  loading code is short and commented if you are curious. Your three remaining
  tasks are at the bottom of this file.
- `main.cpp`: the tests. Tasks 1-5 use small hand-checked tensors like Chapter 1.
  Tasks 6-8 run against the real weights and compare your numbers to values
  computed by the reference implementation.

You need the model file. From the repository root:

```bash
python models/download.py    # fetches models/stories15M.bin (~58 MB)
```

## The model, in numbers

When `Model.h` loads `stories15M.bin` it reads this config:

```
dim = 288          features per token
hidden_dim = 768   feed-forward inner width
n_layers = 6       transformer layers
n_heads = 6        attention heads
vocab_size = 32000 known tokens
seq_len = 256      maximum sequence length
```

Keep `dim = 288` and `n_layers = 6` in your head; every shape below comes from
those two numbers.

The forward pass is the whole model run once:

```
token ids
   -> embedding lookup                          {dim, seqLen}
   -> layer 0: attention block, feed-forward block
   -> layer 1: attention block, feed-forward block
   ...
   -> layer 5: attention block, feed-forward block
   -> final RMSNorm
   -> classifier                                {vocab_size, seqLen}
   -> logits: a score for every vocabulary token
```

Chapter 1 gave you every box. What is missing is glue, and the glue is worth
understanding, so we go through each piece slowly.

## Concept 1: all six layers live in ONE tensor

This is the first thing that confused me. I expected the query projection weight
`wq` to be one matrix. It is not. The file stores the query projection for
**all six layers back to back**, and we load them as a single tensor:

```
wq is {dim, dim, n_layers} = {288, 288, 6}
```

Remember the axes from Chapter 1: axis 0 is the column axis, axis 1 the row
axis, axis 2 the layer axis. So `wq` is six 288x288 matrices stacked along
axis 2. `wq.At(i, o, 3)` reads one number out of layer 3's matrix.

To run layer 3 you need to cut layer 3's matrix out of that stack. That is
**Slice** (Task 1), and it works exactly like the Chapter 1 transpose trick: it
copies nothing. To keep elements `[start, start + length)` of one axis you only
need two edits:

1. Move the data pointer to the first kept element:
   `data + start * byteStrides[dim]`.
2. Shrink that axis's `elementCounts` entry to `length`.

The strides stay as they were. Work it out on `wq`: one layer is
`288 * 288 = 82944` floats, so `byteStrides[2] = 82944 * 4 = 331776` bytes.
`Slice(wq, 2, 3, 1)` moves the pointer forward `3 * 331776` bytes and sets the
layer count to 1. The result is a `{288, 288, 1}` view sitting in the middle of
the original memory. No copy, two integer edits.

Slice has a second job in this chapter: cutting attention heads out of a wide
tensor. Same operation, different axis.

## Concept 2: a token id is a row number

The model does not take words. It takes token ids, integers in
`[0, vocab_size)`. The first thing the forward pass does is turn each id into
its embedding vector, and the mechanism could not be simpler:

```
token_embedding_table is {dim, vocab_size} = {288, 32000}
```

Row `t` of this table IS the embedding of token id `t`. "Embedding lookup" is
literally: read row `t`. For a prompt of 5 tokens you gather 5 rows into a
`{288, 5}` tensor, one column-run of features per token. That is **GetRows**
(Task 2).

The tests use this prompt throughout:

```
"Once upon a time"  ->  { 1, 9038, 2501, 263, 931 }
```

The 1 in front is BOS (beginning of sequence), a special token that tells the
model "a fresh text starts here". Turning text into these numbers is the
tokenizer's job (`header/Tokenizer.h` on `master`, a short and commented read);
in this chapter they are simply input.

## Concept 3: Project, or how to multiply by ONE layer's matrix

Every projection in the model is the same move: take the layer's matrix out of
a stacked weight tensor, multiply the current activations by it. You will do
this eight times per layer, so it earns its own little op, **Project** (Task 3):

```
Project(x, W, layer) = MatMul(x, Transpose(Slice(W, 2, layer, 1)))
```

The Slice picks the layer. The Transpose needs a sentence: our `MatMul(A, B)`
from Chapter 1 wants `B` as `{N, K}` where `K` matches `A`'s axis 0. The sliced
weight matrix comes out of the file as `{inFeatures, outFeatures}`, which is the
wrong way around, so we flip it. Both Slice and Transpose are zero-copy views,
so Project costs exactly one MatMul and nothing else.

Shapes to check yourself against: `x` is `{288, seqLen}`, and
`Project(x, wq, l)` is `{288, seqLen}` again, while `Project(x, w1, l)` widens
to `{768, seqLen}`.

## Concept 4: heads are slices, not copies

`stories15M` has 6 attention heads. The model does not build 6 separate q
tensors for them. After the query projection, q is `{288, seqLen}`, and the
heads are simply **segments of the feature axis**:

```
features [  0,  48)  -> head 0
features [ 48,  96)  -> head 1
...
features [240, 288)  -> head 5
```

Each head is 48 features wide (`288 / 6 = 48`, the "head size"). To get head 2
you call `Slice(q, 0, 96, 48)` and you have a `{48, seqLen}` view, again without
copying anything.

**MultiHeadAttention** (Task 5) is then almost anticlimactic: for each head,
slice q, k, and v, run the Chapter 1 `Attention` on the three views, and write
the head's output into the matching segment of the result. Six small attentions
side by side instead of one big one. Why the model bothers: each head gets to
learn its own notion of relevance, one head might track "who is the subject",
another "what was the last noun". Cutting the features up forces them to
specialize.

## Concept 5: RoPE restarts at every head boundary

This one is THE classic bug, so slow down here. Chapter 1's `RoPE` rotates
feature pairs, and the rotation angle depends on the pair index: pair 0 rotates
fastest, later pairs slower and slower. That schedule is defined **per head**.

If you rotate the whole `{288, seqLen}` q tensor with one RoPE call, the pair
index runs from 0 to 143 and the frequencies keep decaying across head
boundaries. Head 0 comes out right; heads 1 through 5 all get wrong angles, and
nothing crashes, the model just generates garbage. I want you to make this
mistake mentally now so you never make it in code.

The fix is **ApplyRoPE** (Task 4): slice out each 48-wide head, RoPE it alone
(so its pair index starts back at 0), and write it into the output. RoPE is
applied to q and k only; v carries no position information.

## Concept 6: the residual stream

The last idea before you assemble. Look at the shape of one layer:

```
x = x + attention( RMSNorm(x) )      the attention block
x = x + feedforward( RMSNorm(x) )    the feed-forward block
```

`x` is called the residual stream. Notice what the pattern is: the block does
NOT transform x into something new. It computes a small correction and **adds
it onto x**, and x flows on. The RMSNorm only feeds the block's input; the
stream around the block is untouched.

The practical consequence for your code: in `AttentionBlock`, the final Add is
`Add(x, ...)` with the block's **input** x, not the normalized h. Wiring the
residual to the wrong tensor is the second classic bug of this chapter, and the
tests will catch it, because your layer-0 numbers will drift away from the
reference values.

The feed-forward block's inner part is SwiGLU, built from ops you already have:

```
SwiGLU(h) = W2 * ( SiLU(W1 * h) elementwise-multiplied with (W3 * h) )
```

`W1` (gate) and `W3` (up) widen 288 to 768, the elementwise multiply gates one
against the other, `W2` (down) brings it back to 288.

## Your tasks

Tasks 1-5 in `TensorOps.h`, tasks 6-8 in `Model.h`. Do them in order; later ones
call earlier ones.

1. `Slice` (zero-copy view of one axis's range)
2. `GetRows` (embedding lookup)
3. `Project` (one layer's matrix out of the stack, times x)
4. `ApplyRoPE` (per-head RoPE, schedule restarts each head)
5. `MultiHeadAttention` (slice heads, Chapter 1 Attention per head, concatenate)
6. `AttentionBlock` (norm, q/k/v, RoPE, attention, output projection, residual)
7. `FeedForwardBlock` (norm, SwiGLU, residual)
8. `Forward` (embed, 6 layers, final norm, classifier)

As in Chapter 1, the exact recipe is in the comment above each function.

## Build and run the tests

From this chapter's folder:

```bash
cmake -B build
cmake --build build
./build/ch2_tests              # Linux/macOS
# .\build\Debug\ch2_tests.exe  # Windows / MSVC
```

Tasks 6-8 compare your output against reference values computed from the same
weights, at three checkpoints: after layer 0's attention block, after all of
layer 0, and the final logits. If Task 8 fails while 6 and 7 pass, your layer
loop or final norm is wrong; if 6 already fails, the problem is inside the
attention block. That is the point of the checkpoints, they tell you WHERE it
broke.

When everything passes you will see:

```
[       OK ] ForwardPass.Forward
    the model read "Once upon a time" and predicted ","
```

That comma is not hardcoded anywhere. Your MatMul, your RMSNorm, your attention
just ran 15 million real weights forward, and the most likely next token came
out. If you keep feeding the prediction back in (that is Chapter 3, sampling),
the model continues:

```
Once upon a time, there was a little girl named L...
```

The reference implementations live on `master` in `header/TensorOps.h` and
`header/Model.h` if you get stuck, but try each one yourself first.
