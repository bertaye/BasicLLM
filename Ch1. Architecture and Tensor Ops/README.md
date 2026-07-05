# Chapter 1: Architecture and Tensor Ops

This is where we start. By the end of the chapter you will have written every math
operation a Llama 2 model needs, and checked each one against a test.

We do it in small, self-contained steps. Each operation is a single function with
its formula written above it and an empty body. You fill in the body, run the tests,
and watch it go from `TODO` to `PASS`.

## The model we are building

Here is the whole Llama 2 block. Read it top to bottom: a token goes in at the
bottom, and a prediction for the next token comes out at the top.

![Llama 2 architecture](architecture.png)

<!-- Replace architecture.png with the diagram:
     token -> embedding -> ( RMSNorm -> Q/K/V projections -> RoPE -> attention ->
     output projection -> add residual -> RMSNorm -> SwiGLU -> add residual ) x N
     -> final RMSNorm -> classifier -> logits -> next token -->

You do not need to understand the whole picture yet. The point is this: the diagram
is built out of a small number of repeated math operations. If you look at every box,
the complete list of things we must be able to compute is:

- **Matrix multiply** (every projection and the final classifier).
- **RMSNorm** (the normalization before attention and before the feed-forward part).
- **RoPE** (how a token's position is fed into the model).
- **Attention** (which itself is a matrix multiply, a scale, a mask, and a softmax).
- **Softmax** (inside attention, and again at the very end).
- **SiLU** and element-wise **multiply** (the SwiGLU feed-forward part).
- **Add** (the residual connections, the arrows that skip around a box).

That is the entire job of this chapter: implement those operations. Everything in
later chapters just arranges these pieces into the diagram above.

But before any of that, we need the thing all of these operate on: the tensor.

## The Tensor

A tensor sounds fancy. It is not. In this project a tensor is two things:

1. A **flat block of floating point numbers** in memory (one long line of floats).
2. A few numbers that **describe how to read that flat block as if it had a shape**.

That second part is the whole trick, so let us go slowly.

### Shape: `elementCounts`

`elementCounts` says how many elements exist along each axis. We use up to four axes.
The names we give them, from the first to the last:

- axis 0: the **column** axis
- axis 1: the **row** axis
- axis 2: the **layer** axis
- axis 3: the **batch** axis

For most of this chapter you only need the first two. A 2D tensor with 3 columns and
2 rows has `elementCounts = {3, 2, 1, 1}`.

Draw it as a small table. Call the values `a` through `f`:

```
          column0   column1   column2
 row0  [    a         b         c    ]
 row1  [    d         e         f    ]
```

### The one idea that is worth slowing down for: strides

Here is the question that confused me at first. The tensor above is a nice 2D table
in our heads, but memory is not 2D. Memory is one flat line. So how is that table
actually stored?

Answer: one row after another, with the columns sitting next to each other. The
column axis is the **innermost** axis, which means neighbouring columns are
neighbours in memory:

```
 memory:   a   b   c   d   e   f
 index:    0   1   2   3   4   5
```

Now, how do we find element `(column, row)` in that flat line? We need to know how
far to jump in memory for each step. That is exactly what `byteStrides` stores: the
number of **bytes** you move to take one step along each axis.

For the table above (each float is 4 bytes):

- `byteStrides[0]` (step one **column**) is `4` bytes. Columns are right next to each
  other, so one step is one float.
- `byteStrides[1]` (step one **row**) is `4 * 3 = 12` bytes. To go down one row you
  have to skip past a whole row of 3 columns.

And the address of any element is simply:

```
address of (column, row) = data + column * byteStrides[0] + row * byteStrides[1]
```

Check it on `f`, which is at `(column 2, row 1)`:

```
2 * 4  +  1 * 12  =  8 + 12  =  20 bytes  =  float index 5  =  f
```

That is precisely what `Tensor::At(column, row)` does. Nothing more.

### Why strides matter: views that cost nothing

Because addressing goes through the strides (and not some fixed rows-times-columns
formula), we can hand out a **different view of the same memory just by changing the
counts and strides**. No floats are copied.

The clearest example is **transpose**. Take the table above and swap the two axes:

- swap the counts: `{3, 2}` becomes `{2, 3}`
- swap the strides: `{4, 12}` becomes `{12, 4}`

The `data` pointer does not change at all. Now reading `result.At(i, j)` computes
`data + i * 12 + j * 4`, which lands on the **same float** the original called
`At(j, i)`. We transposed a matrix by editing two small numbers. This is why, later,
transposing a big weight matrix is free.

A freshly allocated tensor is **contiguous**: its strides are the tightly packed
values with no gaps. A transposed or sliced view usually is not, and that is fine,
because `At` works either way.

### Ownership (you can skim this)

When you copy a `Tensor`, the copy shares the same `data` and a small counter
(`refCount`) is bumped. The memory is freed only when the last owner is gone. You do
not need this for Chapter 1. Just know that copying a tensor is cheap and does not
duplicate the numbers.

Read through `Tensor.h` once with the above in mind. Then come back here.

## Your tasks

Open `TensorOps.h`. Each function has its exact formula written above it and an empty
body. Implement them in this order (later ones lean on the ideas of earlier ones):

1. `Transpose` (a view; swap the first two axes)
2. `Scale` (multiply every element by a number)
3. `Add` (element-wise)
4. `Mul` (element-wise)
5. `MatMul` (matrix multiply, the big one)
6. `Softmax` (per row)
7. `RMSNorm` (per row)
8. `SiLU` (per element)
9. `RoPE` (rotate feature pairs by position)
10. `MaskCausal` (block attention to future tokens)
11. `Attention` (the capstone: build it out of the ops above)

Every function's math is in the comment directly above it, so you do not need to
memorize anything. Write the loops that carry out that math, using `Tensor::At` to
read and write elements.

Two habits that make this painless:
- To produce output, allocate it first, for example `Tensor out(cols, rows);`, then
  fill each element with `out.At(c, r) = ...`.
- Read the shape off the input with `elementCounts[0]` (columns) and
  `elementCounts[1]` (rows).

## Build and run the tests

From this chapter's folder:

```bash
cmake -B build
cmake --build build
./build/ch1_tests              # Linux/macOS
# .\build\Debug\ch1_tests.exe  # Windows / MSVC
```

You will see one line per operation:

```
[TODO] Transpose  (not implemented yet)
[FAIL] Scale  (expected every element doubled)
[PASS] Add
...
```

- `TODO` means the function is still an empty stub.
- `FAIL` means it ran but the numbers are wrong. The message tells you what was
  expected.
- `PASS` means it matches.

Work one function at a time and rebuild. You are done with the chapter when the last
line reads `11 passed, 0 failed, 0 to do.`

The reference implementations live on `master` in `header/TensorOps.h` if you get
stuck, but try each one yourself first.
