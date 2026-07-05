# Chapter 3: Sampling

At the end of Chapter 2 your `Forward` read "Once upon a time" and produced
32000 logits, and the biggest one belonged to ",". This chapter closes the loop:
turn the 32000 scores into ONE chosen token, feed that token back into the
model, and repeat until a story comes out. By the end you will run:

```
./build/ch3_story "One day a dragon" 0.9
```

and watch your own code write, token by token.

## What is given, what you build

- `Tensor.h`, `TensorOps.h`: everything from Chapters 1 and 2, **completed**,
  plus two new empty functions for you.
- `Model.h`: loading and the full forward pass, **given** (the forward pass is
  the Chapter 2 answers). One empty function for you at the bottom.
- `Tokenizer.h`: **given**. It turns text into token ids and token ids back
  into text. It is short and commented, read it whenever you are curious;
  this chapter only calls `Encode` and `Decode`.
- `main.cpp`: the tests.
- `generate.cpp`: the story demo, given. It compiles into `ch3_story`.

Only three tasks this time, but the third one is the moment the whole project
has been building toward.

## Concept 1: logits are scores, not probabilities

`Forward` returns `{vocab_size, seqLen}`. The last column holds 32000 raw
scores for "which token comes next". Two things about them:

- They are not probabilities. They can be negative, and they do not sum to 1.
- Only their DIFFERENCES matter. Softmax turns them into probabilities, and
  softmax ignores any constant shift.

The simplest way to choose: take the biggest score. That is **ArgMax** (Task 1),
one loop over the last column. Choosing this way every time is called greedy
decoding, and it is completely deterministic: same prompt, same story, every
single run. Deterministic is wonderful for testing (the tests depend on it) and
dull for storytelling, which brings us to temperature.

## Concept 2: temperature

Sampling means: instead of always taking the best token, draw randomly with
probabilities from softmax. Temperature is one number that controls how bold
the draw is, and the mechanism is almost embarrassingly small: **divide every
logit by the temperature before the softmax**.

Work through it with three tokens whose logits are `ln 1`, `ln 2`, `ln 4`
(chosen so the numbers come out clean; these exact values are in the test):

```
temperature 1.0:  probabilities  1/7,  2/7,  4/7      the model's own opinion
temperature 0.5:  probabilities  1/21, 4/21, 16/21    differences amplified
temperature 2.0:  probabilities ~0.21, 0.29, 0.42     differences flattened
temperature -> 0: probabilities  0,    0,    1        greedy again
```

Dividing by a small temperature stretches the gaps between logits apart, so the
softmax concentrates on the winner. Dividing by a big one squeezes the gaps
toward zero, so the choice approaches a coin flip over the whole vocabulary.
Greedy is just the cold limit, which is why **Sample** (Task 2) falls back to
ArgMax when `temperature <= 0`.

## Concept 3: drawing a token with ONE random number

You have 32000 probabilities. How do you actually pick one token? The trick is
called inverse-CDF sampling and I find the dartboard picture the easiest way to
hold it:

Lay all the probabilities side by side on a line. Token 0 owns a segment of
length prob[0], token 1 the next segment of length prob[1], and so on. The
whole line has length 1. Now throw one dart: draw a single uniform random
number r in [0, 1) and walk from the left, adding segment lengths, until the
running total passes r. Whichever token's segment the dart landed in is your
sample. Bigger probability, longer segment, more likely to be hit. That is the
whole algorithm: one random number, one walk, done.

One refinement, and it is in the task recipe: you never need to normalize. Sum
the raw `exp((logit - max) / temperature)` values, draw r from `[0, sum)`
instead of `[0, 1)`, and walk the raw values. Same dartboard, just measured in
different units, and it saves a division over 32000 entries. (The
`subtract the max` part is the same numerical safety trick your Softmax from
Chapter 1 already uses.)

## Concept 4: the autoregressive loop

Now the loop that makes it a generator, **Generate** (Task 3):

```
tokens = prompt
repeat:
    logits = Forward(tokens)         score the whole sequence
    next   = Sample(last column)     choose one token
    if next == EOS: stop             the model says "the end"
    append next to tokens            the prediction becomes input
```

The appended token changes the input, so the next Forward produces different
logits, and so on. The model literally reads its own words to decide the next
one, which is why this is called autoregressive generation.

Three stopping rules, all in the recipe: the caller's `maxNewTokens` budget,
the EOS token (id 2, the model's own full stop; the tokenizer's special tokens
BOS = 1 and EOS = 2 were prepended/emitted during training exactly so this
works), and the model's `seq_len` limit of 256 tokens, beyond which it was
never trained to look.

Generate also takes an `onToken` callback and calls it the moment each token is
sampled. That is a tiny detail with a visible payoff: the demo prints each
token as it arrives, so you see the story being written instead of staring at a
frozen terminal.

And one thing I want you to feel rather than read: step one of the loop
recomputes the forward pass over the ENTIRE sequence every single iteration,
just to read the last column. Token 50 costs ten times what token 5 cost. Run
the demo, watch it visibly slow down as the story grows. Everything about that
waste is fixable, and fixing it is all of Chapter 4.

## Your tasks

Tasks 1-2 in `TensorOps.h`, task 3 in `Model.h`. The exact recipe is in the
comment above each function.

1. `ArgMax` (index of the largest logit at a position)
2. `Sample` (temperature sampling with the dartboard walk)
3. `Generate` (the autoregressive loop, with streaming callback)

Note the placeholder convention changed for 1 and 2: they return an integer,
not a Tensor, so the unimplemented stub returns `-1`.

## Build and run

From this chapter's folder:

```bash
cmake -B build
cmake --build build --config Release
./build/ch3_tests               # Linux/macOS  (add -DCMAKE_BUILD_TYPE=Release to the first cmake)
# .\build\Release\ch3_tests.exe # Windows / MSVC
```

Build **Release** for this chapter: the Generate test runs eight full forward
passes and the demo runs dozens, which is painfully slow in a Debug build.
(When you want to step through your own code with a debugger, build Debug and
keep max tokens small; the asserts inside Tensor.h only fire in Debug.)

The Generate test checks your greedy continuation against the exact token ids
the reference produces. When it passes you will see:

```
[       OK ] Sampling.Generate
    "Once upon a time" continues: ", there was a little girl named L"
```

Then the real reward:

```
./build/ch3_story                                 # "Once upon a time", temperature 0.9
./build/ch3_story "One day a dragon" 0.7 80       # your prompt, temperature, max tokens
```

Run it a few times at different temperatures. At 0 you get the same girl named
Lily every run; around 0.9 the stories fork; push it toward 2 and the prose
falls apart into word salad. That slider is one division in your Sample.

The reference implementations live on `master` in `header/TensorOps.h` and
`header/Model.h` if you get stuck, but try each one yourself first.
