# Chunking in a RAG System

**How a document becomes retrievable pieces — and why the standard tool behaves the way it does.**

# Part 1 — What a RAG system actually does

Start here, because chunking only makes sense as a step inside something larger.

## The problem

An LLM knows what was in its training data. It does not know your company's runbooks, your product docs, or the 400-page PDF you uploaded ten minutes ago. Ask it about them and it will either refuse or invent an answer.

Two ways to fix that:

1. **Fine-tune** the model on your documents — expensive, slow, has to be redone whenever a document changes, and still doesn't let the model cite anything.
2. **Retrieve** the relevant text at question time and paste it into the prompt.

Option 2 is **RAG** — Retrieval-Augmented Generation. The model stays untouched; you just make sure the right text is sitting in front of it when it answers.

## The simplest possible version of option 2

Option 2 needs no database, no vectors, no library. It is one f-string:

```python
prompt = f"""Answer the question using only the document below.

{entire_document}

Question: {question}"""

answer = llm(prompt)
```

That is a complete, working system. The model now "knows" your document, it can quote it, and it will say when the answer isn't there. **There is no chunking anywhere in it.**

Keep this baseline in mind for the whole note. If it worked at scale, nothing else here would need to exist.

## The one sentence that matters

> The LLM never touches your files, your database, or your disk. It only ever sees **text that was pasted into its prompt.**


**→ The next question:** so why doesn't everyone just paste the whole document? Part 2.

---

# Part 2 — Why we can't just paste the whole document

> **Where we are:** Part 1's one-liner pastes the entire document into the prompt. No vectors, no chunking. Let's find out exactly where it breaks — because *how* it breaks determines everything we build afterwards.

It breaks for two reasons. The first is a hard wall. The second is worse, because it has no error message.

## Reason 1 — The context window is a hard limit

Every LLM API call accepts a maximum number of tokens. Prompt, conversation history and the generated answer all share that one budget. That's the **context window**.

Rough conversion for English: **1 token ≈ 4 characters ≈ ¾ of a word.**

| what you're sending | rough tokens | fits in a 200k window? |
|---|---|---|
| one page | ~500 | yes |
| a 50-page PDF | ~25,000 | yes |
| a 500-page manual | ~250,000 | **no** |
| 2,000 internal documents | ~50,000,000 | **not remotely** |

Three consequences, in increasing order of importance:

1. **Exceed it and the call fails** — or worse, something in the chain silently truncates and the model answers from the first half of your document without telling you.
2. **You pay for the whole document on every single question.** Cost and latency scale with document size, per call. A one-word question about page 300 costs the same as a complex one, every time.
3. **It doesn't scale with your corpus.** This is the one that actually kills the baseline. The window is a per-call limit; your knowledge base is not one document, it's thousands. Bigger context windows raise the wall — they never remove it, and corpora grow faster than windows do.

So even the friendliest version of the baseline — one document that happens to fit — collapses the moment you have a *collection*.

## Reason 2 — Even when it fits, the answer gets worse

This is the reason people underestimate, because there's no error to catch. The call succeeds. The answer is fluent. It's just wrong more often.

**Lost in the middle.** Models attend most reliably to the start and end of a long prompt. A fact buried in the middle of a 200k-token context is measurably less likely to be used than the same fact in a 2k-token context. Filling the window doesn't mean the model reads it evenly.

**Blending — the hallucination mode specific to broad context.** With an entire manual in the prompt, many passages look *partly* relevant to the question. The model has no signal telling it which ones are actually on-topic, so it can take a sentence from the billing section and a sentence from the auth section and merge them into one confident, fluent, wrong answer.

This is worth naming precisely, because it isn't the hallucination people usually picture:

> It isn't invented from nothing. It is assembled from **real sentences in your document that should never have been joined.** That makes it harder to spot — every fragment checks out; only the combination is false.

The mechanism is simple: relevance is a *ratio*, not a quantity.

```
3 relevant sentences  in  50 pages of context   ->  signal buried in noise
3 relevant sentences  in  1 paragraph           ->  nothing to confuse
```

**No grounding to point at.** If the context was "the whole document", the model can't tell you which part the answer came from — and neither can you. Citation and verification both disappear.

> **More context is not better context.** Narrow, relevant context beats broad context — *even when broad context fits.*

## What both reasons demand

Reason 1 says the whole document *can't* go in. Reason 2 says that even when it can, it *shouldn't*. Both point at the same replacement:

> Send only the **part of the document this specific question needs**.

That one sentence creates two new problems that did not exist in Part 1's baseline. Everything in the rest of this note descends from them:

| new problem | what it's called |
|---|---|
| the document must first be divided into parts | **chunking** |
| we must pick the right parts per question, without reading them all | **retrieval** |

## Why retrieval needs vectors

We can't ask the LLM which parts are relevant — reading everything to decide what to read is the original problem again. We need a way to compare a question against every part **cheaply**.

So during ingestion we precompute a numeric summary of each part: an **embedding**, a list of numbers positioning that text in a space where similar meanings sit close together. At question time we embed the question the same way and take the nearest ones. That's arithmetic, not inference — fast and cheap over millions of parts.

### And this is why the vector must be per-part, not per-document

Someone will reasonably suggest: store one vector per document, find the right document, paste it. Reason 1 already rules that out — but there's an independent problem worth seeing, because it constrains *how big* our parts may be.

An embedding is a **fixed-size** vector — say 1,536 numbers — regardless of input length. Ten words in, 1,536 numbers out. Ten thousand words in, still 1,536 numbers out.

Embed a whole manual covering billing, authentication, deployment and troubleshooting and you get one point that is roughly the **average** of all four topics. It isn't near "billing". It isn't near "authentication". It's near the blurry centre of a document about nothing in particular.

```
one big unit:      [billing + auth + deploy + errors]  ->  * one blurry point

small units:       [billing]   -> *
                   [auth]      ->      *      <- your question lands right here
                   [deploy]    ->            *
                   [errors]    ->   *
```

Ask *"how do I rotate an API key?"* and the big vector matches poorly — not because the answer isn't in the document, but because the vector no longer represents the answer. It represents the whole book.

> **One vector can only mean one thing.** So a part must be about one thing.

Note what just happened: **the unit you embed and the unit you paste are the same unit.** Shrinking it satisfies Reason 1, Reason 2, and vector precision simultaneously. That's why one decision — how big a part is and where it ends — controls the entire system.

## The pipeline that results

Now the standard RAG diagram is something we derived rather than assumed:

```
  === INGESTION (offline, once per document) ===

     your documents
           |
           v
     cut into parts          <-- CHUNKING lives here
           |
           v
     embed each part  ->  a vector
           |
           v
     store {vector, text} in a vector database


  === QUERY (online, once per question) ===

     user question
           |
           v
     embed the question -> a vector
           |
           v
     find the nearest stored vectors      (cheap: arithmetic, not inference)
           |
           v
     take those parts' ORIGINAL TEXT
           |
           v
     paste ONLY that text into Part 1's prompt slot
           |
           v
     LLM writes the answer
```

Compare it with Part 1's f-string: the prompt is identical. The only thing that changed is what fills the slot.

## The constraint this leaves us with

> Retrieval can only ever return **a whole part**. It cannot return half a part, and it cannot assemble one for you.

So the quality ceiling of the entire system is fixed at ingestion time, by where the parts begin and end. If the answer to a question straddles two parts, no re-ranker, no better embedding model and no smarter prompt will put it back together.

**→ The next question:** parts have edges. Where exactly do the edges go? Nobody had to answer that in Part 1. Answering it is the whole of Part 3 onward.

---

# Part 3 — Chunking: the question we are now forced to answer

> **Where we are:** we've established we must store parts. The moment we say "parts", we have created a decision that didn't exist before — where each part begins and ends.

**Chunking is that decision.** Nothing more:

> Chunking = choosing where to cut a document into the units you will embed and retrieve.

Each resulting unit is a **chunk**.

## Why the decision is hard

From Parts 1 and 2, a chunk has to satisfy four things at once:

| requirement | comes from | pushes chunks... |
|---|---|---|
| **1. under the size limit** | Reason 1: the context window, plus the embedding model's own input cap | smaller |
| **2. about one idea** | Reason 2 (blending) and "one vector can only mean one thing" | smaller |
| **3. self-contained** — readable alone, no dangling references | Part 1: the LLM sees only the text pasted into the slot | **bigger** |
| **4. doesn't start or end mid-thought** | same, plus "retrieval returns whole parts only" | **bigger** |

Requirements 1–2 and 3–4 pull in opposite directions. There is no setting that maximises all four. Every chunking strategy is a position on that trade-off, and the rest of this note is about finding a good one.

## Chunking happens once, and everything inherits it

Chunking is the first transformation in the pipeline. A badly cut chunk produces a bad vector; a bad vector is retrieved for the wrong questions, or never retrieved at all; and the LLM then answers from text that doesn't contain the answer.

> You cannot repair a bad chunk later in the pipeline. There is no later step that knows what was cut off.

**→ The next question:** what is the simplest possible way to cut? Try it and see exactly which of the four requirements it breaks.

---

# Part 4 — Naive attempt #1: cut every N characters

> **Where we are:** we need to cut a document into chunks. Requirement 1 (size limit) is the only one that's easy to guarantee, so let's guarantee that one and see what breaks.

From here on, all examples use a size limit of **100 characters** — small enough that the arithmetic stays readable. Everything scales identically to 1,500.

```python
chunks = [text[i:i+100] for i in range(0, len(text), 100)]
```

Count to 100, cut. Count to 100, cut. Fast, trivially correct on size, completely blind to meaning.

## Run it on real text

```python
text = ("Django models map to tables.\n\n"
        "Each field is a column.\n\n"
        "Migrations are versioned Python files that describe schema changes and "
        "apply them to the database in a deterministic order every single time.")
```

Real output:

```
[  0] 'Django models map to tables.\n\nEach field is a column.\n\nMigrations are versioned Python files that de'
[100] 'scribe schema changes and apply them to the database in a deterministic order every single time.'
```

Look at the seam: `...that de` / `scribe schema...`

The word **describe** has been cut in half. `de` is going into one vector; `scribe` into another.

## Which requirements broke

Check it against the list from Part 3:

| requirement | result |
|---|---|
| 1. under the size limit | ✅ guaranteed, always |
| 2. about one idea | ❌ chunk 0 holds models, fields *and* half of migrations |
| 3. self-contained | ❌ chunk 1 starts with `scribe schema changes and apply them...` — **them** = what? The subject `Migrations` is in the other chunk |
| 4. no mid-thought edges | ❌ mid-*word*, which is worse than mid-thought |

Three out of four broken. The broken word is what everyone notices first, but requirement 3 is what actually costs you retrieval quality: a chunk whose subject lives in a different chunk can never answer a question on its own.

## The diagnosis

The cut position was chosen by a counter, not by the text. Character 100 has no relationship to where a word, sentence or idea ends.

**→ The next question:** so let the *text* choose the cut positions instead of a counter. Part 5.

---

# Part 5 — Naive attempt #2: cut only at natural boundaries

> **Where we are:** Part 4 failed because cut positions were chosen blindly. The fix is obvious — only cut where the text already has a break.

## Try 5a: cut at every space

```python
text.split(" ")
```

Real output, first 8 items:

```
['Django', 'models', 'map', 'to', 'tables.\n\nEach', 'field', 'is', 'a']
```

Nothing is broken now. And nothing is useful. `'to'` is a chunk. `'a'` is a chunk. You are about to embed the word "a" and store it in a vector database.

| requirement | result |
|---|---|
| 1. under the size limit | ✅ (absurdly so) |
| 2. one idea | ❌ a word is not an idea |
| 3. self-contained | ❌ |
| 4. no mid-thought edges | ✅ |

We fixed 4 and destroyed 2 and 3. This is Part 4's failure **mirror-imaged**:

- Part 4: right **size**, wrong **boundaries**
- Part 5a: right **boundaries**, useless **size**

## Try 5b: cut at every paragraph

Paragraphs are real units of meaning, so this looks much better:

```python
text.split("\n\n")
```

Real output — the piece lengths:

```
[28, 23, 141]        limit is 100
```

And there it is. The third paragraph is **141 characters against a 100 limit**. Splitting at paragraphs gives you meaningful units but hands you **zero control over size** — the one thing Part 4 got right. Real documents have 30-character paragraphs and 3,000-character paragraphs. The small ones waste a retrieval slot; the large ones don't fit in the embedding model at all.

| requirement | 5a: spaces | 5b: paragraphs |
|---|---|---|
| 1. under size limit | ✅ | ❌ **no guarantee at all** |
| 2. one idea | ❌ | ✅ |
| 3. self-contained | ❌ | ✅ |
| 4. no mid-thought edges | ✅ | ✅ |

## The tension, stated plainly

```
      cut only at natural boundaries            fill every chunk to the limit
                  |                                          |
           pushes chunks                               pushes chunks
              SMALLER      <-------------------->        BIGGER
         (5a: one word per chunk)              (4: exactly 100, mid-word)
```

Neither goal can be abandoned. Requirement 1 needs the right-hand pull; requirements 2–4 need the left-hand pull.

**→ The next question:** hold on. Even if we solve that tension perfectly, there's a problem *neither* attempt addresses. Part 6.

---

# Part 6 — Naive attempt #3: overlap

> **Where we are:** Parts 4 and 5 fought over *where* to cut. This part is about a problem that exists **no matter how good your cut is**.

## The problem a perfect cut still has

Take the third paragraph and cut it at a completely clean word boundary:

```
'Migrations are versioned Python files that describe schema changes and'
' apply them to the database in a deterministic order every single time.'
```

Nothing is broken. Both halves are grammatical. And chunk 2 is still unusable on its own:

> *apply **them** to the database...*

**Them** = migrations, and that word is in the other chunk. The *idea* — "migrations are applied to the database in a deterministic order" — sits **across the seam**, so it lives complete in neither chunk. A question about applying migrations to a database matches neither one well.

> Every cut, however clean, orphans whatever idea was spanning it. Better boundaries reduce this. Nothing eliminates it.

## The fix: let chunks share their edges

Don't advance the window by a full chunk each time. Advance by less, so each chunk re-includes the tail of the one before it:

```python
step = chunk_size - overlap          # 100 - 20 = 80
chunks = [text[i:i+100] for i in range(0, len(text), step)]
```

Real output at size 100, overlap 20:

```
chunk 0 len=100: 'Django models map to tables.\n\nEach field is a column.\n\nMigrations are versioned Python files that de'
chunk 1 len=100: 'Python files that describe schema changes and apply them to the database in a deterministic order ev'
chunk 2 len= 36: 'terministic order every single time.'

measured overlaps: [20, 20]
```

Chunk 1 now begins with `Python files that...`, which chunk 0 also ended with. The bridge exists. An idea spanning the seam has a better chance of living complete inside at least one chunk.

## Two properties of naive overlap — remember both

**1. It is exact and guaranteed.** Measured overlap at every seam: `[20, 20]`. Not "about 20". Exactly 20, every time, because the stride is arithmetic. **Hold on to this** — Part 20 is about the fact that the real tool does *not* give you this, and most people assume it does.

**2. It costs storage and duplication.** At 100/20 you store ~25% more text than the document contains, embed ~25% more, and the same sentence can now be returned twice in one top-k result set. Overlap is not free, which is why you don't just set it to half your chunk size.

## What overlap did *not* fix

Look at the output again: `...that de` / `terministic...`. Still cutting mid-word, because overlap is orthogonal to boundary quality. It changes *how much chunks share*, not *where the cuts are*.

| requirement | with naive overlap |
|---|---|
| 1. under size limit | ✅ |
| 2. one idea | ❌ still blind cutting |
| 3. self-contained | 🔶 improved by the bridge, but still mid-word |
| 4. no mid-thought edges | ❌ unchanged |

**→ The next question:** we now have three techniques, each fixing something and breaking something else. Can they be combined? Part 7.

---

# Part 7 — Scorecard: why no single naive method works

> **Where we are:** three attempts, three partial solutions. Line them up.

| | 1. size limit | 2. one idea | 3. self-contained | 4. clean edges |
|---|---|---|---|---|
| **#1** fixed slicing | ✅ | ❌ | ❌ | ❌ |
| **#2a** split on spaces | ✅ | ❌ | ❌ | ✅ |
| **#2b** split on paragraphs | ❌ | ✅ | ✅ | ✅ |
| **#3** slicing + overlap | ✅ | ❌ | 🔶 | ❌ |

Read the columns, not the rows. Every column has a ✅ somewhere — so every individual requirement is solvable. No single row has four.

And notice **which methods hold which ✅**:

- Column 1 (size) is held by the methods that **count**.
- Columns 2 and 4 (meaning, clean edges) are held by the method that **respects the text's own structure**.

These are two different kinds of work, and each method only does one of them. That's the actual insight:

> Nobody has failed at chunking. We've been trying to do **two different jobs with one mechanism**.

**→ The next question:** what if we stop trying, and do the two jobs separately? Part 8 — and this is where the real tool finally appears.

---

# Part 8 — The idea: separate "where may I cut?" from "how much do I keep?"

> **Where we are:** we've concluded that boundary-finding and size-filling are two different jobs. Now do them as two different steps.

## The two jobs

| job | question it answers | what it produces |
|---|---|---|
| **SPLIT** | Where am I *allowed* to cut? | many small units, cut at natural boundaries |
| **PACK** | How many of those go into one chunk? | fewer, larger chunks, each under the limit |

SPLIT cuts the text down to the finest **legal** units. It doesn't care about size — that's not its job. PACK then glues those units back together until just under the limit. It doesn't care about meaning — SPLIT already guaranteed every boundary is legal.

- SPLIT alone → every chunk is one word (that's attempt #2a)
- PACK alone → no legal cut points, so it cuts at exactly char 100 (that's attempt #1)
- **SPLIT then PACK** → as large as possible, ending at a natural boundary

That round trip is the entire product. Both jobs get done, neither compromises the other, and every requirement in Part 7's scorecard gets its ✅.

The class that implements this is LangChain's:

```python
from langchain_text_splitters import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=20)
chunks = splitter.split_text(text)
```

The name decodes as: it splits **text**, it works down to individual **characters** if it has to, and it does so **recursively** — it retries with a finer boundary type on anything still too big. All three of those will be concrete by Part 17.

## Terminology — read this box before continuing

I use short names below for readability. Some are LangChain's real names; some are mine. Here is exactly which is which, so nothing is ambiguous:

| name I use | LangChain's actual name | what it is |
|---|---|---|
| **SPLIT** | `_split_text()` | the method that cuts text at separators |
| **PACK** / *the packer* | `_merge_splits()` | the method that glues units together up to the limit |
| **piece** | an element of `splits` / `good_splits` | one output of SPLIT: a paragraph, line, or word. **Never returned to you.** |
| **chunk** | an element of `final_chunks` / `docs` | one output of PACK. **This is what gets embedded.** |
| **pending** | `good_splits` | inside SPLIT: pieces collected so far, waiting to be packed |
| **buffer** | `current_doc` | inside PACK: pieces in the chunk currently being built |
| **total** | `total` | real name — combined length of the buffer |
| **the drain** / *pop loop* | the `while` loop inside `_merge_splits` | the loop that discards pieces after a chunk closes |
| **the ladder** | `self._separators` | the ordered list of separators |

**"piece" vs "chunk" is the distinction to internalise.** LangChain uses "split" and "chunk" loosely for both, which is exactly why people misread the algorithm.


## One fact to fix now

> `chunk_size` and `chunk_overlap` are used **only inside PACK**. SPLIT has never heard of `chunk_overlap`, and uses `chunk_size` for exactly one yes/no test.

Most confusing behaviour in this class comes from forgetting that.

**→ The next question:** how do the two jobs actually fit together in one run? Part 9 gives you the whole machine before we take it apart.

---

# Part 9 — The whole algorithm, start to finish

> **Where we are:** we know the two jobs. This part shows the complete mechanism in one place. **Parts 10–19 are all zoom-ins on this part** — so if something here is only half-clear, that's expected and fine. Come back and re-read this part after Part 19; it will read completely differently.

## The story: you hand over a text

You call:

```python
splitter.split_text(long_text)
```

Here is what happens, in order, in plain English.

**1.** `split_text` does nothing itself. It immediately calls `SPLIT(long_text, ["\n\n", "\n", " ", ""])` — the text plus the full separator ladder.

**2.** SPLIT looks down the ladder and picks **the first separator that actually occurs in this text**. Paragraph breaks present? Use `"\n\n"`, and stop looking. It also remembers **what's left below** on the ladder — `["\n", " ", ""]` — for later.

**3.** It cuts the whole text on that one separator. Now it has a list of **pieces**, in document order. If it picked `"\n\n"`, the pieces are paragraphs.

**4.** It walks those pieces left to right and asks each one a single yes/no question: *are you, on your own, smaller than `chunk_size`?*

**5a. Yes (the common case).** The piece is a usable ingredient. It goes onto the **pending** list. Move to the next piece. Nothing else happens.

**5b. No — this piece is too big to fit in a chunk even alone.** Two things happen, in this order:
   - **Hand off what's pending.** Everything collected so far is sent to PACK and comes back as finished chunks. Those pieces come earlier in the document, so they must be emitted first. The pending list resets to empty.
   - **Then deal with the big piece by itself:** call SPLIT again on *just that piece*, with the **remaining** ladder (`["\n", " ", ""]`). That's the recursion — the same procedure one rung finer. (If the ladder is exhausted, the piece is emitted as-is, oversized.)

**6.** When the walk ends, whatever is still pending goes to PACK.

**7.** PACK is the size-filling half. It keeps a **buffer** and a running **total**. For each piece: if adding it would exceed `chunk_size`, it **closes** the current chunk *without that piece*, then **drains** the buffer from the front until what's left is small enough — and whatever survives the drain is still sitting there when the next chunk starts building. **That leftover is the overlap.** Then it adds the piece and continues.

**8.** At the end, PACK closes one last chunk with whatever remains. The chunks come back up through any recursions, in document order, and that list is your return value.

## The pseudocode

Both halves in one place. Compare against `character.py` and `base.py` — this is faithful, with two tiny simplifications flagged in comments.

```python
def split_text(text):
    return SPLIT(text, ["\n\n", "\n", " ", ""])


def SPLIT(text, separators):                     # LangChain: _split_text
    chunks_out = []
    pending    = []                              # LangChain: good_splits

    # ---- STEP A: pick exactly ONE separator ------------------------
    sep, remaining = first separator in `separators` that occurs in `text`
                     # if none occurs, sep = last one, remaining = []
    pieces = text.split(sep)                     # separator is kept, glued to
                                                 # the FRONT of the next piece

    # ---- walk the pieces in document order -------------------------
    for piece in pieces:

        # ---- STEP B: the size test ("Check 1") ---------------------
        if len(piece) < chunk_size:
            pending.append(piece)                        # ROAD A
        else:                                            # ROAD B
            if pending:
                chunks_out += PACK(pending)              #   1. hand off
                pending = []
            if not remaining:
                chunks_out.append(piece)                 #   2a. ladder exhausted
            else:
                chunks_out += SPLIT(piece, remaining)    #   2b. RECURSE

    if pending:
        chunks_out += PACK(pending)              # final hand-off
    return chunks_out


def PACK(pieces):                                # LangChain: _merge_splits
    chunks = []
    buffer = []                                  # LangChain: current_doc
    total  = 0

    for piece in pieces:

        # ---- STEP C/D: does it fit? -------------------------------
        if total + len(piece) > chunk_size and buffer:

            chunks.append(join(buffer))          # STEP D: CLOSE
                                                 # note: WITHOUT `piece`

            # ---- STEP E: the drain --------------------------------
            while total > chunk_overlap or (
                  total + len(piece) > chunk_size and total > 0):
                total -= len(buffer[0])
                buffer = buffer[1:]              # pop from the FRONT

            # whatever is left in `buffer` now becomes the OVERLAP

        buffer.append(piece)
        total += len(piece)

    chunks.append(join(buffer))                  # final close
    return chunks

# simplifications: `join` strips whitespace and drops empty results;
# the real fit-check and drain also add len(separator), which is 0 here
# because keep_separator glues separators onto the pieces themselves.
```

## The same thing as a diagram

```
                 split_text(text)
                        |
                        v
  +===========================================================+
  |  SPLIT(text, separators)                                  |
  |                                                           |
  |  STEP A   pick the FIRST separator present in the text    |
  |           cut the text -> PIECES (in document order)      |
  |                                                           |
  |  for each piece:                                          |
  |                                                           |
  |  STEP B      len(piece) < chunk_size ?                    |
  |                  /                  \                     |
  |               YES                    NO                   |
  |                |                      |                   |
  |            ROAD A                  ROAD B                 |
  |         pending.append()       1. PACK(pending) --------. |
  |                                2. SPLIT(piece,         | |
  |                                        remaining) ---. | |
  |                                        (RECURSION)   | | |
  |                                                      | | |
  |  end of pieces: PACK(pending) ---------------------. | | |
  +====================================================|=|=|==+
                                                       v v v
  +===========================================================+
  |  PACK(pieces)          buffer = []   total = 0            |
  |                                                           |
  |  for each piece:                                          |
  |     STEP C   fits?  -> buffer.append(piece)               |
  |     STEP D   no?    -> CLOSE chunk  (piece NOT included)  |
  |     STEP E          -> DRAIN buffer from the FRONT        |
  |                        survivors == the OVERLAP           |
  |                        then buffer.append(piece)          |
  |                                                           |
  |  end: CLOSE final chunk                                   |
  +===========================================================+
                        |
                        v
                 list of chunks
```

## The five open questions, and where each is answered

Every remaining part of this note answers one of these. Use this as your map:

| step | the question | answered in |
|---|---|---|
| **A** | Why only *one* separator? Why not all of them? | **Part 10** |
| **B** | What exactly does the size test compare — and what's the fork? | **Part 11** |
| **C** | What goes into the buffer, and when? | **Part 12** |
| **D** | When does a chunk close, and is the causing piece inside it? | **Part 13** |
| **E** | When does the drain stop? Where does overlap come from? | **Parts 14–15** |
| **B / ROAD B** | What really happens to an oversized piece? | **Part 17** |
| — | Why does overlap so often turn out to be zero? | **Parts 15, 17, 18, 20** |

**→ Next:** Step A. Part 10.

---

# Part 10 — Step A: choosing one separator (the ladder)

> **Where we are:** step A of Part 9. We have raw text and need pieces.

## The ladder

```python
separators = ["\n\n", "\n", " ", ""]
```

| separator | what you get | meaning preserved |
|---|---|---|
| `"\n\n"` | paragraphs | most |
| `"\n"` | lines | |
| `" "` | **words** | |
| `""` | characters | none |

The list is ordered **best boundary first** — this is Part 5's insight encoded as data. Paragraph breaks preserve the most meaning; character breaks preserve none.

## The rule most people get wrong

> SPLIT scans the ladder top to bottom, picks **the first separator that occurs in this text**, cuts on **that one only**, and stops. The remaining rungs are saved for later, but are **not used on this text**.

Not "try all of them". Not "split on paragraphs and then also on words".

If the text contains `"\n\n"`, you get paragraphs and the ladder stops there. `" "` never runs on that text — even though the text obviously contains spaces.

**This single rule is responsible for the biggest production surprise in this note (Part 20).** If it feels arbitrary now, that's fine; just remember it's "pick one", not "try all".

Two details:

- **`" "` gives words, not characters.** Only `""` gives characters, and it's a genuine last resort — it fires only when one unbroken token is longer than `chunk_size` (a base64 blob, a long URL, a minified JS line). In normal prose it never runs.
- **`keep_separator=True`** (the default) glues the separator onto the **front** of the following piece, so no text is lost. `"A\n\nB"` becomes `["A", "\n\nB"]`. Every piece length below except the first therefore includes **2 extra characters** for its leading `\n\n`. Remember this — several numbers later look off by two until you account for it.

## What comes out

For our running text, SPLIT picks `"\n\n"` and produces three pieces:

```
[28, 25, 143]        (with the leading "\n\n" counted on pieces 2 and 3)
```

**→ Next:** those three pieces now face the size test. Notice the 143 already — the limit is 100. Part 11.

---

# Part 11 — Step B: the size test, and the fork in the road

> **Where we are:** SPLIT has a list of pieces in document order. It now walks them left to right — nothing sorted, nothing rearranged — asking each one **one** question.

## The question

> **Is this piece, on its own, smaller than `chunk_size`?**

```python
if len(piece) < chunk_size:   # 100
```

Call this **Check 1** (there is a second, different size check inside PACK — Part 13 — and confusing the two is the single most common mistake with this class).

## The fork

```
                      piece
                        |
             len(piece) < chunk_size ?
                /                    \
             YES                      NO
              |                        |
         --- ROAD A ---           --- ROAD B ---
      A usable ingredient.     Cannot fit in a chunk
      Add to `pending`;        even alone. Never mixes
      it will be packed        with its neighbours.
      with its neighbours.     Handled by itself,
                               immediately.
```

**Road A is the common case** — almost every piece in a normal document takes it.

**Road B is where the surprising behaviour lives.** It is also the source of the "recursive" in the class name. Handling it now would mean explaining recursion before you've seen the packer work even once.

> **We are deliberately parking Road B until Part 17.** It is on the map in Part 9 and it is not forgotten. Every scenario in Parts 12–16 assumes **every piece is under 100**, so every piece takes Road A.

## What Road A produces

For our three pieces `[28, 25, 143]`, the first two take Road A and land in `pending`. (The 143 takes Road B — parked.)

**→ Next:** follow the pending pieces into PACK. Part 12.

---

# Part 12 — Step C: the packer fills the buffer

> **Where we are:** step C. SPLIT has handed PACK a list of pieces that are each known to be under the limit. PACK's only job now is size.

Settings for every scenario below: **`chunk_size=100`, `chunk_overlap=20`.**

## The mental picture

The buffer is a box you're filling. Drop pieces in one at a time. When the next piece would overflow the box, tape it shut and start a new one.

```
buffer:  [ piece ][ piece ][ piece ] ...
total:   sum of their lengths
```

## Scenario 1 — everything fits

Pieces: `30, 25, 20`. All under 100 individually, and their sum is under 100 too.

```
+30 -> total 30   (fits)
+25 -> total 55   (fits)
+20 -> total 75   (fits)
end of input -> CLOSE final chunk (internal total 75)
```

Real result:

```
chunk 0: len=75
```

**One chunk**, three paragraphs merged into it.

This is PACK doing its actual job — this merge is exactly what saves us from attempt #2a's one-word chunks. Three separate legal pieces became one chunk of usable size, and every edge of that chunk is still a real paragraph boundary. **That is Part 7's scorecard finally getting four ✅s.**

Note what did *not* happen:

- no overflow — the `if` never fired
- no drain — the drain lives inside the overflow branch
- no second chunk, so overlap isn't even a question yet

> **Answer to "what goes into the buffer?"** — every Road A piece, one at a time, in document order, for as long as they keep fitting.

**→ Next:** what happens on the piece that *doesn't* fit? Part 13.

---

# Part 13 — Step D: overflow closes a chunk

> **Where we are:** Scenario 1 never overflowed. Add one more piece so it does.

## Scenario 2 (first half)

Pieces: `30, 25, 20, 40`.

```
+30 -> total 30   (fits)
+25 -> total 55   (fits)
+20 -> total 75   (fits)
+40 -> 75+40=115 > 100   CLOSE chunk (internal total 75)
```

Read that last line slowly — there's a trap in it.

> **The 40 is not in chunk 0.**

The chunk closed with what was **already** in the buffer — `[30, 25, 20]` = 75. The 40 is the piece that *caused* the close; it goes into the **next** chunk.

## Why the causing piece is excluded

Because including it is the exact thing we're preventing. `75 + 40 = 115 > 100`. The check exists to keep it out. Simple, but it explains something you'll see constantly:

> **`chunk_size` is a ceiling, not a target.** Chunk 0 is 75, not 100. PACK stopped there because the next available piece happened to be 40 characters. It cannot cut that 40 in half to top up to exactly 100 — it only moves whole pieces. Whole-piece movement is *precisely* what guarantees clean edges, so this "wasted" space is the price of requirement 4.

## Check 1 vs Check 2

Now both size checks exist, so pin them down:

| | what it compares | where it lives | what failing it means |
|---|---|---|---|
| **Check 1** | the piece **alone** vs `chunk_size` | SPLIT, Part 11 | Road B: this piece needs re-splitting |
| **Check 2** | buffer **+ piece** vs `chunk_size` | PACK, here | close the chunk; the piece starts the next one |

Same constant, completely different consequences. Part 17 turns on the difference.

## Two separate events

This distinction carries the rest of the note:

```
1. the chunk CLOSED
2. and THEN the drain ran
```

They are two steps, not one. Parts 17 and 18 are about closes where **step 2 never happens at all** — and that is exactly when overlap disappears.

---

# Part 14 — Step E: the drain, and where overlap comes from

> **Where we are:** in Part 13 a chunk closed by overflow. The buffer still holds `[30, 25, 20]` and the 40 is waiting outside. This part is about what happens to that buffer — and it is the part where people lose the thread, so we go one event at a time.

## First, the thing that confuses everyone

Before any rule, settle this question, because everything else depends on it:

> **When the chunk closes, is the chunk a photograph of the buffer, or a window onto it?**

It is a **photograph**. Look at the pseudocode from Part 9:

```python
chunks.append(join(buffer))     # <- builds a STRING, right now, from the buffer's
                                #    current contents. That string is finished.
while ...:
    buffer = buffer[1:]         # <- mutates the LIVE buffer afterwards.
                                #    The string above cannot change.
```

`join(buffer)` produces a finished string and puts it in the output list. Everything the drain does afterwards happens to a **different object** — the live buffer. The chunk is already gone; it cannot be edited, shortened or un-emitted.

So, to answer the question directly:

- **Chunk 0 is `[30][25][20]`** — all three pieces, 75 characters. It was photographed *before* the drain ran.
- **The buffer after the drain is `[20]`** — one piece, 20 characters.
- **The 20 is therefore in both places at once.** It's baked into the finished chunk 0 *and* still sitting live in the buffer where chunk 1 will be built on top of it.

**That double presence is the overlap.** Nothing was copied to make it happen; it's the same piece, once frozen into an output string and once still in the working list.

## The full timeline, one row per event

`chunk_size=100`, `chunk_overlap=20`, pieces `30, 25, 20, 40`.

| # | event | output list (frozen) | buffer (live) | total |
|---|---|---|---|---|
| 1 | start | — | `[]` | 0 |
| 2 | +30 fits | — | `[30]` | 30 |
| 3 | +25 fits | — | `[30, 25]` | 55 |
| 4 | +20 fits | — | `[30, 25, 20]` | 75 |
| 5 | 40 arrives: `75+40=115 > 100` | — | `[30, 25, 20]` | 75 |
| 6 | **CLOSE** → photograph the buffer | **chunk 0 = 30+25+20 = 75** | `[30, 25, 20]` | 75 |
| 7 | drain: pop 30 | chunk 0 *(frozen, unaffected)* | `[25, 20]` | 45 |
| 8 | drain: pop 25 | chunk 0 *(frozen)* | `[20]` | 20 |
| 9 | drain: stop — `20 > 20` is false | chunk 0 | `[20]` | 20 |
| 10 | now add the 40 | chunk 0 | `[20, 40]` | 60 |
| 11 | input ends → **CLOSE** | chunk 0, **chunk 1 = 20+40 = 60** | — | — |

Read rows 6–9 as a block. The chunk is emitted at row 6 and never touched again. Rows 7–9 are housekeeping for the *next* chunk.

The 20 is in the row-6 photograph **and** survives to row 10. It is in both chunks. Overlap.

## The same thing in real text

Those four piece lengths are exactly these four paragraphs:

```python
text = ("Models map to database tables.\n\n"        # piece 1: 30
        "Each field is a column.\n\n"               # piece 2: 25 (with leading \n\n)
        "Indexes are quick.\n\n"                    # piece 3: 20 (with leading \n\n)
        "Migrations version each schema change.")   # piece 4: 40 (with leading \n\n)
```

Real output:

```
chunk 0  len=75
   'Models map to database tables.\n\nEach field is a column.\n\nIndexes are quick.'

chunk 1  len=58
   'Indexes are quick.\n\nMigrations version each schema change.'

overlap = 18 chars: 'Indexes are quick.'
```

There it is, visible: **`Indexes are quick.` is the last sentence of chunk 0 and the first sentence of chunk 1.** One paragraph, stored twice, because it was still in the buffer when chunk 1 started building.

And that's exactly the fix Part 6 asked for. A question about indexes now has a chunk where that sentence sits next to the migrations sentence, and another where it sits next to the fields sentence. Whichever neighbour the question needs, some chunk has the pair.

## Why 20 + 40 = 58

This is the arithmetic that looks broken. It isn't — there are two different numbers and they measure different things.

**`total` counts pieces as stored in the buffer, including their leading separators:**

```
piece 3 = '\n\nIndexes are quick.'                       -> 20 chars
piece 4 = '\n\nMigrations version each schema change.'   -> 40 chars
                                                     total = 60
```

**The emitted chunk is those pieces joined and then `.strip()`ed** (`strip_whitespace=True` is the default):

```
join   : '\n\nIndexes are quick.\n\nMigrations version each schema change.'   60 chars
strip  :     'Indexes are quick.\n\nMigrations version each schema change.'   58 chars
          ^^
          the two leading newlines are removed - they were piece 3's
          separator, and there is nothing before them any more
```

So `total = 60` is the packer's internal bookkeeping; `58` is what lands in your database.

Compare with chunk 0, which loses nothing:

```
join   : 'Models map to database tables.\n\nEach field is a column.\n\nIndexes are quick.'   75
strip  : (unchanged - no whitespace at either edge)                                        75
```

> **The rule:** `total` is always ≥ the reported chunk length. The gap is whatever whitespace sat at the edges — usually 0 or 2 characters. It also explains the overlap reading 18 instead of 20: piece 3 is 20 characters *with* its `\n\n`, but only 18 survive at the head of chunk 1.

Every off-by-one and off-by-two in this note is this, and nothing else.

## Now the drain rule itself

```
while total > chunk_overlap:
    drop the OLDEST piece from the FRONT of the buffer
```

**Why the front?** Because the **tail** of chunk N should become the **head** of chunk N+1. Front pieces are oldest — furthest from the seam, and already well covered by the chunk just closed. Back pieces are the ones touching the seam, and those are what the next chunk needs for context.

```
buffer at close:   [ 30 ][ 25 ][ 20 ]
                     ^oldest       ^touching the seam - keep this end
```

Popping from the back would keep `[30]` — the paragraph *furthest* from where the next chunk begins. Useless as a bridge.

## "When do we empty the buffer, and when not?"

Reframe the question, because this is where the flow feels arbitrary:

> **The drain has no "empty it" branch and no "keep some" branch.** There is one loop with one stopping condition. It pops until the condition is satisfied. Sometimes the condition is only satisfied once the buffer is empty — and *that* is all "emptying" means.

Nobody chooses. Both outcomes come out of the same three lines:

```
pieces are small  ->  total drops below 20 while pieces remain  ->  survivors -> OVERLAP
pieces are big    ->  total is still above 20 after the last pop -> empty     -> NO OVERLAP
```

Walk the same loop with fat pieces `[45, 30]` and watch it fall out:

```
total 75:  75 > 20 -> pop 45 -> 30 left
total 30:  30 > 20 -> pop 30 ->  0 left     <- last piece gone, and only NOW is 0 <= 20
total  0:   0 > 20 is false -> stop, buffer empty
```

The loop never decided to empty the buffer. It just never got below 20 until nothing was left. Part 15 turns this into three concrete outcomes.

## What overlap actually *is*

The sentence to memorise, now that you've seen it happen:

> The survivors are **not copied anywhere.** They are the pieces that **were not thrown away** — still sitting in the buffer when the next chunk starts building. That double presence, once in the frozen chunk and once in the live buffer, **is** the overlap.

Overlap here is not a quantity anyone allocates. It is a **residue**. Nobody sets out to produce 20 characters of bridge; 20 is just where the draining stops, and whatever happens to remain becomes the head of the next chunk.

Compare the two mechanisms directly:

| | naive sliding window (Part 6) | PACK's drain |
|---|---|---|
| how overlap is produced | stride arithmetic | leftover after draining |
| amount | **exactly** `overlap`, always | **at most** `chunk_overlap`, often less, sometimes 0 |
| respects boundaries | no | yes |

## Three properties of the drain

1. **Pops from the front, oldest first** — for the reason drawn above.
2. **Pops whole pieces.** It cannot cut a piece in half to land exactly on 20. You land on whatever total the piece boundaries allow: 20, 19, 17, 15, 10 — or 0.
3. **Stops the instant the remainder is `<= chunk_overlap`.** The condition is `while total > 20`, so at exactly 20 it stops. **`chunk_overlap` is a ceiling too**, exactly like `chunk_size`. You will never get more than you asked for; you will very often get less.

**→ The next question:** property 2 says the amount depends on where the piece boundaries happen to fall. So how much actually survives? Part 15 — three outcomes, same settings each time.

# Part 15 — How much survives the drain? Three outcomes

> **Where we are:** we know the drain produces overlap as a residue. Now vary the pieces and watch the residue change, with everything else held constant.

All three scenarios below use the **same** `chunk_size=100`, `chunk_overlap=20`, and all end with an overflow close. Only the piece sizes differ.

## Outcome 1 — the buffer drains to empty (Scenario 3)

Pieces: `45, 30, 40`. Same shape as Scenario 2 — pieces fit, then one overflows — but the pieces are **fatter**.

```
+45 -> total 45   (fits)
+30 -> total 75   (fits)
+40 -> 75+40=115 > 100   CLOSE chunk (internal total 75)
    pop 45 -> 30 left    (30 > 20, keep going)
    pop 30 ->  0 left    (0 > 20 is False -> STOP)
    STOP. carry = [] = 0 chars
end of input -> CLOSE final chunk (internal total 40)
```

Real result:

```
chunk 0: len=75
chunk 1: len=38
overlap 0->1: 0
```

**Zero overlap.** And the key point:

> The drain **did run**, start to finish. It simply had nothing small enough to keep.

Both pieces were individually larger than 20. After dropping the 45, the remaining 30 was still over the ceiling, so it went too. Draining past the last piece leaves you at 0.

### Reason #1 for a total drain

> If **every piece in the buffer is individually bigger than `chunk_overlap`**, the drain cannot stop before it empties. Overlap is structurally impossible — no setting of anything else will produce it.

Scenario 2 vs Scenario 3, identical settings:

| pieces | buffer at close | drain | carry | overlap |
|---|---|---|---|---|
| `30, 25, 20` +40 | 75 | 30 → 45, 25 → 20 stop | `[20]` | **18** |
| `45, 30` +40 | 75 | 45 → 30, 30 → 0 stop | `[]` | **0** |

The only difference is the **granularity of the pieces**. Hold that thought — it is the entirety of Part 20.

## Outcome 2 — several pieces survive (Scenario 4)

Pieces: `32, 27, 17, 10, 9, 55`.

```
+32 -> total 32   (fits)
+27 -> total 59   (fits)
+17 -> total 76   (fits)
+10 -> total 86   (fits)
+9  -> total 95   (fits)
+55 -> 95+55=150 > 100   CLOSE chunk (internal total 95)
    pop 32 -> 63 left    (63 > 20, keep going)
    pop 27 -> 36 left    (36 > 20, keep going)
    pop 17 -> 19 left    (19 > 20 is False -> STOP)
    STOP. carry = [10, 9] = 19 chars      <- BOTH survive
end of input -> CLOSE final chunk (internal total 74)
```

Real result:

```
chunk 0: len=95
chunk 1: len=72
overlap 0->1: 17
```

The loop checks the **remaining total**, not individual pieces. After dropping the 17, what's left is `10 + 9 = 19` — already under 20 — so it stops and keeps **both**.

> The carry is *whatever run of trailing pieces happens to sum to ≤ `chunk_overlap`*. One piece, two, five. It is not "keep one piece"; it is "keep as many as fit under the ceiling."

A fat piece can still block survivors behind it. With buffer `[32, 27, 30, 9]` (total 98), draining goes 98 → 66 → 39 → 9: the 30 is gone even though the 9 beside it survived. Whole pieces only, front first.

## Outcome 3 — survivors existed, and were thrown away anyway (Scenario 5)

Everything so far used the simplified condition `while total > chunk_overlap`. The real source has a second clause:

```python
while total > self._chunk_overlap or (
    total + len_ > self._chunk_size and total > 0
):
```

Keep popping while **either**:

- **(a)** what's left is still bigger than `chunk_overlap`, **or**
- **(b)** what's left **plus the incoming piece** would still overflow, and the buffer isn't empty.

Clause (b) is a safety net: the incoming piece has to actually fit after the drain. There's no point carrying a 19-character tail if the piece arriving next is 95 — `19 + 95 = 114`, and the new chunk would be born already over the limit.

Pieces: `32, 27, 17, 10, 9, 95` — **identical to Scenario 4 except the last piece is 95, not 55.**

```
+32 -> 32,  +27 -> 59,  +17 -> 76,  +10 -> 86,  +9 -> 95   (all fit)
+95 -> 95+95=190 > 100   CLOSE chunk (internal total 95)
    pop 32 -> 63 left    (clause a: 63 > 20)
    pop 27 -> 36 left    (clause a: 36 > 20)
    pop 17 -> 19 left    (clause a satisfied at 19...)
    pop 10 ->  9 left    (clause b: 19+95=114 > 100, incoming still won't fit)
    pop  9 ->  0 left    (clause b:  9+95=104 > 100, incoming still won't fit)
    STOP. carry = [] = 0 chars
end of input -> CLOSE final chunk (internal total 95)
```

Real result:

```
chunk 0: len=95
chunk 1: len=93
overlap 0->1: 0
```

**Zero overlap — and this time perfectly good survivors existed and were discarded.** Clause (a) had already stopped at `[10, 9]`. Clause (b) cleared the buffer because the incoming 95 needed the whole chunk to itself.

### Reason #2 for a total drain

> Even when small pieces exist, the carry is discarded if the **incoming piece is so large that no tail can fit alongside it.**

| scenario | last piece | carry | overlap |
|---|---|---|---|
| 4 | 55 | `[10, 9]` = 19 | **17** |
| 5 | 95 | `[]` | **0** |

First five pieces identical. The size of the *next* piece decided whether the *previous* seam got a bridge.

## Summary of Part 15

```
overflow close -> the drain runs
                      |
                      +-- all pieces > chunk_overlap        -> empties   -> overlap 0
                      +-- incoming piece leaves no room     -> empties   -> overlap 0
                      +-- a small trailing run survives     -> carry     -> OVERLAP
```

## The survival rule, stated exactly

The three outcomes above are three faces of one rule. Here it is in closed form — you can apply it to any buffer without simulating the loop:

> **At an overflow close, the survivors are the LONGEST trailing run of the buffer such that:**
>
> 1. **`sum(run) <= chunk_overlap`**, and
> 2. **`sum(run) + len(incoming piece) <= chunk_size`**
>
> If no such run exists, the buffer empties and there is no overlap.

Both conditions use `<=`, not `<` — a run summing to exactly 20 survives (that was Scenario 2).

I checked this against the library over 20,000 randomly generated buffer + incoming-piece combinations. **Zero mismatches.** The loop and the rule are the same thing.

### The trap in the obvious phrasing

It is tempting to shorten condition 1 to *"a piece survives if it is smaller than `chunk_overlap`"*. That is **necessary but not sufficient**, and the difference bites.

- **Necessary:** a piece longer than `chunk_overlap` can never be overlap, in any document, at any settings. Once the drain reaches it, the total is already above the ceiling, so it gets popped. This is the one-line version of Part 20.
- **Not sufficient:** the condition is on the **sum of the whole surviving run**, not on each piece separately. A small piece is still discarded if the pieces after it already use up the budget.

Watch the difference on two inputs that are identical except for one number:

```
pieces [40, 30, 15, 14] + incoming 30
   buffer at close = 99, both 15 and 14 are individually <= 20
   trailing runs, longest first:
      [15, 14] = 29   -> 29 <= 20?  NO
      [14]     = 14   -> 14 <= 20?  yes;  14 + 30 = 44 <= 100?  yes  -> KEEP
   carry = [14]                    <- the 15 does NOT survive

pieces [40, 30,  6, 14] + incoming 30
   trailing runs, longest first:
      [6, 14]  = 20   -> 20 <= 20?  yes;  20 + 30 = 50 <= 100?  yes  -> KEEP
   carry = [6, 14]                 <- both survive
```

Real output confirms both:

```
[40, 30, 15, 14, 30]  ->  chunks [99, 42]  overlap 12     (only the 14 survived, minus 2 stripped)
[40, 30,  6, 14, 30]  ->  chunks [90, 48]  overlap 18     (6 + 14 = 20 survived, minus 2 stripped)
```

### Checklist you can run in your head

Given a seam, ask these in order. The first "no" ends it:

```
1. Did this chunk close by OVERFLOW?
      no  -> hand-off or end of input -> NO OVERLAP, stop here      (Part 18)
2. Is the last piece <= chunk_overlap?
      no  -> NO OVERLAP - it alone busts the ceiling                (Part 15, outcome 1)
3. How far back can I go before the running sum exceeds chunk_overlap?
      -> that trailing run is your candidate                        (Part 15, outcome 2)
4. Does candidate_sum + incoming_piece fit in chunk_size?
      no  -> drop pieces from the front of the candidate until it does,
             possibly down to nothing                               (Part 15, outcome 3)
5. Whatever is left is the overlap - minus any stripped edge whitespace.   (Part 14)
```

Step 2 is the one worth internalising, because it is checkable *before* you index anything: **compare your typical piece size against `chunk_overlap`.** If pieces are bigger, every seam fails at step 2 and your configured overlap is decorative. That single comparison is what Part 20 measures at production scale.

**→ Next:** put steps C, D and E together across a longer run. Part 16.

---

# Part 16 — A full run, all steps together

> **Where we are:** every mechanism in PACK has now been seen in isolation. Here they are in one uninterrupted run, with two seams behaving differently.

Pieces: `32, 27, 17, 10, 9, 55, 40, 30, 12` — all under 100, so PACK runs once, straight through.

```
+32 -> total 32   (fits)          <- Step C
+27 -> total 59   (fits)
+17 -> total 76   (fits)
+10 -> total 86   (fits)
+9  -> total 95   (fits)

+55 -> 95+55=150 > 100   CLOSE chunk 0 (internal total 95)    <- Step D
    pop 32 -> 63     pop 27 -> 36     pop 17 -> 19   STOP     <- Step E
    carry = [10, 9] = 19
    then add the 55           -> buffer [10, 9, 55], total 74

+40 -> 74+40=114 > 100   CLOSE chunk 1 (internal total 74)    <- Step D
    pop 10 -> 64     (>20, keep going)                        <- Step E
    pop  9 -> 55     (>20, keep going)
    pop 55 ->  0     (0 <= 20, STOP)
    carry = [] = 0            <- the 55 was too fat to survive
    then add the 40           -> buffer [40], total 40

+30 -> total 70   (fits)
+12 -> total 82   (fits)
end of input -> CLOSE chunk 2 (internal total 82)
```

Real result:

```
chunk 0: len=95
chunk 1: len=72
chunk 2: len=80
overlap 0->1: 17
overlap 1->2: 0
```

| chunk | built from | how it closed | overlap with previous |
|---|---|---|---|
| 0 | `32,27,17,10,9` | overflow | — |
| 1 | `10,9,55` | overflow | **17** |
| 2 | `40,30,12` | **end of input** | **0** |

Three things to read off this, all of them callbacks:

- **Seam 0→1 got a bridge** — `[10, 9]` were small enough to survive (Part 15, outcome 2).
- **Seam 1→2 got nothing** — after draining, the only candidate was the 55, still over the 20 ceiling (Part 15, outcome 1). Same run, same settings, adjacent seams, opposite results.
- **Chunk 2 closed because the input ran out**, not by overflow. The drain is *inside* the overflow branch, so it never ran — and there's no chunk 3 to carry into anyway.

That last point is a new way for a chunk to close, and it produces no overlap. **There is one more.** Part 17.

**→ Next:** it's time to un-park Road B.

---

# Part 17 — Road B: the oversized piece

> **Where we are:** Part 11 forked pieces onto two roads and we followed Road A through Parts 12–16. Road B is a piece that is `>= chunk_size` **all by itself** — it cannot fit in a chunk even alone. This is the "recursive" in the class name.

## What happens, in order

1. **Hand off `pending`** — everything collected so far goes to PACK and comes back as finished chunks. `pending` resets to empty.
2. **Recurse** — call SPLIT on *just that piece*, with the **remaining** rungs of the ladder, and pack its output from a **fresh, empty buffer**.

Step 1 exists purely for **ordering**. The pending pieces come earlier in the document than anything the oversized piece will produce, so they must be emitted first. It is a handover, not a rejection.

Step 2 is the recursion: the same procedure, one rung finer, on a smaller input. A paragraph that's too big gets split into lines; if a line is still too big, into words; if a word is still too big, into characters. Each level only descends as far as it has to.

## Trap 1 — the oversized piece never meets the buffer

Go back to Part 13's table:

| | what it compares | where |
|---|---|---|
| **Check 1** | the piece **alone** vs `chunk_size` | SPLIT |
| **Check 2** | buffer **+ piece** vs `chunk_size` | PACK |

> An oversized piece **fails Check 1 and therefore never reaches Check 2.** It was not "tried against the buffer and rejected". It took a different road out of the building.

That one sentence produces the entire trap below.

## Trap 2 — recursion is per-piece, not global

Only the oversized piece descends the ladder. A 400-character paragraph sitting right next to a 3,000-character one is **never** touched by the `" "` splitter. Its neighbour's problem is not its problem.

## Trap 3 — recursion buffers never leak

The recursion packs from an empty buffer, and when it returns, the outer walk continues with an empty buffer again. A tail from inside a recursion is never carried into the pieces that follow it.

## The killer example (Scenario 7)

Two inputs built from **the same five short paragraphs**. Only the sixth differs — and it differs only in length.

```python
shared = ("Django models map to the tables.\n\n"    # piece 1:  32
          "Each field is one column.\n\n"           # piece 2:  27  (25 + leading \n\n)
          "Indexes matter.\n\n"                     # piece 3:  17
          "Cascade.\n\n"                            # piece 4:  10
          "Vacuum.")                                # piece 5:   9

small = "Migrations version each schema change in tight order."          # -> piece 6:  55
big   = ("Migrations are versioned Python files that describe every "
         "schema change and apply them to the database in a very "
         "strict deterministic order without fail.")                     # -> piece 6: 155
```

### Version A — sixth piece = 55 (Road A)

```
SPLIT: pieces [32, 27, 17, 10, 9, 55]   all < 100  ->  all Road A
PACK receives all SIX items:

+32 -> 32,  +27 -> 59,  +17 -> 76,  +10 -> 86,  +9 -> 95
+55 -> 95+55=150 > 100   CLOSE chunk 0 (internal total 95)
     pop 32 -> 63,  pop 27 -> 36,  pop 17 -> 19   STOP
     carry = [10, 9] = 19
end -> CLOSE chunk 1 (internal total 74)
```

Real output:

```
chunk 0 (95): 'Django models map to the tables.\n\nEach field is one column.\n\nIndexes matter.\n\nCascade.\n\nVacuum.'
chunk 1 (72): 'Cascade.\n\nVacuum.\n\nMigrations version each schema change in tight order.'

overlap 0->1: 17  'Cascade.\n\nVacuum.'
```

`Cascade.` and `Vacuum.` appear in both chunks. Standard drain behaviour from Part 14.

### Version B — sixth piece = 155 (Road B)

```
SPLIT: pieces [32, 27, 17, 10, 9, 155]
       155 fails Check 1  ->  Road B
PACK receives only FIVE items:

+32 -> 32,  +27 -> 59,  +17 -> 76,  +10 -> 86,  +9 -> 95
end of list -> CLOSE chunk 0 (internal total 95)     <- no overflow, NO DRAIN
```

Real output:

```
chunk 0 (95): 'Django models map to the tables.\n\nEach field is one column.\n\nIndexes matter.\n\nCascade.\n\nVacuum.'
chunk 1 (93): 'Migrations are versioned Python files that describe every schema change and apply them to the'
chunk 2 (77): 'apply them to the database in a very strict deterministic order without fail.'

overlap 0->1:  0
overlap 1->2: 17  'apply them to the'
```

Same first five paragraphs, and seam 0→1 lost its bridge entirely.

## So where did the 155 actually go?

The trace above says "Road B" and moves on. Here is what Road B actually does to that paragraph — **three nested levels of SPLIT**, each one rung further down the ladder.

### Level 1 — the top-level call

```
SPLIT(full text, ['\n\n', '\n', ' ', ''])
   STEP A: '\n\n' is present -> cut on it
           remaining ladder saved = ['\n', ' ', '']
   pieces: [32, 27, 17, 10, 9, 155]

   STEP B on each:  32 -> Road A -> pending
                    27 -> Road A -> pending
                    17 -> Road A -> pending
                    10 -> Road A -> pending
                     9 -> Road A -> pending
                   155 -> ROAD B:
                            (a) hand off: PACK([32,27,17,10,9]) -> chunk 0 (95)
                            (b) recurse:  SPLIT(the 155 piece, ['\n', ' ', ''])
```

Note **(a) happens before (b)**. Chunk 0 is finished and emitted before the big paragraph is touched at all — that is the ordering guarantee from the top of this part.

### Level 2 — one rung down: `'\n'`

The 155-character piece is not the paragraph text alone. `keep_separator` glued its separator to the front, so the piece literally is:

```
'\n\nMigrations are versioned Python files ... without fail.'
 ^^
 its own leading blank line - 2 of the 155 characters
```

That means the piece **still contains `'\n'`**, so level 2 picks `'\n'`, not `' '`:

```
SPLIT(155-char piece, ['\n', ' ', ''])
   STEP A: '\n' is present -> cut on it
           remaining ladder saved = [' ', '']
   pieces: [1, 154]        <- '\n'  and  '\nMigrations ... fail.'

   STEP B:   1 -> Road A -> pending
           154 -> ROAD B:  (a) hand off: PACK([1])
                               -> joins to '\n', strips to '', dropped. NO CHUNK.
                           (b) recurse: SPLIT(the 154 piece, [' ', ''])
```

This level produces **nothing**. It exists only because the separator got carried along. It's worth seeing once, because it explains the stray empty PACK call you'll notice if you ever instrument the class yourself — and the identical thing happens in Part 19's trace.

### Level 3 — one more rung: `' '` → words

```
SPLIT(154-char piece, [' ', ''])
   STEP A: ' ' is present -> cut on it
           remaining ladder = ['']    (never needed - no word is >= 100)
   pieces: 24 words, every one Road A:

   '\nMigrations'(11) ' are'(4) ' versioned'(10) ' Python'(7) ' files'(6)
   ' that'(5) ' describe'(9) ' every'(6) ' schema'(7) ' change'(7) ' and'(4)
   ' apply'(6) ' them'(5) ' to'(3) ' the'(4) ' database'(9) ' in'(3) ' a'(2)
   ' very'(5) ' strict'(7) ' deterministic'(14) ' order'(6) ' without'(8) ' fail.'(6)
```

Every word is under 100, so all 24 go to PACK **from a fresh, empty buffer** — the 95-character chunk 0 is long gone and cannot influence this.

```
PACK([11,4,10,7,6,5,9,6,7,7,4,6,5,3,4,9,3,2,5,7,14,6,8,6])

+11 -> 11    +4  -> 15    +10 -> 25    +7 -> 32    +6 -> 38
+5  -> 43    +9  -> 52    +6  -> 58    +7 -> 65    +7 -> 72
+4  -> 76    +6  -> 82    +5  -> 87    +3 -> 90    +4 -> 94

+9  -> 94+9=103 > 100   CLOSE chunk 1 (internal total 94)
     pop 11 -> 83     pop 4 -> 79     pop 10 -> 69    pop 7 -> 62
     pop  6 -> 56     pop 5 -> 51     pop  9 -> 42    pop 6 -> 36
     pop  7 -> 29     pop 7 -> 22     pop  4 -> 18    STOP  (18 <= 20)
     carry = [6, 5, 3, 4] = 18   ->  ' apply them to the'

+9 (' database') -> 27
+3 -> 30   +2 -> 32   +5 -> 37   +7 -> 44   +14 -> 58   +6 -> 64
+8 -> 72   +6 -> 78
end -> CLOSE chunk 2 (internal total 78)
```

Which is exactly the real output: chunk 1 = 93 (94 minus a stripped leading newline), chunk 2 = 77 (78 minus a stripped leading space), and the overlap is `' apply them to the'` → `'apply them to the'`, 17 characters.

### The shape of the whole thing

```
SPLIT level 1  ['\n\n','\n',' ','']   ->  chunk 0 (95)            [hand-off, no drain]
   |
   +-- SPLIT level 2  ['\n',' ','']   ->  (nothing)
          |
          +-- SPLIT level 3  [' ','']  ->  chunk 1 (93)
                                           chunk 2 (77)           [overflow -> drain -> 17]
```

Two facts fall straight out of that picture:

- **The seam 0→1 crosses a level boundary.** Chunk 0 was produced by level 1's hand-off; chunk 1 by level 3's packer, from a buffer that started empty. There is no shared buffer between them, so **no mechanism exists that could have bridged them** — regardless of settings.
- **The seam 1→2 is inside one packer run** on word-sized pieces, so the drain works normally and a 17-character bridge survives.

## What just happened

The oversized piece never entered PACK. **There was no sixth item to compare against 100**, so Check 2 never fired — and the drain lives *inside* Check 2's branch, so it never executed. The `Cascade.` and `Vacuum.` paragraphs would have been perfect survivors. They never got the chance.

| 6th piece | passes Check 1? | how chunk 0 closes | drain | carry | overlap at seam 0→1 |
|---|---|---|---|---|---|
| `55` | yes → enters PACK | overflow | **runs** | `[10, 9]` | **17** |
| `155` | no → routed to recursion | hand-off | **never runs** | none | **0** |

> The size of a piece **you cannot even see in the output** determines whether the *previous* seam gets overlap.

And note where the overlap *did* appear in version B: seam 1→2, **inside** the recursion, where pieces are single words of 2–14 characters. Small pieces easily pass the survival rule from Part 15; 27-character paragraphs at a 20 ceiling do not.

> **Overlap clusters where pieces are fine-grained, and disappears where they are coarse.** Remember this for Part 20; it's the whole mechanism behind the production surprise.

**→ Next:** we've now seen every way a chunk can close. Collect them. Part 18.

---

# Part 18 — Master table: four ways a chunk closes

> **Where we are:** Parts 12–17 each showed a different closing event. Here they are together — this table is the summary of the entire mechanism.

Every iteration produces exactly one of these:

| event | first seen | chunk closes? | drain runs? | overlap |
|---|---|---|---|---|
| small piece, fits in buffer | Part 12 | no | no | — |
| **small piece, doesn't fit (overflow)** | Part 13 | **yes** | **yes** | **0 to 20** |
| oversized piece hit → hand-off | Part 17 | yes | **no** | **none** |
| input ran out → final close | Part 16 | yes | **no** | **none** |

The rule in one line:

> **Popping happens when a chunk closes *and the run keeps going*. If the chunk closes because the run is ending, nothing is kept.**

Why the bottom two rows get nothing: in both cases there is no "next chunk in this run" to hand a tail to. Either the input is finished, or we're leaving to go split something else entirely. Nobody selects survivors, so the buffer is just reset.

And within the one row that *can* produce overlap, it often still produces zero (Part 15). The full picture:

```
chunk closes
   |
   +-- by hand-off (oversized piece)   -> no drain          -> overlap 0
   +-- by end of input                 -> no drain          -> overlap 0
   +-- by overflow                     -> drain runs
             |
             +-- all pieces > chunk_overlap                 -> overlap 0
             +-- incoming piece leaves no room              -> overlap 0
             +-- a small trailing run survives              -> OVERLAP
```

**Five paths to a closed chunk. Four of them give zero overlap.**

**→ Next:** all of it, on real prose, in one trace. Part 19.

---

# Part 19 — Real text, end to end

> **Where we are:** every step and every scenario has been seen in isolation. This is the whole machine on the text from Part 4.

```python
text = ("Django models map to tables.\n\n"
        "Each field is a column.\n\n"
        "Migrations are versioned Python files that describe schema changes and "
        "apply them to the database in a deterministic order every single time.")
```

## Full trace

```
SPLIT(len=196, separators=['\n\n', '\n', ' ', ''])
    STEP A: '\n\n' occurs -> split on it -> pieces [28, 25, 143]
                             remaining ladder = ['\n', ' ', '']

    STEP B: 28  -> under 100 -> ROAD A -> pending
    STEP B: 25  -> under 100 -> ROAD A -> pending
    STEP B: 143 -> OVER 100  -> ROAD B -> hand off pending, then recurse

    PACK call #1: pieces = [28, 25]
       +28 -> total 28
       +25 -> total 53
       end of list -> CLOSE chunk of 53      (no overflow -> NO DRAIN)

  SPLIT(len=143, separators=['\n', ' ', ''])
      STEP A: '\n' occurs -> pieces ['\n', '\nMigrations...'(142)]
      PACK call #2: pieces = [1]             (the stray newline; stripped to nothing)
      STEP B: 142 -> still OVER 100 -> recurse again

    SPLIT(len=142, separators=[' ', ''])
        STEP A: ' ' occurs -> 22 word pieces
        PACK call #3: [11,4,10,7,6,5,9,7,8,4,6,5,3,4,9,3,2,14,6,6,7,6]

          +11 -> 11   +4  -> 15   +10 -> 25   +7 -> 32   +6 -> 38
          +5  -> 43   +9  -> 52   +7  -> 59   +8 -> 67   +4 -> 71
          +6  -> 77   +5  -> 82   +3  -> 85   +4 -> 89   +9 -> 98
          +3  -> 98+3=101 > 100   CLOSE chunk (internal total 98)
              pop 11 -> 87   pop 4 -> 83   pop 10 -> 73   pop 7 -> 66
              pop 6  -> 60   pop 5 -> 55   pop 9  -> 46   pop 7 -> 39
              pop 8  -> 31   pop 4 -> 27   pop 6  -> 21   pop 5 -> 16   STOP
              carry = [3, 4, 9] = 16 chars
          +3 -> 19  +2 -> 21  +14 -> 35  +6 -> 41  +6 -> 47  +7 -> 54  +6 -> 60
          end -> CLOSE chunk (internal total 60)
```

## Result

```
chunk 0 (len=53): 'Django models map to tables.\n\nEach field is a column.'
chunk 1 (len=97): 'Migrations are versioned Python files that describe schema changes and apply them to the database'
chunk 2 (len=59): 'to the database in a deterministic order every single time.'

overlap 0 -> 1:  0 chars
overlap 1 -> 2: 15 chars   'to the database'
```

## Every rule in the note, visible in three chunks

- **Compare with Part 4.** Same text, same limit. No chunk starts mid-word; chunk 0 ends exactly at a paragraph boundary. That's SPLIT and PACK doing their two jobs (Part 8).
- **Seam 0→1 has zero overlap.** Chunk 0 closed by **hand-off**, not overflow — the drain never ran (Part 18, row 3). The two chunks also come from different recursion levels, so no bridge was ever possible (Part 17, trap 3).
- **Seam 1→2 has 15 characters, not 20.** The carry was 3 whole word pieces summing to 16, minus a stripped leading space. Granular, not exact (Part 14, property 2).
- **Chunk 1 is 97, not 100.** Ceiling, not target (Part 13). PACK stopped at an internal 98 because the next word would have made 101.
- **Chunk 0 is 53** — barely half the limit — because the only thing left to add was a 143-character piece that took the other road entirely (Part 17).
- **The 15-char overlap appeared inside the recursion**, where pieces are words. Fine-grained pieces → overlap survives (Part 17's closing note).

**→ Next:** that last point, at production scale, is going to cost you something. Part 20.

---

# Part 20 — Production settings: the promise from Part 6, broken

> **Where we are:** Part 6 measured naive overlap at exactly 20 characters at every seam, guaranteed. Part 17 showed overlap survives only where pieces are fine-grained. Now put realistic numbers in and see which of those two worlds you actually live in.

## The experiment

A realistic document: **30 paragraphs, 241–617 characters each, 13,096 characters total.** Split at `chunk_size=1600, chunk_overlap=200` with the default ladder.

```
chunks = 9
seams  = 8
seams with any overlap = 0        <-- zero. not "small". zero.
overlaps = [0, 0, 0, 0, 0, 0, 0, 0]
```

You asked for 200 characters of bridge at every seam. You got none, anywhere.

## Why — three facts you already know

1. Every paragraph is **under 1600**, so every one passes Check 1 and reaches PACK as a piece (Part 11).
2. Pieces are therefore **241–617 characters**. Every single one is **bigger than `chunk_overlap=200`** (Part 15, reason #1).
3. So when a chunk overflows, the drain finds nothing small enough to keep. It empties. Every time.

This is **Scenario 3 exactly** (`45, 30` with overlap 20), scaled up 15×.

> **Overlap only materialises when your typical piece is *smaller* than `chunk_overlap`.**

Where overlap *does* appear in real documents: inside a single paragraph long enough to exceed `chunk_size` and get recursed down to words. Words are tiny, so plenty survive. Which gives the general shape:

> Overlap **clusters inside long paragraphs and vanishes at paragraph boundaries.**

## The fix that does NOT work

The advice you'll find everywhere is to add a sentence-level separator:

```python
RecursiveCharacterTextSplitter(
    chunk_size=1600, chunk_overlap=200,
    separators=["\n\n", "\n", ". ", " ", ""],   # looks right, does nothing here
)
```

Same document, measured:

```
chunks = 9
seams with any overlap = 0        <-- identical. no change whatsoever.
```

**Why it fails:** re-read the ladder rule from Part 10. SPLIT picks the **first separator present in the text** and cuts on that one only. `"\n\n"` is present, so it splits on paragraphs and **stops**. `". "` is never reached — it can only be used inside a piece that failed Check 1, i.e. a paragraph longer than 1600 characters. If your paragraphs are 250–600 characters, adding `". "` to the ladder changes literally nothing.

This is the single most common misunderstanding about this class, and it comes straight from reading the ladder as "try all of these" instead of "pick one".

## Two fixes that do work

Same document, same `chunk_size=1600`.

**Fix A — remove the paragraph separators, so pieces *are* sentences:**

```python
RecursiveCharacterTextSplitter(
    chunk_size=1600, chunk_overlap=200,
    separators=[". ", " ", ""],      # no "\n\n", no "\n"
)
```

```
chunks = 10
seams with overlap = 9 / 9
overlaps = [200, 135, 199, 179, 142, 178, 92, 142, 128]
```

The first separator present is now `". "`, so pieces are sentences of ~60–90 characters. Several fit under the 200 ceiling, so every seam gets a real bridge.
**Cost:** paragraph boundaries are no longer preferred cut points, so a chunk can begin mid-paragraph. You traded some of requirement 2 for requirement 3.

**Fix B — raise `chunk_overlap` above your typical paragraph size:**

```python
RecursiveCharacterTextSplitter(chunk_size=1600, chunk_overlap=700)
```

```
chunks = 14
seams with overlap = 13 / 13
overlaps = [291, 617, 496, 610, 390, 632, 390, 653, 454, 490, 546, 489, 532]
```

Paragraphs now routinely fit under the ceiling, so whole paragraphs survive the drain.
**Cost:** 14 chunks instead of 9 — ~55% more text stored and embedded, and duplicate paragraphs will surface repeatedly in one top-k result set. This is Part 6's "overlap is not free", with a bill attached.

## How to choose

| your paragraphs are... | what to do |
|---|---|
| mostly **larger** than `chunk_size` | default ladder is fine — recursion reaches words, overlap appears naturally |
| mostly **smaller** than `chunk_overlap` | default ladder is fine — whole paragraphs survive the drain |
| **between** the two (the common case) | pick Fix A or Fix B deliberately, or accept hard cuts at paragraph seams |
| you need *guaranteed uniform* overlap | this class is the wrong tool — Part 6's sliding window, or a token-window splitter, gives you that |

The honest fourth option is **accept it**. A paragraph boundary is a genuinely good place for a hard cut — the topic changes there anyway, which is why the ladder ranks it first. The problem isn't the zero; it's believing you have 200 when you have 0.

## Measure your own corpus

Don't infer it, check it:

```python
chunks = splitter.split_text(doc)

def overlap(a, b):
    for k in range(min(len(a), len(b)), 0, -1):
        if a[-k:] == b[:k]:
            return k
    return 0

seams = [overlap(chunks[i], chunks[i+1]) for i in range(len(chunks) - 1)]
print(f"{sum(1 for s in seams if s)}/{len(seams)} seams have overlap; "
      f"mean {sum(seams)/len(seams):.0f}")
```

**→ Next:** the remaining sharp edges. Part 21.

---

# Part 21 — Gotchas

> **Where we are:** the mechanism is fully explained. These are the consequences that bite in production.

**Overlap is granular, not exact.** Assembled from whole pieces, it lands *at or under* `chunk_overlap`, on whatever value the boundaries allow: 20, 19, 17, 15, 10 — or 0. Part 14, property 2.

**`chunk_size` is a ceiling, but it *can* be exceeded.** If a piece has no remaining separator (one unbroken token — base64, a long URL, a minified line), the ladder runs out and it's emitted as-is:

```
Created a chunk of size N, which is longer than the specified 100
```

That warning means a chunk in your index is oversized and may be silently truncated by the embedding API. Don't ignore it. (This is the `if not remaining` branch in Part 9's pseudocode.)

**Chunk boundaries are content-dependent.** Insert one paragraph near the top of a document and **every downstream boundary shifts**, because all of PACK's running totals change. This bites hard if you hash chunks to skip re-embedding on re-upload: text you never touched now produces different chunk text, therefore different hashes, therefore a full re-embed. The blast radius of a one-paragraph edit is the whole document, not the paragraph.

**Whitespace is stripped from chunk edges** (`strip_whitespace=True`). Final chunk lengths run 1–3 characters under the internal `total` in every trace above, and reported overlap runs under the carry total for the same reason. Every apparent off-by-one or off-by-two in this note is this — worked through character by character in Part 14 ("Why 20 + 40 = 58").

**Empty pieces vanish.** After stripping, an all-whitespace chunk becomes `None` and is dropped — which is why the stray `'\n'` in Part 19 produced no chunk at all.

---

# Part 22 — Where this lives in LangChain, and what generalises

> **Where we are:** everything above described one class. How much of it transfers?

**PACK is shared.** `buffer`, the fit check and the drain all live in `_merge_splits` on the base `TextSplitter`. `CharacterTextSplitter` and `TokenTextSplitter` use the **same** packer — so Parts 12–16 and 18 are true of them too. `chunk_size` and `chunk_overlap` behave identically across all of them.

**SPLIT is the specific part.** The ladder, Check 1, and hand-off-then-recurse are `_split_text` on `RecursiveCharacterTextSplitter`.

**The language splitters are this exact class with a different ladder:**

```python
Language.PYTHON -> ["\nclass ", "\ndef ", "\n\tdef ", "\n\n", "\n", " ", ""]
```

`MarkdownTextSplitter`, `PythonCodeTextSplitter`, `LatexTextSplitter` — same flow exactly, just trying class and function boundaries before falling back to blank lines. Everything in this note applies unchanged; only the ladder differs. Note that Part 20's trap applies here too: if `"\nclass "` is present, that's the only separator used at the top level.

**`length_function` changes the unit of everything.** The default `len` means every number in this note is characters. Pass a tokenizer's count function and identical logic runs in **tokens** — buffer totals, the ceiling and the drain threshold all become token counts. Usually what you want, since both limits from Part 2 — the LLM's context window and the embedding model's input cap — are measured in tokens, and character count is only a rough proxy.

**What doesn't generalise:** the assumption that overlap is uniform. That belongs to the naive sliding window of Part 6, not to anything built on this packer.

---

# Cheat sheet

## The whole thing in one paragraph

Split the text on the **first separator that appears in it**, and keep the rest of the ladder for later. Walk the resulting pieces in document order. Any piece already too big to be an ingredient causes everything pending to be **packed and emitted first**, and is then split again with the remaining separators, from a clean slate. Everything else accumulates and goes to the packer, which fills a **buffer** until the next piece would overflow the limit — at which point the chunk **closes without that piece** and the buffer is **drained from the front** until what remains fits under `chunk_overlap` *and* leaves room for the incoming piece. Whatever survives that drain is already sitting in the buffer when the next chunk starts, and that leftover **is** the overlap. Which is why overlap appears only where pieces are small, and vanishes at every seam created by a hand-off.

## Quick lookup

| question | answer | part |
|---|---|---|
| Why chunk at all? | the context window can't hold your corpus, and broad context degrades answers | 2 |
| Why not paste the whole doc when it *does* fit? | lost-in-the-middle, and blending real sentences into wrong answers | 2 |
| Why not slice every N chars? | breaks words, orphans subjects | 4 |
| Why not split on paragraphs alone? | no size guarantee at all | 5 |
| Which separator gets used? | the **first one present**; the rest are saved, not used | 10 |
| What goes in the buffer? | every Road A piece, in order, while it still fits | 12 |
| When does a chunk close? | `total + next > chunk_size`, or hand-off, or end of input | 13, 18 |
| Is the causing piece in the closed chunk? | **No.** It starts the next one. | 13 |
| Does the drain shrink the chunk that just closed? | **No.** The chunk is a snapshot taken before the drain runs. | 14 |
| So a surviving piece is in two chunks? | **Yes** — frozen in chunk N, still live in the buffer for chunk N+1. That *is* the overlap. | 14 |
| Why is a chunk shorter than `total`? | `total` counts leading separators; the emitted chunk is stripped at both edges. | 14 |
| When does draining start? | only after an **overflow** close — never after a hand-off | 18 |
| Which end do we pop from? | the **front**; the tail is what bridges to the next chunk | 14 |
| When does draining stop? | `total <= chunk_overlap` **and** `total + incoming <= chunk_size` | 14, 15 |
| When is the buffer fully drained? | all pieces > `chunk_overlap`; or incoming piece too big to share | 15 |
| When do several pieces survive? | when the trailing run sums to ≤ `chunk_overlap` | 15 |
| Exact survival rule? | longest trailing run with `sum ≤ chunk_overlap` **and** `sum + incoming ≤ chunk_size` | 15 |
| Is "piece < chunk_overlap" enough? | **Necessary, not sufficient** — the limit is on the run's sum, not each piece | 15 |
| When is overlap zero? | hand-off close; end of input; fat buffer; fat incoming piece; across recursion levels | 18 |
| Is `chunk_size` a target? | No — a **ceiling**. So is `chunk_overlap`. | 13, 14 |

## Diagnostic

```
No overlap anywhere?
   -> your pieces are bigger than chunk_overlap. Measure paragraph lengths.  -> Part 20
Overlap only in some places?
   -> normal. those are long paragraphs that got recursed down to words.     -> Part 17
Chunks much smaller than chunk_size?
   -> a big piece next door forced an early hand-off.                        -> Part 17
Added ". " to separators and nothing changed?
   -> "\n\n" is still present, so the ladder never reaches it.               -> Part 20
"Created a chunk of size N" warning?
   -> one unbreakable token longer than chunk_size.                          -> Part 21
Re-upload re-embeds everything?
   -> boundary shift. one edit moves every downstream boundary.              -> Part 21
```