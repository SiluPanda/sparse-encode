# sparse-encode -- Specification

## 1. Overview

`sparse-encode` is a JavaScript library for generating BM25 and TF-IDF sparse vectors suitable for hybrid search with Pinecone, Qdrant, and Milvus. It accepts text, tokenizes it, scores each term using BM25 or TF-IDF against a fitted corpus, maps terms to integer indices, and returns a sparse vector in the `{ indices: number[], values: number[] }` format that all major vector databases expect. It provides both a TypeScript/JavaScript API for programmatic use and a CLI for encoding text, fitting corpora, and exporting fitted models.

The gap this package fills is specific and well-defined. Hybrid search -- combining dense embeddings (semantic similarity) with sparse vectors (keyword matching) -- is the recommended retrieval strategy for production RAG pipelines. Pinecone, Qdrant, and Milvus all support hybrid queries that accept both a dense vector and a sparse vector and fuse the results. The dense vector captures meaning ("what is the document about?"), while the sparse vector captures exact terms ("does the document contain this specific keyword?"). In practice, hybrid search consistently outperforms either approach alone, particularly for domain-specific terminology, proper nouns, code identifiers, and long-tail queries that dense models underrepresent.

Generating sparse vectors in Python is straightforward. Pinecone publishes `pinecone-text`, which provides `BM25Encoder` and `SpladeEncoder` classes that fit on a corpus, encode documents and queries into sparse vectors, and serialize fitted models for reuse. The `rank_bm25` library provides BM25 scoring for ranking. Scikit-learn provides `TfidfVectorizer` with full sparse matrix output. Every Python-based RAG tutorial uses one of these tools.

In JavaScript, no equivalent exists. The `bm25` npm package is abandoned (last publish 2017), does not produce sparse vector output, and is incompatible with modern vector databases. The `natural` npm package provides TF-IDF scoring but returns dense objects keyed by document index, not sparse vectors keyed by term index -- the format vector databases need. Multiple Pinecone community forum posts ask "How do I generate sparse vectors in JavaScript?" and the answer is invariably "use Python" or "write your own." Teams building JavaScript-native RAG pipelines, serverless functions, edge workers, or browser-based search interfaces have no way to generate the sparse vectors that Pinecone, Qdrant, and Milvus require without calling out to a Python service.

`sparse-encode` fills this gap. It implements BM25 (Okapi BM25) and TF-IDF scoring from scratch, with configurable tokenization, built-in English stopwords, optional Porter stemming, hash-based and vocabulary-based term-to-index mapping, corpus fitting for IDF statistics, model serialization for deployment, and separate document/query encoding paths. The output is the exact `{ indices: number[], values: number[] }` format that Pinecone, Qdrant, and Milvus sparse vector fields expect.

---

## 2. Goals and Non-Goals

### Goals

- Provide a `createBM25(options?)` factory that returns a `BM25Encoder` implementing the Okapi BM25 scoring algorithm with configurable k1 and b parameters.
- Provide a `createTFIDF(options?)` factory that returns a `TFIDFEncoder` implementing TF-IDF scoring with configurable TF variants (raw, log-normalized, augmented) and IDF smoothing.
- Provide a `encoder.fit(documents)` method that computes IDF statistics (document frequencies, average document length) from a corpus of text documents, required before encoding.
- Provide a `encoder.encode(text)` method that returns a `SparseVector` representing a document's term scores as `{ indices: number[], values: number[] }`.
- Provide a `encoder.encodeBatch(texts)` method that encodes multiple documents in a single call, returning `SparseVector[]`.
- Provide a `encoder.encodeQuery(query)` method that returns a `SparseVector` using query-specific scoring (no document length normalization, IDF-weighted term presence), matching the distinction Pinecone requires between document and query encoding.
- Provide a `encoder.serialize()` method and a static `BM25Encoder.deserialize()` / `TFIDFEncoder.deserialize()` method for saving and loading fitted models as JSON, enabling deployment without re-fitting.
- Implement hash-based term-to-index mapping using MurmurHash3 modulo a configurable vocabulary size (default: 2^18 = 262,144), matching Pinecone's `pinecone-text` approach. No vocabulary file needed.
- Implement vocabulary-based term-to-index mapping as an alternative, where terms are assigned sequential integer IDs from the fitted corpus. Exact mapping, no collisions, but requires the vocabulary to be present at encoding time.
- Implement a default tokenizer: lowercase, split on whitespace and punctuation, remove English stopwords, with optional Porter stemming.
- Accept a custom tokenizer function `(text: string) => string[]` for language-specific or domain-specific tokenization.
- Produce output compatible with Pinecone sparse vector format, Qdrant sparse vector format, and Milvus sparse vector format without transformation.
- Provide a CLI (`sparse-encode`) for fitting corpora, encoding text, encoding queries, and exporting fitted models.
- Integrate with `embed-cache` (dense vectors) and `fusion-rank` (score fusion) from this monorepo for complete hybrid search pipelines.
- Zero mandatory runtime dependencies. Tokenization, hashing, stemming, and scoring are all self-contained.
- Target Node.js 18 and above.

### Non-Goals

- **Not a dense embedding provider.** This package generates sparse vectors from term statistics. It does not call any embedding API and does not produce dense semantic vectors. Use OpenAI, Cohere, or a local model for dense embeddings. Use `embed-cache` for caching them.
- **Not a SPLADE model.** SPLADE (SParse Lexical AnD Expansion) is a learned sparse representation that uses a transformer model to assign term weights. `sparse-encode` implements classical BM25 and TF-IDF, which are unsupervised statistical methods. SPLADE requires a neural model at inference time; BM25 and TF-IDF require only precomputed corpus statistics.
- **Not a full-text search engine.** This package scores individual documents or queries against a fitted corpus and returns sparse vectors. It does not maintain an inverted index, does not rank documents against a query, and does not perform retrieval. Use the sparse vectors with a vector database (Pinecone, Qdrant, Milvus) for retrieval.
- **Not a multilingual NLP pipeline.** The built-in tokenizer and stopword list are English-focused. Other languages are supported through the custom tokenizer hook, but `sparse-encode` does not include language detection, multilingual stopword lists, or language-specific stemmers beyond Porter (English). For multilingual production deployments, supply a language-appropriate tokenizer.
- **Not a corpus storage or indexing system.** The `fit()` method computes aggregate statistics (document frequencies, average document length) from a corpus but does not store the corpus itself. After fitting, the individual documents are not retained in memory.
- **Not a ranking algorithm.** BM25 is often used as a ranking function (score documents against a query and sort by score). `sparse-encode` uses BM25 and TF-IDF as scoring functions to produce sparse vector values, not to rank a set of candidate documents. Ranking is handled by the vector database's hybrid query engine.

---

## 3. Target Users and Use Cases

### Hybrid Search Pipeline Engineers

Teams building JavaScript-native RAG pipelines that combine dense and sparse retrieval for Pinecone, Qdrant, or Milvus. They need to generate sparse vectors from their document corpus and from incoming queries, in the exact format their vector database expects. Today they either call a Python sidecar service or skip sparse retrieval entirely. With `sparse-encode`, they fit a BM25 encoder on their corpus, encode documents during indexing, and encode queries at search time -- all in JavaScript, no Python dependency.

### Pinecone Users in JavaScript Environments

Pinecone's hybrid search requires both a dense vector and a sparse vector per document and per query. Pinecone's documentation and examples use `pinecone-text` (Python) to generate sparse vectors. JavaScript and TypeScript users of the `@pinecone-database/pinecone` client library have no equivalent. `sparse-encode` produces sparse vectors in the exact format Pinecone's `upsert` and `query` APIs accept.

### Qdrant and Milvus Users

Qdrant supports sparse vectors via named vector fields with `{ indices, values }` payloads. Milvus 2.4+ supports sparse vectors as a first-class field type. Both require the same `{ indices: number[], values: number[] }` format. `sparse-encode` produces this format directly.

### Serverless and Edge Deployments

Teams deploying search functionality on Cloudflare Workers, Vercel Edge Functions, AWS Lambda, or Deno Deploy. These environments run JavaScript natively. Calling a Python microservice adds latency and infrastructure complexity. A zero-dependency JavaScript BM25 encoder that serializes to a JSON blob runs natively in any JavaScript runtime with no cold-start penalty beyond loading the fitted model.

### RAG Pipeline Developers Building Keyword-Aware Retrieval

Developers who have observed that pure dense retrieval misses exact keyword matches -- product IDs, error codes, API names, medical terms, legal citations -- and want to add keyword-aware sparse retrieval without abandoning their JavaScript stack. BM25 sparse vectors excel at exact term matching while dense vectors handle semantic paraphrase.

---

## 4. Core Concepts

### Sparse Vector

A sparse vector represents a point in a high-dimensional space where most dimensions have a value of zero. Only the non-zero dimensions are stored. The representation is a pair of parallel arrays: `indices` (the dimensions that have non-zero values) and `values` (the corresponding values at those dimensions).

```typescript
interface SparseVector {
  indices: number[];  // sorted ascending, each value in [0, vocabSize)
  values: number[];   // same length as indices, each value > 0
}
```

For a vocabulary of 262,144 possible terms, a document containing 50 unique terms produces a sparse vector with 50 non-zero entries out of 262,144 possible dimensions -- a sparsity of 99.98%. This is far more memory-efficient than storing a dense 262,144-dimensional vector where 262,094 entries are zero.

The sparse vector format is the standard interchange format for keyword-based retrieval in vector databases. Pinecone, Qdrant, and Milvus all accept sparse vectors in this representation. Each non-zero dimension corresponds to a term in the vocabulary, and the value at that dimension is the term's importance score (BM25 or TF-IDF score).

### Dense Vector vs. Sparse Vector

Dense vectors and sparse vectors capture different aspects of text. A dense vector is the output of a neural embedding model -- a fixed-length array of floating-point numbers (typically 768 to 3072 dimensions) where every dimension has a non-zero value. Dense vectors encode semantic meaning: "automobile" and "car" have similar dense vectors because they mean the same thing, even though they share no characters. Dense vectors cannot distinguish between a document that mentions "CUDA 12.4" and one that does not, because neural embeddings do not represent specific tokens as individual dimensions.

A sparse vector is derived from term statistics. Each dimension corresponds to a specific term (or a hash bucket that maps to one or more terms). A document that contains the term "CUDA" has a non-zero value at the dimension corresponding to "CUDA"; a document that does not contain "CUDA" has a zero at that dimension. Sparse vectors perform exact keyword matching: they find documents that contain the query terms, weighted by how important those terms are (IDF) and how prominently they appear in the document (TF).

Hybrid search combines both: the dense vector finds semantically similar documents, the sparse vector finds keyword-matched documents, and the vector database fuses the results.

### BM25 (Okapi BM25)

BM25 (Best Matching 25) is a probabilistic term-weighting function used for information retrieval. It is an evolution of TF-IDF that addresses two limitations of raw TF-IDF: unbounded term frequency and no document length normalization.

The BM25 score for a query Q containing terms q_1, q_2, ..., q_n against a document D is:

```
score(D, Q) = sum_{i=1}^{n} IDF(q_i) * (f(q_i, D) * (k1 + 1)) / (f(q_i, D) + k1 * (1 - b + b * |D| / avgdl))
```

Where:
- `f(q_i, D)` is the term frequency of q_i in document D (raw count of occurrences).
- `|D|` is the length of document D in terms.
- `avgdl` is the average document length across the corpus.
- `k1` is the term frequency saturation parameter (default: 1.2).
- `b` is the document length normalization parameter (default: 0.75).
- `IDF(q_i)` is the inverse document frequency of q_i.

The key innovation of BM25 over raw TF-IDF is the saturation curve. In raw TF-IDF, a term appearing 100 times in a document scores 100 times more than a term appearing once. In BM25, the `(f * (k1 + 1)) / (f + k1 * ...)` term creates a saturation effect: the first few occurrences of a term contribute significantly to the score, but additional occurrences contribute less and less. A term appearing 100 times scores only marginally more than a term appearing 10 times. This prevents long documents from dominating search results simply because they repeat terms more often.

The `k1` parameter controls how quickly saturation occurs. At k1 = 0, term frequency is ignored entirely (Boolean matching). At k1 = 1.2 (the default), the score saturates around 5-10 occurrences. At very large k1, BM25 approaches raw TF weighting.

The `b` parameter controls document length normalization. At b = 1, a document twice the average length is penalized heavily (its effective term frequency is halved). At b = 0, no length normalization occurs. At b = 0.75 (the default), there is moderate length normalization that prevents long documents from scoring higher simply because they have more room for term occurrences.

### IDF (Inverse Document Frequency)

IDF measures how informative a term is. A term that appears in every document (like "the") has low IDF -- it provides no discriminative value. A term that appears in only one document (like "mitochondria") has high IDF -- finding it in a query strongly suggests that specific document is relevant.

BM25 uses a specific IDF variant:

```
IDF(t) = log((N - df(t) + 0.5) / (df(t) + 0.5) + 1)
```

Where:
- `N` is the total number of documents in the corpus.
- `df(t)` is the number of documents containing term t (the document frequency).

The `+ 0.5` terms provide smoothing, preventing division by zero when df = 0 (unseen terms) or df = N (terms in every document). The `+ 1` inside the log prevents negative IDF values for very common terms.

TF-IDF uses a simpler IDF variant:

```
IDF(t) = log(N / df(t))
```

Or with smoothing:

```
IDF(t) = log(1 + N / df(t))
```

### TF-IDF

TF-IDF (Term Frequency -- Inverse Document Frequency) is the product of a term's frequency in a document and its inverse document frequency across the corpus. It weights terms that are both frequent in the current document and rare across the corpus -- the terms most likely to distinguish this document from others.

`sparse-encode` supports three TF variants:

- **Raw TF**: `TF(t, D) = f(t, D)` -- the raw count of term occurrences. Simple but unbounded; penalizes short documents.
- **Log-normalized TF** (sublinear TF): `TF(t, D) = 1 + log(f(t, D))` if f > 0, else 0. Dampens the effect of high term frequencies. This is the default and recommended variant.
- **Augmented TF**: `TF(t, D) = 0.5 + 0.5 * f(t, D) / max_f(D)` -- normalizes by the maximum term frequency in the document. Useful when comparing documents of very different lengths.

The final TF-IDF score is `TF(t, D) * IDF(t)`. The resulting vector is optionally L2-normalized to unit length, which is recommended when the sparse vector will be combined with a dense vector in a hybrid search (ensures comparable magnitudes).

### Term-to-Index Mapping

A sparse vector requires integer indices. Each term in the vocabulary must map to a unique integer in `[0, vocabSize)`. `sparse-encode` provides two mapping strategies.

**Hash-based mapping** (default): Each term is hashed using MurmurHash3 (32-bit) and the result is taken modulo `vocabSize`. No vocabulary file is needed. The mapping is deterministic -- the same term always maps to the same index. The downside is hash collisions: two different terms may map to the same index, in which case their scores are summed. With vocabSize = 262,144 (2^18) and a typical document vocabulary of 50-200 terms, the collision probability is low (birthday problem: ~1% for 50 terms in 262,144 buckets). Hash-based mapping is the approach used by Pinecone's `pinecone-text` (Python), ensuring compatibility.

**Vocabulary-based mapping**: During `fit()`, every unique term encountered in the corpus is assigned a sequential integer ID starting from 0. The vocabulary is a `Map<string, number>` stored as part of the fitted model. This approach guarantees no collisions but requires the vocabulary to be available at encoding time. Terms not in the vocabulary (out-of-vocabulary terms) can be handled in three ways: ignore (skip the term), hash (fall back to hash-based mapping for unknown terms), or error (throw).

### Tokenization

Tokenization is the process of splitting raw text into a sequence of terms for scoring. The quality of tokenization directly affects retrieval quality: if important terms are not isolated as individual tokens, they will not appear in the sparse vector.

`sparse-encode` provides a built-in default tokenizer and accepts custom tokenizer functions.

The default tokenizer pipeline:
1. **Lowercase**: Convert the entire text to lowercase. "BM25" becomes "bm25".
2. **Punctuation removal**: Replace all non-alphanumeric characters (except hyphens within words) with spaces. "state-of-the-art" remains "state-of-the-art"; "hello, world!" becomes "hello  world ".
3. **Whitespace splitting**: Split on one or more whitespace characters.
4. **Stopword removal**: Remove tokens that appear in the built-in English stopword list (~175 common function words: "the", "is", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", "of", "with", "by", "from", "as", "into", "through", "during", "before", "after", "above", "below", "between", "out", "off", "over", "under", "again", "further", "then", "once", etc.).
5. **Minimum length filter**: Remove tokens shorter than 2 characters. Single characters are rarely meaningful search terms.
6. **Optional stemming**: If `stemming: true` is configured, apply the Porter stemmer to each token. "running" becomes "run", "connections" becomes "connect". Stemming increases recall (matching different word forms) at the cost of precision (collapsing distinct meanings).

The custom tokenizer hook replaces the entire pipeline:

```typescript
const encoder = createBM25({
  tokenizer: (text: string): string[] => {
    // Custom tokenization for code search
    return text.split(/[^a-zA-Z0-9_]+/).filter(t => t.length > 0);
  },
});
```

### Corpus Fitting

BM25 and TF-IDF both require corpus-level statistics: document frequencies (how many documents contain each term) and, for BM25, the average document length. These statistics are computed during the `fit()` step.

Fitting is a one-time operation. The encoder processes every document in the corpus, tokenizes each document, counts term occurrences, and aggregates:
- `N`: total number of documents.
- `df(t)`: the number of documents containing term t, for every term t encountered.
- `avgdl`: the average document length (in tokens) across all documents.
- Total tokens processed (for diagnostics).

After fitting, the encoder can encode any text against the fitted statistics. The fit does not need to be repeated unless the corpus changes substantially. For incremental updates, `encoder.partialFit(newDocuments)` updates the statistics without re-processing the entire corpus.

Fitting is fast: tokenizing and counting term frequencies across 100,000 documents of ~500 tokens each takes approximately 1-3 seconds on modern hardware.

### Hybrid Search

Hybrid search is a retrieval strategy that combines the results of dense (semantic) retrieval and sparse (keyword) retrieval. The vector database accepts both a dense query vector and a sparse query vector, retrieves candidate documents using both vector types, and fuses the results.

The fusion formula varies by database:
- **Pinecone**: `final_score = alpha * dense_score + (1 - alpha) * sparse_score`, where `alpha` is configurable per query.
- **Qdrant**: Separate dense and sparse sub-queries with configurable fusion via Reciprocal Rank Fusion (RRF) or linear combination.
- **Milvus**: Separate ANN index and sparse inverted index, with ranker fusion in the query plan.

In all cases, the sparse vector is expected in `{ indices: number[], values: number[] }` format. `sparse-encode` produces this format.

---

## 5. BM25 Algorithm

### Complete BM25 Formula

For a single term t in a document D with fitted corpus statistics:

```
BM25(t, D) = IDF(t) * (f(t, D) * (k1 + 1)) / (f(t, D) + k1 * (1 - b + b * |D| / avgdl))
```

This is the per-term score. A document's sparse vector contains one entry per unique term in the document, where the index is the term's mapped integer and the value is the BM25 score for that term.

### IDF Component

```
IDF(t) = log((N - df(t) + 0.5) / (df(t) + 0.5) + 1)
```

This is the Robertson-Sparck Jones IDF variant used in the original Okapi BM25 formulation. Properties:
- When df(t) = 0 (term never seen in corpus): IDF = log(N + 1), the maximum value. Unseen terms are treated as maximally informative.
- When df(t) = N (term in every document): IDF = log(1 + 0.5 / (N + 0.5)), which approaches log(1) = 0 as N grows. Universal terms contribute almost nothing.
- When df(t) = N/2 (term in half the documents): IDF = log(2), a moderate score.
- The `+ 1` inside the log ensures IDF is never negative. Without it, terms appearing in more than half the documents would have negative IDF (counter-intuitive for sparse vector values).

### Term Frequency Saturation

The numerator `f(t, D) * (k1 + 1)` and denominator `f(t, D) + k1 * (1 - b + b * |D| / avgdl)` together create a saturation curve:

- When `f(t, D) = 0`: score = 0 (term not in document).
- When `f(t, D) = 1`: score = `IDF * (k1 + 1) / (1 + k1 * (1 - b + b * |D| / avgdl))`.
- As `f(t, D)` approaches infinity: score approaches `IDF * (k1 + 1)` -- the saturation ceiling.

For k1 = 1.2 and b = 0.75 with a document of average length (|D| = avgdl):
- f = 1: effective TF = 1.22
- f = 2: effective TF = 1.76
- f = 5: effective TF = 2.07
- f = 10: effective TF = 2.15
- f = 100: effective TF = 2.19

The first occurrence contributes significantly; subsequent occurrences contribute diminishing amounts. This is the core advantage of BM25 over raw TF-IDF.

### Document Length Normalization

The term `k1 * (1 - b + b * |D| / avgdl)` in the denominator normalizes for document length:

- If `|D| = avgdl` (average-length document): the denominator simplifies to `f + k1`, no length adjustment.
- If `|D| > avgdl` (long document): the denominator increases, reducing the score. Long documents contain more terms by chance; length normalization compensates.
- If `|D| < avgdl` (short document): the denominator decreases, boosting the score. Short documents that contain a term are more likely to be about that term.

At b = 0, no length normalization: `1 - 0 + 0 * |D|/avgdl = 1`, so the denominator is `f + k1` regardless of document length.

At b = 1, full length normalization: `1 - 1 + 1 * |D|/avgdl = |D|/avgdl`, so a document twice the average length has its effective term frequency halved.

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `k1` | `number` | `1.2` | Term frequency saturation. Higher values allow more influence from repeated terms. Range: [0, 3]. |
| `b` | `number` | `0.75` | Document length normalization. 0 = no normalization, 1 = full normalization. Range: [0, 1]. |

The defaults (k1 = 1.2, b = 0.75) are the values used in the original Okapi BM25 paper and are widely adopted across search engines including Elasticsearch, Solr, and Lucene. They are suitable for most English text corpora. Tuning recommendations:
- For short documents (tweets, titles, product names): reduce b to 0.3-0.5 (less length normalization, since all documents are short).
- For code search: increase k1 to 1.5-2.0 (allow more weight for repeated identifiers) and reduce b to 0.5.
- For long documents (articles, papers): the defaults work well.

### How BM25 Scores Become Sparse Vector Values

When encoding a document:
1. Tokenize the document text into terms.
2. Count the frequency of each unique term in the document.
3. For each unique term:
   a. Look up or compute `IDF(t)` from the fitted corpus statistics.
   b. Compute the BM25 score for this term in this document.
   c. Map the term to an integer index (hash-based or vocabulary-based).
   d. If the index already has a value (hash collision), sum the scores.
4. Collect all (index, value) pairs where value > 0.
5. Sort indices ascending.
6. Return `{ indices, values }`.

### Fitting: Computing IDF Statistics

The `fit(documents)` method processes the corpus:

```
N = documents.length
avgdl = 0
df = new Map<string, number>()

for each document in documents:
  tokens = tokenize(document)
  avgdl += tokens.length
  uniqueTerms = new Set(tokens)
  for each term in uniqueTerms:
    df.set(term, (df.get(term) || 0) + 1)

avgdl = avgdl / N
```

After fitting, the encoder stores `{ N, avgdl, df }`. This is the minimum state needed to compute BM25 scores for any new document or query.

---

## 6. TF-IDF Algorithm

### TF Variants

`sparse-encode` supports three term frequency variants, configurable via the `tfVariant` option.

**Raw TF** (`tfVariant: 'raw'`):
```
TF(t, D) = f(t, D)
```
The raw count of term occurrences. Simple but problematic for long documents: a 10,000-word document naturally has higher raw TF than a 100-word document.

**Log-normalized TF** (`tfVariant: 'log'`, default):
```
TF(t, D) = 1 + log(f(t, D))    if f(t, D) > 0
TF(t, D) = 0                    if f(t, D) = 0
```
Sublinear TF scaling. The logarithm dampens the effect of repeated occurrences, similar in spirit to BM25's saturation but without a configurable saturation parameter. This is the default because it provides the best balance of simplicity and quality.

**Augmented TF** (`tfVariant: 'augmented'`):
```
TF(t, D) = 0.5 + 0.5 * f(t, D) / max_t'(f(t', D))
```
Normalizes term frequency by the maximum term frequency in the document. The result is always in [0.5, 1.0] for terms present in the document. This variant is useful when comparing documents of very different lengths, because the normalization removes length bias.

### IDF Computation

TF-IDF uses a simpler IDF than BM25:

```
IDF(t) = log(1 + N / df(t))
```

The `1 +` provides smoothing, ensuring IDF is always positive. Without it, `log(N / df)` would be 0 when df = N.

For terms with df = 0 (unseen in corpus), IDF is undefined. `sparse-encode` handles this by assigning `IDF = log(1 + N)`, the maximum value, treating unseen terms as maximally informative (consistent with BM25 handling).

### L2 Normalization

After computing the raw TF-IDF scores for all terms in a document, the resulting vector can be L2-normalized:

```
norm = sqrt(sum(value_i^2))
normalized_value_i = value_i / norm
```

L2 normalization ensures the sparse vector has unit length, making it comparable in magnitude to dense embedding vectors (which are typically unit-normalized). This is important for hybrid search where sparse and dense scores are combined: without normalization, sparse scores may dominate or be dominated by dense scores depending on the raw magnitude.

L2 normalization is enabled by default for TF-IDF (`normalize: true`). For BM25, normalization is off by default because BM25 scores are already bounded and calibrated by the algorithm's saturation mechanism.

### TF-IDF Sparse Vector Construction

The process mirrors BM25:
1. Tokenize the document.
2. Count term frequencies.
3. For each unique term: compute `TF(t, D) * IDF(t)`.
4. Map term to index.
5. Optionally L2-normalize.
6. Return `{ indices, values }`.

---

## 7. Sparse Vector Format

### Output Structure

Every encoding method returns a `SparseVector`:

```typescript
interface SparseVector {
  indices: number[];
  values: number[];
}
```

- `indices` is sorted in ascending order.
- `values[i]` corresponds to `indices[i]`.
- All values are positive (zero-valued entries are excluded).
- Indices are in the range `[0, vocabSize)`.
- Both arrays have the same length.
- For an empty document (no terms after tokenization), both arrays are empty.

### Pinecone Compatibility

Pinecone's `upsert` API accepts sparse vectors in exactly this format:

```typescript
await pinecone.index('my-index').upsert([{
  id: 'doc-1',
  values: denseVector,
  sparseValues: {
    indices: sparseVector.indices,
    values: sparseVector.values,
  },
}]);
```

Pinecone's `query` API similarly accepts a sparse vector:

```typescript
const results = await pinecone.index('my-index').query({
  vector: denseQueryVector,
  sparseVector: {
    indices: sparseQueryVector.indices,
    values: sparseQueryVector.values,
  },
  topK: 10,
});
```

No transformation is needed. The `SparseVector` output of `sparse-encode` can be passed directly to the Pinecone client.

### Qdrant Compatibility

Qdrant accepts sparse vectors as named vectors:

```typescript
await qdrant.upsert('my-collection', {
  points: [{
    id: 'doc-1',
    vector: {
      dense: denseVector,
      sparse: {
        indices: sparseVector.indices,
        values: sparseVector.values,
      },
    },
  }],
});
```

The Qdrant client expects the same `{ indices, values }` structure. No transformation needed.

### Milvus Compatibility

Milvus 2.4+ supports sparse vectors as a field type. The SDK accepts sparse vectors as a dictionary of `{ [index: number]: value: number }` or as two parallel arrays. `sparse-encode` provides a convenience method `toMilvusSparse()`:

```typescript
function toMilvusSparse(sv: SparseVector): Record<number, number> {
  const result: Record<number, number> = {};
  for (let i = 0; i < sv.indices.length; i++) {
    result[sv.indices[i]] = sv.values[i];
  }
  return result;
}
```

This is exported as a utility function alongside the core API.

---

## 8. Term-to-Index Mapping

### Hash-Based Mapping (Default)

Hash-based mapping converts each term string to an integer index using a hash function modulo the vocabulary size. No vocabulary file or corpus-specific state is needed for the mapping itself -- the same term always maps to the same index regardless of what corpus was used for fitting.

**Algorithm**:
```
index(term) = MurmurHash3_32(term) mod vocabSize
```

**MurmurHash3** is a non-cryptographic hash function designed for fast, well-distributed hashing of short strings. Properties relevant to sparse vector indexing:
- Extremely fast: ~1 ns per hash for typical term lengths (5-20 characters).
- Good distribution: low collision rate for realistic vocabularies.
- Deterministic: same input always produces same output, across platforms and environments.
- 32-bit output: sufficient for vocabulary sizes up to 2^32.

`sparse-encode` implements MurmurHash3 (32-bit, x86 variant) from scratch in TypeScript -- no dependency needed. The implementation follows the reference specification (Austin Appleby, 2008) and produces identical output to the reference C implementation for all inputs.

**Vocabulary size**: The default is `vocabSize = 262144` (2^18). This matches the default in Pinecone's `pinecone-text`. The choice of 2^18 balances:
- Collision avoidance: with 50 unique terms per document, the expected number of collisions is ~0.005 per document (birthday problem: n^2 / (2 * m) = 50^2 / (2 * 262144) = 0.005). Negligible.
- Memory efficiency: Pinecone, Qdrant, and Milvus allocate sparse vector dimensions on the fly; 262,144 possible dimensions is well within their capacity.
- Compatibility: using the same vocabSize as `pinecone-text` ensures that sparse vectors generated by `sparse-encode` and `pinecone-text` for the same text use the same index space (assuming the same hash function), enabling mixed-language pipelines.

**Collision handling**: When two terms map to the same index, their BM25 or TF-IDF scores are summed. This is a lossy approximation -- the vector database cannot distinguish between the two terms. In practice, collisions are rare enough that retrieval quality is not measurably affected. If exact mapping is required, use vocabulary-based mapping.

### Vocabulary-Based Mapping

Vocabulary-based mapping assigns each unique term a sequential integer ID during `fit()`.

**Construction** (during `fit`):
```
vocab = new Map<string, number>()
nextId = 0
for each document in corpus:
  for each term in tokenize(document):
    if !vocab.has(term):
      vocab.set(term, nextId++)
```

**Encoding** (after fit):
```
index(term) = vocab.get(term)  // returns undefined if term not in vocabulary
```

**Out-of-vocabulary (OOV) handling**: Configurable via `oovStrategy`:
- `'ignore'` (default): Skip OOV terms. They do not appear in the sparse vector.
- `'hash'`: Fall back to hash-based mapping for OOV terms. This provides graceful degradation for new terms not in the training corpus.
- `'error'`: Throw an error. Useful for strict deployments where all terms must be in the vocabulary.

**Advantages**: Zero collisions. The index space is compact (indices 0 through vocab.size - 1). Vocabulary is human-inspectable.

**Disadvantages**: Requires the vocabulary to be serialized with the model. New terms not in the vocabulary need OOV handling. The vocabulary grows with corpus size (English text corpora typically produce 50,000-200,000 unique terms after stemming and stopword removal).

### Configuration

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `indexing` | `'hash' \| 'vocab'` | `'hash'` | Term-to-index mapping strategy. |
| `vocabSize` | `number` | `262144` | Vocabulary size for hash-based mapping. Must be a positive integer. Powers of 2 recommended. |
| `hashSeed` | `number` | `0` | Seed for MurmurHash3. Changing the seed changes the mapping. |
| `oovStrategy` | `'ignore' \| 'hash' \| 'error'` | `'ignore'` | OOV handling for vocabulary-based mapping. |

---

## 9. Tokenization

### Default Tokenizer

The built-in tokenizer is designed for English text. It applies five steps in order:

1. **Lowercase**: `text.toLowerCase()`.
2. **Punctuation normalization**: Replace all characters matching `/[^\p{L}\p{N}\-]/gu` (not a Unicode letter, not a Unicode digit, not a hyphen) with a space. This preserves hyphenated compound terms ("real-time", "state-of-the-art") while removing punctuation.
3. **Split on whitespace**: `text.split(/\s+/).filter(Boolean)`.
4. **Stopword removal**: Filter out tokens present in the built-in English stopword set.
5. **Minimum length**: Filter out tokens with length < 2.

If `stemming: true` is enabled, a sixth step is added after stopword removal:
6. **Porter stemming**: Apply the Porter stemmer algorithm to each token.

### Built-In English Stopwords

The stopword list contains approximately 175 common English function words. These are words that appear in virtually every English document and provide no discriminative value for retrieval. Including them in sparse vectors wastes dimensions and degrades retrieval quality.

The list includes: "a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any", "are", "aren't", "as", "at", "be", "because", "been", "before", "being", "below", "between", "both", "but", "by", "can", "can't", "cannot", "could", "couldn't", "did", "didn't", "do", "does", "doesn't", "doing", "don't", "down", "during", "each", "few", "for", "from", "further", "get", "got", "had", "hadn't", "has", "hasn't", "have", "haven't", "having", "he", "her", "here", "hers", "herself", "him", "himself", "his", "how", "i", "if", "in", "into", "is", "isn't", "it", "its", "itself", "just", "let", "me", "might", "more", "most", "mustn't", "my", "myself", "no", "nor", "not", "of", "off", "on", "once", "only", "or", "other", "ought", "our", "ours", "ourselves", "out", "over", "own", "same", "she", "should", "shouldn't", "so", "some", "such", "than", "that", "the", "their", "theirs", "them", "themselves", "then", "there", "these", "they", "this", "those", "through", "to", "too", "under", "until", "up", "very", "was", "wasn't", "we", "were", "weren't", "what", "when", "where", "which", "while", "who", "whom", "why", "will", "with", "won't", "would", "wouldn't", "you", "your", "yours", "yourself", "yourselves".

### Porter Stemmer

The Porter stemmer (Martin Porter, 1980) is a suffix-stripping algorithm for English. It reduces inflected words to a common stem by applying a series of rules:

- **Step 1**: Remove plural suffixes (-s, -es, -ies) and verb suffixes (-ing, -ed).
- **Step 2**: Reduce double suffixes (-ational to -ate, -fulness to -ful, etc.).
- **Step 3**: Remove derivational suffixes (-alize to -al, -icate to -ic, etc.).
- **Step 4**: Remove further suffixes (-ment, -ence, -able, etc.) in specific contexts.
- **Step 5**: Clean up (-e removal, double consonant reduction).

Examples: "running" -> "run", "connections" -> "connect", "effectively" -> "effect", "computational" -> "comput".

Stemming increases recall (queries for "run" match documents containing "running", "runs", "runner") but may decrease precision ("university" and "universe" both stem to "univers"). For general-purpose search, stemming is beneficial. For domain-specific search with precise terminology (medical, legal, code), stemming should be disabled.

`sparse-encode` implements the Porter stemmer from scratch. No dependency on `natural`, `stemmer`, or other npm packages.

### Custom Tokenizer

The custom tokenizer replaces the entire built-in pipeline. The caller provides a function `(text: string) => string[]` that takes raw text and returns an array of term strings:

```typescript
const encoder = createBM25({
  tokenizer: (text) => {
    // Domain-specific: split camelCase and snake_case, keep digits
    return text
      .replace(/([a-z])([A-Z])/g, '$1 $2')  // camelCase
      .replace(/_/g, ' ')                      // snake_case
      .toLowerCase()
      .split(/\s+/)
      .filter(t => t.length >= 2);
  },
});
```

When a custom tokenizer is provided, the built-in stopword removal and stemming are not applied -- the custom tokenizer is responsible for its own preprocessing. This gives full control for non-English languages, specialized domains, or novel tokenization strategies.

### Configuration

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `tokenizer` | `(text: string) => string[]` | built-in | Custom tokenizer function. Replaces the default pipeline. |
| `stemming` | `boolean` | `false` | Enable Porter stemming in the default tokenizer. Ignored if custom tokenizer is provided. |
| `stopwords` | `string[] \| false` | built-in list | Custom stopword list, or `false` to disable stopword removal. Ignored if custom tokenizer is provided. |
| `additionalStopwords` | `string[]` | `[]` | Additional stopwords to append to the built-in list. Ignored if custom tokenizer or custom stopwords are provided. |
| `minTokenLength` | `number` | `2` | Minimum token length after tokenization. Ignored if custom tokenizer is provided. |

---

## 10. Corpus Fitting

### `encoder.fit(documents)`

Computes IDF statistics from a corpus of documents.

```typescript
const encoder = createBM25();
encoder.fit([
  'the quick brown fox jumps over the lazy dog',
  'a quick brown dog runs in the park',
  'the fox and the dog are friends',
]);
```

**What `fit` computes**:
- `N`: Total number of documents.
- `df`: Document frequency for every term encountered. A `Map<string, number>` where `df.get(term)` is the number of documents containing that term.
- `avgdl`: Average document length in tokens (after tokenization).
- `totalTokens`: Total number of tokens across all documents (for diagnostics).
- `vocab` (vocabulary-based indexing only): A `Map<string, number>` mapping each term to its integer index.

**After `fit`**, the encoder is ready to encode documents and queries. Calling `encode()` or `encodeQuery()` before `fit()` throws an error.

**Fit is not incremental by default**: Calling `fit(docs1)` and then `fit(docs2)` replaces the first fit entirely. The second call does not merge with the first. For incremental fitting, use `partialFit`.

### `encoder.partialFit(documents)`

Incrementally updates the fitted statistics with additional documents without re-processing the entire corpus.

```typescript
encoder.fit(initialCorpus);
// Later, when new documents arrive:
encoder.partialFit(newDocuments);
```

**How partial fit works**:
1. Tokenize each new document.
2. Update `N` by adding the count of new documents.
3. Update `df` by incrementing document frequencies for all terms in the new documents.
4. Update `avgdl` as a weighted average: `avgdl = (avgdl * oldN + newTotalTokens) / newN`.
5. For vocabulary-based indexing: add new terms to the vocabulary with sequential IDs continuing from the last assigned ID.

`partialFit` does not recompute statistics for previously fitted documents. This means that removing documents from the corpus is not supported incrementally -- a full `fit()` is required to reflect document removals.

### Serialization: Save and Load Fitted Models

Fitted models are serialized to JSON for deployment. This avoids re-fitting the corpus every time the application starts.

**Serialize**:
```typescript
const modelJson = encoder.serialize();
// modelJson is a JSON string containing:
// {
//   "type": "bm25" | "tfidf",
//   "version": 1,
//   "params": { "k1": 1.2, "b": 0.75, ... },
//   "corpus": { "N": 10000, "avgdl": 142.5, "df": { "term1": 523, "term2": 12, ... } },
//   "indexing": { "type": "hash", "vocabSize": 262144, "hashSeed": 0 }
//     // or: { "type": "vocab", "vocab": { "term1": 0, "term2": 1, ... } }
//   "tokenizer": "default" | "custom"
// }
fs.writeFileSync('bm25-model.json', modelJson);
```

**Deserialize**:
```typescript
const modelJson = fs.readFileSync('bm25-model.json', 'utf-8');
const encoder = BM25Encoder.deserialize(modelJson);
// encoder is ready to encode -- no fit() needed
```

When `tokenizer: "custom"` is recorded in the serialized model, the deserializer does not restore the tokenizer function (functions are not serializable). The caller must provide the same custom tokenizer when deserializing:

```typescript
const encoder = BM25Encoder.deserialize(modelJson, {
  tokenizer: myCustomTokenizer,
});
```

### Model Size

The serialized model size depends on the corpus vocabulary. For a corpus of 100,000 documents:
- **Hash-based indexing**: The serialized model stores `{ N, avgdl, df }`. The `df` map typically has 50,000-200,000 entries (one per unique term after stemming). Serialized JSON size: approximately 2-8 MB.
- **Vocabulary-based indexing**: Additionally stores the vocabulary map. Adds approximately 1-4 MB. Total: 3-12 MB.

For serverless deployments, the model JSON can be bundled with the application code, stored in a CDN, or fetched from an object store at cold start.

---

## 11. Query vs. Document Encoding

### Why They Differ

When using sparse vectors for hybrid search, the document encoding and query encoding serve different purposes:

- **Document encoding** answers: "How important is each term in this document relative to the corpus?" It uses the full BM25/TF-IDF formula with document length normalization. A long document that mentions "transformer" 50 times gets a saturated (but not maximal) BM25 score for "transformer".

- **Query encoding** answers: "How important is each query term for discriminating among documents?" It uses a simplified formula that weights terms by IDF only, without document length normalization. A query is not a document in the corpus -- it does not have a meaningful "document length" relative to `avgdl`, and applying length normalization to a short query would artificially inflate term scores.

Pinecone's `pinecone-text` makes this distinction explicit, and it is critical for correct hybrid search. Using document encoding for queries (or vice versa) degrades retrieval quality.

### Document Encoding

`encoder.encode(text)` encodes text as a document:

For BM25:
```
For each term t in tokenize(text):
  value = IDF(t) * (f(t, text) * (k1 + 1)) / (f(t, text) + k1 * (1 - b + b * |text| / avgdl))
```

Full BM25 formula with document length normalization.

For TF-IDF:
```
For each term t in tokenize(text):
  value = TF(t, text) * IDF(t)
```

Optionally L2-normalized.

### Query Encoding

`encoder.encodeQuery(query)` encodes text as a query:

For BM25:
```
For each term t in tokenize(query):
  value = IDF(t)
```

Query encoding uses IDF only. Each query term's sparse vector value is its IDF score. Term frequency within the query is not used (a query that repeats a term does not get extra weight). Document length normalization is not applied.

For TF-IDF:
```
For each term t in tokenize(query):
  value = IDF(t)
```

Same approach: IDF-only weighting for query terms.

This means the query sparse vector is a simple IDF-weighted indicator of which terms the query contains. The vector database's similarity computation (dot product between query sparse vector and document sparse vectors) then naturally computes a weighted sum of BM25/TF-IDF scores for the matching terms -- which is exactly BM25 retrieval.

---

## 12. API Surface

### Installation

```bash
npm install sparse-encode
```

### Factory: `createBM25`

Creates a `BM25Encoder` instance.

```typescript
import { createBM25 } from 'sparse-encode';

const encoder = createBM25({
  k1: 1.2,
  b: 0.75,
});

encoder.fit(documents);
const sparseVector = encoder.encode('text to encode');
```

**Signature**:
```typescript
function createBM25(options?: BM25Options): BM25Encoder;
```

### Factory: `createTFIDF`

Creates a `TFIDFEncoder` instance.

```typescript
import { createTFIDF } from 'sparse-encode';

const encoder = createTFIDF({
  tfVariant: 'log',
  normalize: true,
});

encoder.fit(documents);
const sparseVector = encoder.encode('text to encode');
```

**Signature**:
```typescript
function createTFIDF(options?: TFIDFOptions): TFIDFEncoder;
```

### `encoder.fit`

Computes IDF statistics from a corpus. Required before encoding.

```typescript
encoder.fit(documents);
```

**Signature**:
```typescript
fit(documents: string[]): void;
```

Throws `SparseEncodeError` if `documents` is empty.

### `encoder.partialFit`

Incrementally updates fitted statistics with new documents.

```typescript
encoder.partialFit(newDocuments);
```

**Signature**:
```typescript
partialFit(documents: string[]): void;
```

Throws `SparseEncodeError` if the encoder has not been fitted (call `fit()` first).

### `encoder.encode`

Encodes a document as a sparse vector.

```typescript
const sv: SparseVector = encoder.encode('the quick brown fox');
// { indices: [4821, 19203, 51847, 182033], values: [1.23, 0.87, 1.56, 2.01] }
```

**Signature**:
```typescript
encode(text: string): SparseVector;
```

Throws `SparseEncodeError` if the encoder has not been fitted.

### `encoder.encodeBatch`

Encodes multiple documents as sparse vectors.

```typescript
const vectors: SparseVector[] = encoder.encodeBatch([
  'document one text',
  'document two text',
  'document three text',
]);
```

**Signature**:
```typescript
encodeBatch(texts: string[]): SparseVector[];
```

### `encoder.encodeQuery`

Encodes a query as a sparse vector (IDF-only weighting, no length normalization).

```typescript
const querySv: SparseVector = encoder.encodeQuery('quick brown fox');
```

**Signature**:
```typescript
encodeQuery(query: string): SparseVector;
```

### `encoder.serialize`

Serializes the fitted model to a JSON string.

```typescript
const json: string = encoder.serialize();
fs.writeFileSync('model.json', json);
```

**Signature**:
```typescript
serialize(): string;
```

Throws `SparseEncodeError` if the encoder has not been fitted.

### `BM25Encoder.deserialize` / `TFIDFEncoder.deserialize`

Restores an encoder from a serialized JSON string.

```typescript
const json = fs.readFileSync('model.json', 'utf-8');
const encoder = BM25Encoder.deserialize(json);
// or
const encoder = BM25Encoder.deserialize(json, { tokenizer: customTokenizer });
```

**Signature**:
```typescript
static deserialize(json: string, options?: { tokenizer?: TokenizerFn }): BM25Encoder;
static deserialize(json: string, options?: { tokenizer?: TokenizerFn }): TFIDFEncoder;
```

### `encoder.getStats`

Returns statistics about the fitted corpus.

```typescript
const stats: FitStats = encoder.getStats();
// { N: 10000, avgdl: 142.5, vocabSize: 87432, totalTokens: 1425000 }
```

**Signature**:
```typescript
getStats(): FitStats;
```

### Utility: `toMilvusSparse`

Converts a `SparseVector` to Milvus's `Record<number, number>` format.

```typescript
import { toMilvusSparse } from 'sparse-encode';

const milvusSparse = toMilvusSparse(sparseVector);
// { 4821: 1.23, 19203: 0.87, 51847: 1.56, 182033: 2.01 }
```

**Signature**:
```typescript
function toMilvusSparse(sv: SparseVector): Record<number, number>;
```

### Type Definitions

```typescript
// ── Core Output ───────────────────────────────────────────────────────

/** Sparse vector in the format expected by Pinecone, Qdrant, and Milvus. */
interface SparseVector {
  /** Sorted ascending integer indices in [0, vocabSize). */
  indices: number[];
  /** Values corresponding to each index. Same length as indices. All positive. */
  values: number[];
}

// ── Tokenizer ─────────────────────────────────────────────────────────

/** Custom tokenizer function. Takes raw text, returns array of term strings. */
type TokenizerFn = (text: string) => string[];

// ── BM25 Options ──────────────────────────────────────────────────────

interface BM25Options {
  /** Term frequency saturation parameter. Default: 1.2. Range: [0, 3]. */
  k1?: number;
  /** Document length normalization parameter. Default: 0.75. Range: [0, 1]. */
  b?: number;
  /** Term-to-index mapping strategy. Default: 'hash'. */
  indexing?: 'hash' | 'vocab';
  /** Vocabulary size for hash-based mapping. Default: 262144 (2^18). */
  vocabSize?: number;
  /** MurmurHash3 seed. Default: 0. */
  hashSeed?: number;
  /** OOV strategy for vocabulary-based mapping. Default: 'ignore'. */
  oovStrategy?: 'ignore' | 'hash' | 'error';
  /** Custom tokenizer function. Overrides built-in tokenizer. */
  tokenizer?: TokenizerFn;
  /** Enable Porter stemming in default tokenizer. Default: false. */
  stemming?: boolean;
  /** Custom stopword list, or false to disable. Default: built-in English list. */
  stopwords?: string[] | false;
  /** Additional stopwords to append to built-in list. Default: []. */
  additionalStopwords?: string[];
  /** Minimum token length. Default: 2. */
  minTokenLength?: number;
}

// ── TF-IDF Options ────────────────────────────────────────────────────

interface TFIDFOptions {
  /** Term frequency variant. Default: 'log'. */
  tfVariant?: 'raw' | 'log' | 'augmented';
  /** L2-normalize the output vector. Default: true. */
  normalize?: boolean;
  /** Term-to-index mapping strategy. Default: 'hash'. */
  indexing?: 'hash' | 'vocab';
  /** Vocabulary size for hash-based mapping. Default: 262144 (2^18). */
  vocabSize?: number;
  /** MurmurHash3 seed. Default: 0. */
  hashSeed?: number;
  /** OOV strategy for vocabulary-based mapping. Default: 'ignore'. */
  oovStrategy?: 'ignore' | 'hash' | 'error';
  /** Custom tokenizer function. Overrides built-in tokenizer. */
  tokenizer?: TokenizerFn;
  /** Enable Porter stemming in default tokenizer. Default: false. */
  stemming?: boolean;
  /** Custom stopword list, or false to disable. Default: built-in English list. */
  stopwords?: string[] | false;
  /** Additional stopwords to append to built-in list. Default: []. */
  additionalStopwords?: string[];
  /** Minimum token length. Default: 2. */
  minTokenLength?: number;
}

// ── Fit Statistics ────────────────────────────────────────────────────

interface FitStats {
  /** Total documents in the fitted corpus. */
  N: number;
  /** Average document length in tokens. */
  avgdl: number;
  /** Number of unique terms in the vocabulary. */
  vocabSize: number;
  /** Total tokens processed across all documents. */
  totalTokens: number;
}

// ── Encoder Interfaces ────────────────────────────────────────────────

interface BM25Encoder {
  fit(documents: string[]): void;
  partialFit(documents: string[]): void;
  encode(text: string): SparseVector;
  encodeBatch(texts: string[]): SparseVector[];
  encodeQuery(query: string): SparseVector;
  serialize(): string;
  getStats(): FitStats;
}

interface TFIDFEncoder {
  fit(documents: string[]): void;
  partialFit(documents: string[]): void;
  encode(text: string): SparseVector;
  encodeBatch(texts: string[]): SparseVector[];
  encodeQuery(query: string): SparseVector;
  serialize(): string;
  getStats(): FitStats;
}

// ── Error ─────────────────────────────────────────────────────────────

class SparseEncodeError extends Error {
  code: 'NOT_FITTED' | 'INVALID_OPTIONS' | 'EMPTY_CORPUS' | 'SERIALIZATION_ERROR';
}
```

---

## 13. Configuration

### All Options with Defaults

#### BM25 Configuration

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `k1` | `number` | `1.2` | Term frequency saturation. |
| `b` | `number` | `0.75` | Document length normalization. |
| `indexing` | `'hash' \| 'vocab'` | `'hash'` | Term-to-index mapping strategy. |
| `vocabSize` | `number` | `262144` | Hash-based vocabulary size. |
| `hashSeed` | `number` | `0` | MurmurHash3 seed. |
| `oovStrategy` | `'ignore' \| 'hash' \| 'error'` | `'ignore'` | OOV handling for vocabulary-based mapping. |
| `tokenizer` | `TokenizerFn` | built-in | Custom tokenizer. |
| `stemming` | `boolean` | `false` | Enable Porter stemming. |
| `stopwords` | `string[] \| false` | built-in | Stopword list or false. |
| `additionalStopwords` | `string[]` | `[]` | Extra stopwords. |
| `minTokenLength` | `number` | `2` | Minimum token length. |

#### TF-IDF Configuration

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `tfVariant` | `'raw' \| 'log' \| 'augmented'` | `'log'` | Term frequency variant. |
| `normalize` | `boolean` | `true` | L2-normalize output vectors. |
| `indexing` | `'hash' \| 'vocab'` | `'hash'` | Term-to-index mapping strategy. |
| `vocabSize` | `number` | `262144` | Hash-based vocabulary size. |
| `hashSeed` | `number` | `0` | MurmurHash3 seed. |
| `oovStrategy` | `'ignore' \| 'hash' \| 'error'` | `'ignore'` | OOV handling for vocabulary-based mapping. |
| `tokenizer` | `TokenizerFn` | built-in | Custom tokenizer. |
| `stemming` | `boolean` | `false` | Enable Porter stemming. |
| `stopwords` | `string[] \| false` | built-in | Stopword list or false. |
| `additionalStopwords` | `string[]` | `[]` | Extra stopwords. |
| `minTokenLength` | `number` | `2` | Minimum token length. |

---

## 14. CLI

### Installation and Invocation

```bash
# Global install
npm install -g sparse-encode
sparse-encode fit --input corpus.jsonl --output model.json

# npx (no install)
npx sparse-encode encode --model model.json --text "query text"

# Package script
# package.json: { "scripts": { "fit": "sparse-encode fit --input corpus.jsonl --output model.json" } }
npm run fit
```

### CLI Binary Name

`sparse-encode`

### Commands

#### `sparse-encode fit`

Fits an encoder on a corpus and saves the model.

```
sparse-encode fit [options]

Options:
  --input <path>        Path to corpus file. JSONL (one text per line) or JSON array of strings. Required.
  --output <path>       Path to write the fitted model JSON. Required.
  --algorithm <alg>     Scoring algorithm: bm25 (default) | tfidf.
  --k1 <n>              BM25 k1 parameter. Default: 1.2.
  --b <n>               BM25 b parameter. Default: 0.75.
  --tf-variant <v>      TF-IDF TF variant: raw | log | augmented. Default: log.
  --indexing <type>     Index mapping: hash (default) | vocab.
  --vocab-size <n>      Hash vocabulary size. Default: 262144.
  --stemming            Enable Porter stemming.
  --format <fmt>        Output format: human (default) | json.
```

**Output**:
```
$ sparse-encode fit --input corpus.jsonl --output bm25-model.json

  sparse-encode v0.1.0

  Algorithm:    BM25 (k1=1.2, b=0.75)
  Indexing:     hash (vocabSize=262144)
  Stemming:     off
  Corpus:       10,000 documents
  Vocabulary:   87,432 unique terms
  Avg doc length: 142.5 tokens
  Total tokens: 1,425,000

  Model saved to: bm25-model.json (3.2 MB)
```

#### `sparse-encode encode`

Encodes one or more texts as document sparse vectors using a fitted model.

```
sparse-encode encode [options]

Options:
  --model <path>        Path to fitted model JSON. Required.
  --text <text>         Single text to encode. Mutually exclusive with --input.
  --input <path>        Path to file with texts (one per line or JSONL). Mutually exclusive with --text.
  --output <path>       Path to write output. Default: stdout.
  --format <fmt>        Output format: json (default) | jsonl.
```

**Output** (single text):
```json
{
  "indices": [4821, 19203, 51847, 182033],
  "values": [1.23, 0.87, 1.56, 2.01]
}
```

#### `sparse-encode query`

Encodes a query as a sparse vector using query-specific encoding.

```
sparse-encode query [options]

Options:
  --model <path>        Path to fitted model JSON. Required.
  --text <text>         Query text to encode. Required.
  --format <fmt>        Output format: json (default) | human.
```

#### `sparse-encode stats`

Prints statistics about a fitted model.

```
sparse-encode stats [options]

Options:
  --model <path>        Path to fitted model JSON. Required.
  --format <fmt>        Output format: human (default) | json.
```

### Exit Codes

| Code | Meaning |
|------|---------|
| `0` | Success. |
| `1` | Operation failed (IO error, invalid model file). |
| `2` | Configuration error (missing required options, invalid flags). |

---

## 15. Integration

### Integration with Pinecone

Complete hybrid search pipeline using `sparse-encode` and the Pinecone client:

```typescript
import { createBM25 } from 'sparse-encode';
import { createCache } from 'embed-cache';
import { Pinecone } from '@pinecone-database/pinecone';

// 1. Set up encoders
const bm25 = createBM25();
bm25.fit(allDocumentTexts);

const embedCache = createCache({
  model: 'text-embedding-3-small',
  embedder: openaiEmbedder,
});

const pinecone = new Pinecone();
const index = pinecone.index('hybrid-search');

// 2. Index documents (dense + sparse)
for (const doc of documents) {
  const denseVector = await embedCache.embed(doc.text);
  const sparseVector = bm25.encode(doc.text);

  await index.upsert([{
    id: doc.id,
    values: denseVector,
    sparseValues: sparseVector,
    metadata: { title: doc.title },
  }]);
}

// 3. Query (dense + sparse)
async function search(query: string) {
  const denseQuery = await embedCache.embed(query);
  const sparseQuery = bm25.encodeQuery(query);

  return index.query({
    vector: denseQuery,
    sparseVector: sparseQuery,
    topK: 10,
    includeMetadata: true,
  });
}
```

### Integration with Qdrant

```typescript
import { createBM25 } from 'sparse-encode';
import { QdrantClient } from '@qdrant/js-client-rest';

const bm25 = createBM25();
bm25.fit(corpus);

const qdrant = new QdrantClient({ url: 'http://localhost:6333' });

// Index with named sparse vector
await qdrant.upsert('my-collection', {
  points: documents.map(doc => ({
    id: doc.id,
    vector: {
      dense: denseVector,
      sparse: bm25.encode(doc.text),
    },
  })),
});

// Query with hybrid search
const results = await qdrant.query('my-collection', {
  query: {
    fusion: 'rrf',  // Reciprocal Rank Fusion
  },
  prefetch: [
    { query: denseQueryVector, using: 'dense', limit: 20 },
    { query: bm25.encodeQuery(queryText), using: 'sparse', limit: 20 },
  ],
  limit: 10,
});
```

### Integration with Milvus

```typescript
import { createBM25, toMilvusSparse } from 'sparse-encode';
import { MilvusClient } from '@zilliz/milvus2-sdk-node';

const bm25 = createBM25();
bm25.fit(corpus);

const milvus = new MilvusClient({ address: 'localhost:19530' });

// Insert with sparse field
await milvus.insert({
  collection_name: 'hybrid_search',
  data: documents.map(doc => ({
    id: doc.id,
    dense_vector: denseVector,
    sparse_vector: toMilvusSparse(bm25.encode(doc.text)),
    text: doc.text,
  })),
});
```

### Integration with `embed-cache`

`embed-cache` provides dense embedding caching; `sparse-encode` provides sparse vector generation. Together they form the complete encoding layer for hybrid search:

```typescript
import { createBM25 } from 'sparse-encode';
import { createCache } from 'embed-cache';

const bm25 = createBM25();
bm25.fit(corpus);

const embedCache = createCache({
  model: 'text-embedding-3-small',
  embedder: openaiEmbedder,
  storage: { type: 'sqlite', path: './embed-cache.db' },
});

async function encodeDocument(text: string) {
  const [dense, sparse] = await Promise.all([
    embedCache.embed(text),
    Promise.resolve(bm25.encode(text)), // sync, wrapped for Promise.all
  ]);
  return { dense, sparse };
}
```

### Integration with `fusion-rank`

`fusion-rank` provides reciprocal rank fusion and other score combination methods. When implementing hybrid search at the application level (rather than relying on the vector database's built-in fusion), `fusion-rank` combines the dense and sparse retrieval results:

```typescript
import { createBM25 } from 'sparse-encode';
import { fuse } from 'fusion-rank';

const denseResults = await vectorDb.searchDense(denseQueryVector, { topK: 50 });
const sparseResults = await vectorDb.searchSparse(bm25.encodeQuery(query), { topK: 50 });

const fusedResults = fuse([denseResults, sparseResults], {
  method: 'rrf',
  k: 60,
});
```

---

## 16. Testing Strategy

### Unit Tests

Each algorithm component has isolated unit tests with deterministic inputs and analytically verifiable outputs.

**BM25 scoring tests:**
- Verify BM25 score for a single term with known f, df, N, avgdl, |D|, k1, b matches the hand-computed value.
- Verify BM25 score is 0 when f = 0 (term not in document).
- Verify BM25 score increases with term frequency (f = 1 < f = 2 < f = 5), but with diminishing returns.
- Verify BM25 score decreases for longer documents (|D| > avgdl) and increases for shorter documents (|D| < avgdl) when b > 0.
- Verify BM25 score is independent of document length when b = 0.
- Verify IDF decreases as df increases.
- Verify IDF is always positive (never negative).

**TF-IDF scoring tests:**
- Verify raw TF variant returns the raw count.
- Verify log-normalized TF returns `1 + log(f)`.
- Verify augmented TF returns `0.5 + 0.5 * f / max_f`.
- Verify L2 normalization produces a unit-length vector.
- Verify L2 normalization can be disabled.

**Tokenizer tests:**
- Verify lowercase conversion.
- Verify punctuation removal.
- Verify stopword removal.
- Verify minimum length filtering.
- Verify Porter stemming produces correct stems for known test cases.
- Verify custom tokenizer replaces the default pipeline entirely.

**MurmurHash3 tests:**
- Verify hash output matches the reference C implementation for a set of known test vectors.
- Verify determinism: same input always produces same output.
- Verify distribution: hashing 10,000 unique English words into 262,144 buckets produces fewer than 5 collisions (probabilistic, run multiple seeds).

**Term-to-index mapping tests:**
- Hash-based: verify index = MurmurHash3(term) mod vocabSize.
- Hash collision handling: verify scores are summed when two terms map to the same index.
- Vocabulary-based: verify sequential ID assignment.
- OOV handling: verify 'ignore' skips, 'hash' falls back, 'error' throws.

**Fitting tests:**
- Verify N, avgdl, df are computed correctly for a small corpus with known values.
- Verify partialFit updates N, avgdl, df correctly.
- Verify encoding before fitting throws.
- Verify fitting with empty corpus throws.

**Serialization tests:**
- Verify round-trip: `deserialize(serialize())` produces an encoder that gives identical output.
- Verify deserialized encoder produces the same sparse vectors as the original.
- Verify deserialization with custom tokenizer works.
- Verify deserialization of corrupted JSON throws a meaningful error.

**Query encoding tests:**
- Verify query encoding uses IDF-only weighting (no TF, no length normalization).
- Verify query encoding and document encoding for the same text produce different sparse vectors.
- Verify repeated terms in a query do not increase the score.

**Sparse vector format tests:**
- Verify indices are sorted ascending.
- Verify all values are positive.
- Verify indices are in range [0, vocabSize).
- Verify indices and values have the same length.
- Verify empty text produces empty sparse vector.

### Integration Tests

- **End-to-end BM25**: Fit on a 100-document corpus, encode a document, verify the top-scoring terms are the most distinctive terms in that document (high IDF, moderate TF).
- **End-to-end TF-IDF**: Same test with TF-IDF encoder.
- **Pinecone format compatibility**: Encode a document, verify the output passes Pinecone client's type validation for `sparseValues`.
- **Batch encoding**: Verify `encodeBatch(texts)` returns the same results as encoding each text individually.
- **Cross-language parity**: For a set of test inputs, verify that `sparse-encode` produces the same indices (not necessarily the same values, since BM25 parameters may differ) as Pinecone's `pinecone-text` Python library with the same hash function and vocabSize.

### Performance Tests

A benchmark script (`src/__benchmarks__/encode-throughput.ts`) measures:
- **Fit throughput**: Time to fit on 10,000 documents of ~500 tokens each. Target: under 2 seconds.
- **Encode throughput**: Time to encode 10,000 documents after fitting. Target: under 1 second (> 10,000 documents/second).
- **Query encode throughput**: Time to encode 10,000 queries. Target: under 500ms (> 20,000 queries/second).
- **Serialization roundtrip**: Time to serialize and deserialize a model fitted on 100,000 documents. Target: under 500ms.

---

## 17. Performance

### Fit Latency

Fitting processes each document by tokenizing and counting term frequencies. For a corpus of 100,000 documents with an average of 142 tokens per document:
- Tokenization: ~10us per document (lowercase, split, filter) = ~1 second total.
- Document frequency counting: ~5us per document = ~0.5 seconds.
- Total fit time: approximately 1-3 seconds.

The `df` map grows with the number of unique terms. English corpora typically produce 50,000-200,000 unique terms (after stemming). The `Map` overhead is approximately 50-100 bytes per entry, so 200,000 entries use approximately 10-20 MB of memory.

### Encode Latency

Encoding a single document:
1. Tokenization: ~10us.
2. Term frequency counting: ~5us.
3. BM25/TF-IDF score computation (per unique term): ~0.1us per term, ~50 terms = ~5us.
4. MurmurHash3 (per unique term): ~0.05us per term, ~50 terms = ~2.5us.
5. Sort indices: ~2us.
6. Total per document: approximately 25-50us.

Throughput: approximately 20,000-40,000 documents per second on a single thread (2024 M3 MacBook Pro). This is fast enough for real-time encoding of individual documents and queries, and for batch indexing tens of thousands of documents in seconds.

### Query Encode Latency

Query encoding is faster than document encoding because it skips term frequency computation and length normalization. A typical query (5-15 tokens after stopword removal):
- Tokenization: ~5us.
- IDF lookup: ~1us.
- Hashing: ~0.5us.
- Total: approximately 10-15us per query.

This is fast enough for real-time search (sub-millisecond query encoding).

### Memory Footprint

| Component | Memory |
|-----------|--------|
| Fitted model (100K docs, 100K unique terms) | ~15-25 MB |
| Fitted model (10K docs, 50K unique terms) | ~5-10 MB |
| Single sparse vector (50 non-zero entries) | ~0.8 KB |
| Serialized model JSON (100K unique terms) | ~3-8 MB on disk |

The fitted model is the dominant memory cost. It scales with the number of unique terms in the corpus, not the number of documents. Stemming significantly reduces the number of unique terms (typically by 30-50%).

---

## 18. Dependencies

### Runtime Dependencies

**Zero mandatory runtime dependencies.** All functionality is implemented in pure TypeScript using Node.js built-ins:

- `node:crypto` is NOT required -- MurmurHash3 is implemented in-package. No cryptographic hashing is used.
- Tokenization, stemming, stopword filtering, BM25 scoring, TF-IDF scoring, and sparse vector construction are all self-contained.

This means `sparse-encode` works in any JavaScript runtime: Node.js 18+, Deno, Bun, Cloudflare Workers, and modern browsers (with a bundler).

### Dev Dependencies

| Package | Purpose |
|---------|---------|
| `typescript` | TypeScript compiler |
| `vitest` | Test runner |
| `eslint` | Linting |
| `@types/node` | Node.js type definitions |

---

## 19. File Structure

```
sparse-encode/
├── package.json
├── tsconfig.json
├── SPEC.md
├── README.md
├── src/
│   ├── index.ts                    # Public API exports
│   ├── bm25.ts                     # BM25Encoder class, fit/encode/encodeQuery
│   ├── tfidf.ts                    # TFIDFEncoder class, fit/encode/encodeQuery
│   ├── tokenizer/
│   │   ├── tokenizer.ts            # Default tokenizer pipeline
│   │   ├── stopwords.ts            # Built-in English stopword list
│   │   └── stemmer.ts              # Porter stemmer implementation
│   ├── indexing/
│   │   ├── hash.ts                 # MurmurHash3 implementation and hash-based mapping
│   │   └── vocab.ts                # Vocabulary-based mapping
│   ├── scoring/
│   │   ├── bm25-score.ts           # BM25 per-term scoring function
│   │   ├── tfidf-score.ts          # TF-IDF per-term scoring function
│   │   └── idf.ts                  # IDF computation (BM25 variant and TF-IDF variant)
│   ├── sparse-vector.ts            # SparseVector construction, sorting, normalization
│   ├── serialization.ts            # serialize/deserialize fitted models
│   ├── utils.ts                    # toMilvusSparse and other utilities
│   ├── errors.ts                   # SparseEncodeError class
│   ├── types.ts                    # All TypeScript type definitions
│   └── cli.ts                      # CLI entry point (fit, encode, query, stats)
├── src/__tests__/
│   ├── bm25.test.ts                # BM25 encoder unit tests
│   ├── tfidf.test.ts               # TF-IDF encoder unit tests
│   ├── tokenizer.test.ts           # Tokenizer pipeline tests
│   ├── stemmer.test.ts             # Porter stemmer tests
│   ├── murmurhash.test.ts          # MurmurHash3 reference vector tests
│   ├── indexing.test.ts            # Hash and vocab mapping tests
│   ├── scoring.test.ts             # BM25 and TF-IDF scoring function tests
│   ├── serialization.test.ts       # Serialize/deserialize round-trip tests
│   ├── sparse-vector.test.ts       # Sparse vector format validation tests
│   ├── query-encoding.test.ts      # Query vs document encoding tests
│   ├── integration/
│   │   ├── bm25-e2e.test.ts        # End-to-end BM25 pipeline test
│   │   ├── tfidf-e2e.test.ts       # End-to-end TF-IDF pipeline test
│   │   └── pinecone-compat.test.ts # Pinecone format compatibility test
│   └── fixtures/
│       ├── corpus-small.json       # 50-document test corpus
│       └── reference-vectors.json  # Known-good sparse vectors for regression testing
├── src/__benchmarks__/
│   └── encode-throughput.ts        # Performance benchmarks
└── dist/                           # Build output (gitignored)
    ├── index.js
    ├── index.d.ts
    └── ...
```

---

## 20. Implementation Roadmap

### Phase 1: Core BM25 (v0.1.0)

- Implement MurmurHash3 (32-bit x86) with reference vector tests.
- Implement default tokenizer: lowercase, punctuation removal, whitespace splitting, stopword removal, minimum length filtering.
- Implement `bm25.ts`: `BM25Encoder` class with `fit()`, `encode()`, `encodeQuery()`.
- Implement `idf.ts`: BM25 IDF variant.
- Implement `bm25-score.ts`: per-term BM25 scoring.
- Implement hash-based term-to-index mapping.
- Implement `sparse-vector.ts`: construction and sorting.
- Wire up `createBM25()` factory and `index.ts` exports.
- Write unit tests for all Phase 1 components.
- Write `types.ts` with all public type definitions.

### Phase 2: TF-IDF and Tokenization Options (v0.2.0)

- Implement `tfidf.ts`: `TFIDFEncoder` class with `fit()`, `encode()`, `encodeQuery()`.
- Implement TF variants: raw, log-normalized, augmented.
- Implement L2 normalization.
- Implement Porter stemmer (`stemmer.ts`).
- Add `stemming`, `stopwords`, `additionalStopwords`, `minTokenLength` options.
- Add custom tokenizer support.
- Write unit tests for TF-IDF and tokenization options.

### Phase 3: Vocabulary Mapping and Serialization (v0.3.0)

- Implement vocabulary-based term-to-index mapping.
- Implement OOV strategies: ignore, hash, error.
- Implement `serialization.ts`: `serialize()` and `deserialize()` for both BM25 and TF-IDF.
- Implement `partialFit()` for incremental corpus updates.
- Implement `encodeBatch()`.
- Write serialization round-trip tests.
- Write vocabulary mapping tests.

### Phase 4: CLI (v0.4.0)

- Implement `cli.ts`: `fit`, `encode`, `query`, `stats` commands.
- Add CLI binary to `package.json` (`"bin": { "sparse-encode": "dist/cli.js" }`).
- Write CLI integration tests (spawn as subprocess, verify output and exit codes).
- Implement `toMilvusSparse()` utility.

### Phase 5: Polish and Integration (v0.5.0)

- Write integration tests: end-to-end BM25, end-to-end TF-IDF, Pinecone format compatibility.
- Write performance benchmarks.
- Document integration patterns with Pinecone, Qdrant, Milvus, `embed-cache`, `fusion-rank`.
- Write `README.md` with quickstart, examples, and API reference.
- Publish v0.5.0 to npm.

---

## 21. Example Use Cases

### Example 1: Pinecone Hybrid Search Pipeline

A RAG pipeline indexes 50,000 knowledge base articles into Pinecone with both dense and sparse vectors for hybrid retrieval.

```typescript
import { createBM25 } from 'sparse-encode';
import { createCache } from 'embed-cache';
import { Pinecone } from '@pinecone-database/pinecone';
import fs from 'node:fs';

// Load or fit the BM25 model
let bm25: BM25Encoder;
if (fs.existsSync('./bm25-model.json')) {
  bm25 = BM25Encoder.deserialize(fs.readFileSync('./bm25-model.json', 'utf-8'));
} else {
  bm25 = createBM25({ stemming: true });
  bm25.fit(articles.map(a => a.text));
  fs.writeFileSync('./bm25-model.json', bm25.serialize());
}

const embedCache = createCache({
  model: 'text-embedding-3-small',
  embedder: openaiEmbedder,
  storage: { type: 'sqlite', path: './embed-cache.db' },
});

const pinecone = new Pinecone();
const index = pinecone.index('knowledge-base');

// Index all articles
const batchSize = 100;
for (let i = 0; i < articles.length; i += batchSize) {
  const batch = articles.slice(i, i + batchSize);
  const denseVectors = await embedCache.embedBatch(batch.map(a => a.text));

  await index.upsert(batch.map((article, j) => ({
    id: article.id,
    values: denseVectors[j],
    sparseValues: bm25.encode(article.text),
    metadata: { title: article.title, category: article.category },
  })));
}

// Search
async function search(query: string, topK = 10) {
  const [denseQuery, sparseQuery] = await Promise.all([
    embedCache.embed(query),
    Promise.resolve(bm25.encodeQuery(query)),
  ]);

  return index.query({
    vector: denseQuery,
    sparseVector: sparseQuery,
    topK,
    includeMetadata: true,
  });
}

// "CUDA 12.4 installation error" finds articles about CUDA via sparse match
// even if dense embedding underweights version numbers
const results = await search('CUDA 12.4 installation error');
```

The sparse vector ensures that the exact terms "CUDA", "12.4", "installation", and "error" contribute to the retrieval score. Dense embeddings might find articles about "GPU setup" or "driver problems" (semantically related), but sparse vectors find the article that specifically mentions "CUDA 12.4" -- which is what the user wants.

### Example 2: Serverless Query Encoding

A Cloudflare Worker handles search queries. The BM25 model is loaded from R2 object storage at cold start. No Python, no external service.

```typescript
import { BM25Encoder } from 'sparse-encode';

let encoder: BM25Encoder | null = null;

export default {
  async fetch(request: Request, env: Env): Promise<Response> {
    // Lazy-load model from R2 on first request
    if (!encoder) {
      const modelObj = await env.MODELS_BUCKET.get('bm25-model.json');
      const modelJson = await modelObj!.text();
      encoder = BM25Encoder.deserialize(modelJson);
    }

    const url = new URL(request.url);
    const query = url.searchParams.get('q');
    if (!query) return new Response('Missing q parameter', { status: 400 });

    const sparseVector = encoder.encodeQuery(query);

    return Response.json(sparseVector);
    // { "indices": [4821, 19203, 51847], "values": [2.31, 1.87, 3.12] }
  },
};
```

The entire model (3-8 MB JSON) loads in under 100ms. Subsequent queries encode in under 0.05ms. No API calls, no Python, no cold-start embedding latency.

### Example 3: Domain-Specific Code Search

A code search tool generates sparse vectors for source code files, using a custom tokenizer that understands camelCase, snake_case, and programming identifiers.

```typescript
import { createBM25 } from 'sparse-encode';

const codeEncoder = createBM25({
  k1: 1.5,   // higher saturation for code (repeated identifiers are more significant)
  b: 0.5,    // less length normalization (file length varies widely)
  tokenizer: (text) => {
    return text
      // Split camelCase: "handleUserRequest" -> ["handle", "User", "Request"]
      .replace(/([a-z])([A-Z])/g, '$1 $2')
      // Split snake_case: "handle_user_request" -> ["handle", "user", "request"]
      .replace(/_/g, ' ')
      // Split on non-alphanumeric
      .split(/[^a-zA-Z0-9]+/)
      .map(t => t.toLowerCase())
      .filter(t => t.length >= 2);
  },
});

codeEncoder.fit(allSourceFiles.map(f => f.content));

// Index
const sparseVectors = codeEncoder.encodeBatch(allSourceFiles.map(f => f.content));

// Query
const queryVector = codeEncoder.encodeQuery('handleUserRequest timeout');
```

### Example 4: RAG Pipeline with Score Fusion

A pipeline that performs dense retrieval, sparse retrieval, and fuses the results using `fusion-rank`:

```typescript
import { createBM25 } from 'sparse-encode';
import { createCache } from 'embed-cache';
import { fuse } from 'fusion-rank';

const bm25 = createBM25();
bm25.fit(corpus);

const embedCache = createCache({
  model: 'text-embedding-3-small',
  embedder: openaiEmbedder,
});

async function hybridSearch(query: string, topK = 10) {
  const denseQuery = await embedCache.embed(query);
  const sparseQuery = bm25.encodeQuery(query);

  // Retrieve from both indexes
  const [denseResults, sparseResults] = await Promise.all([
    vectorDb.searchDense(denseQuery, { topK: 50 }),
    vectorDb.searchSparse(sparseQuery, { topK: 50 }),
  ]);

  // Fuse results using Reciprocal Rank Fusion
  const fused = fuse([
    denseResults.map(r => ({ id: r.id, score: r.score })),
    sparseResults.map(r => ({ id: r.id, score: r.score })),
  ], { method: 'rrf', k: 60 });

  return fused.slice(0, topK);
}
```

### Example 5: Incremental Corpus Updates

A pipeline that handles new documents without re-fitting the entire corpus:

```typescript
import { createBM25, BM25Encoder } from 'sparse-encode';
import fs from 'node:fs';

// Load existing model
const modelPath = './bm25-model.json';
let encoder: BM25Encoder;

if (fs.existsSync(modelPath)) {
  encoder = BM25Encoder.deserialize(fs.readFileSync(modelPath, 'utf-8'));
} else {
  encoder = createBM25();
  encoder.fit(initialCorpus);
}

// New documents arrive
const newDocs = fetchNewDocuments();
encoder.partialFit(newDocs);

// Encode new documents
const newVectors = encoder.encodeBatch(newDocs);

// Save updated model
fs.writeFileSync(modelPath, encoder.serialize());

console.log(encoder.getStats());
// { N: 10500, avgdl: 143.2, vocabSize: 89102, totalTokens: 1503600 }
```
