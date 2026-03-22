# sparse-encode

BM25 and TF-IDF sparse vector generation for hybrid search with Pinecone, Qdrant, and Milvus.

[![npm version](https://img.shields.io/npm/v/sparse-encode.svg)](https://www.npmjs.com/package/sparse-encode)
[![npm downloads](https://img.shields.io/npm/dt/sparse-encode.svg)](https://www.npmjs.com/package/sparse-encode)
[![license](https://img.shields.io/npm/l/sparse-encode.svg)](https://github.com/SiluPanda/sparse-encode/blob/master/LICENSE)
[![node](https://img.shields.io/node/v/sparse-encode.svg)](https://nodejs.org)

---

## Description

`sparse-encode` is a zero-dependency JavaScript library that generates BM25 and TF-IDF sparse vectors in the `{ indices: number[], values: number[] }` format expected by Pinecone, Qdrant, and Milvus. It fills the gap left by Python-only tools like `pinecone-text` and `scikit-learn`, giving JavaScript and TypeScript teams a native way to produce sparse vectors for hybrid search without calling out to a Python service.

Hybrid search -- combining dense embeddings (semantic similarity) with sparse vectors (keyword matching) -- consistently outperforms either approach alone, particularly for domain-specific terminology, proper nouns, code identifiers, and long-tail queries. Every major vector database supports hybrid queries that accept both a dense vector and a sparse vector. `sparse-encode` provides the sparse half of that equation.

The library implements Okapi BM25 and TF-IDF scoring from scratch, with configurable tokenization, built-in English stopwords, optional Porter stemming, MurmurHash3-based and vocabulary-based term-to-index mapping, corpus fitting for IDF statistics, model serialization, and separate document/query encoding paths. Output is directly compatible with Pinecone, Qdrant, and Milvus sparse vector fields.

---

## Installation

```bash
npm install sparse-encode
```

Requires Node.js 18 or later. Zero runtime dependencies.

---

## Quick Start

### BM25 Encoding

```typescript
import { createBM25 } from 'sparse-encode';

const encoder = createBM25();

// Fit on your document corpus
encoder.fit([
  'the quick brown fox jumps over the lazy dog',
  'a quick brown dog runs in the park',
  'the fox and the dog are friends',
]);

// Encode a document as a sparse vector
const docVector = encoder.encode('quick brown fox');
// { indices: [4821, 19203, 51847], values: [1.23, 0.87, 1.56] }

// Encode a query (IDF-only weighting, no length normalization)
const queryVector = encoder.encodeQuery('quick brown fox');
```

### TF-IDF Encoding

```typescript
import { createTFIDF } from 'sparse-encode';

const encoder = createTFIDF({ tfVariant: 'log', normalize: true });

encoder.fit(documents);

const sparseVector = encoder.encode('text to encode');
```

---

## Features

- **BM25 (Okapi BM25)** -- Full implementation with configurable k1 (term frequency saturation) and b (document length normalization) parameters. Matches the scoring used by Elasticsearch, Solr, and Lucene.
- **TF-IDF** -- Three term frequency variants: raw, log-normalized (default), and augmented. Optional L2 normalization for magnitude-comparable vectors in hybrid search.
- **Separate document and query encoding** -- `encode()` uses the full BM25/TF-IDF formula with document length normalization. `encodeQuery()` uses IDF-only weighting with no length normalization, matching the distinction Pinecone requires between document and query sparse vectors.
- **Hash-based term mapping** -- MurmurHash3 (32-bit, x86) modulo a configurable vocabulary size (default: 262,144). Same approach as Pinecone's `pinecone-text`, enabling mixed-language pipelines.
- **Vocabulary-based term mapping** -- Sequential integer ID assignment during `fit()` for collision-free mapping. Configurable OOV strategies: ignore, hash fallback, or error.
- **Built-in English tokenizer** -- Lowercasing, Unicode-aware punctuation removal, whitespace splitting, English stopword removal (~175 words), and minimum token length filtering.
- **Porter stemming** -- Optional suffix-stripping stemmer implemented from scratch. Increases recall by collapsing word forms ("running", "runs", "runner" all map to "run").
- **Custom tokenizer hook** -- Supply a `(text: string) => string[]` function to replace the entire tokenization pipeline. Enables non-English languages, domain-specific tokenization, and code search.
- **Corpus fitting** -- `fit()` computes document frequencies, average document length, and vocabulary from a corpus. `partialFit()` incrementally updates statistics with new documents.
- **Model serialization** -- `serialize()` and `deserialize()` save and restore fitted models as JSON for deployment without re-fitting.
- **Batch encoding** -- `encodeBatch()` encodes multiple documents in a single call.
- **Milvus utility** -- `toMilvusSparse()` converts sparse vectors to the `Record<number, number>` format Milvus expects.
- **Zero dependencies** -- Tokenization, hashing, stemming, and scoring are all self-contained.
- **CLI** -- Command-line interface for fitting corpora, encoding documents, encoding queries, and inspecting model statistics.

---

## API Reference

### `createBM25(options?)`

Creates a `BM25Encoder` instance.

```typescript
import { createBM25 } from 'sparse-encode';

const encoder = createBM25({
  k1: 1.2,
  b: 0.75,
  indexing: 'hash',
  vocabSize: 262144,
});
```

**Parameters:**

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `k1` | `number` | `1.2` | Term frequency saturation. Range: [0, 3]. |
| `b` | `number` | `0.75` | Document length normalization. Range: [0, 1]. |
| `indexing` | `'hash' \| 'vocab'` | `'hash'` | Term-to-index mapping strategy. |
| `vocabSize` | `number` | `262144` | Vocabulary size for hash-based mapping. |
| `hashSeed` | `number` | `0` | MurmurHash3 seed value. |
| `oovStrategy` | `'ignore' \| 'hash' \| 'error'` | `'ignore'` | Out-of-vocabulary handling for vocabulary-based mapping. |
| `tokenizer` | `TokenizerFn` | built-in | Custom tokenizer function. Replaces the default pipeline. |
| `stemming` | `boolean` | `false` | Enable Porter stemming in the default tokenizer. |
| `stopwords` | `string[] \| false` | built-in list | Custom stopword list, or `false` to disable stopword removal. |
| `additionalStopwords` | `string[]` | `[]` | Extra stopwords appended to the built-in list. |
| `minTokenLength` | `number` | `2` | Minimum token length after tokenization. |

**Returns:** `BM25Encoder`

**Throws:** `SparseEncodeError` with code `'INVALID_OPTIONS'` if parameter values are out of range (e.g., k1 not in [0, 3], b not in [0, 1], vocabSize not positive).

---

### `createTFIDF(options?)`

Creates a `TFIDFEncoder` instance.

```typescript
import { createTFIDF } from 'sparse-encode';

const encoder = createTFIDF({
  tfVariant: 'log',
  normalize: true,
});
```

**Parameters:**

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `tfVariant` | `'raw' \| 'log' \| 'augmented'` | `'log'` | Term frequency variant. |
| `normalize` | `boolean` | `true` | L2-normalize the output vector. |
| `indexing` | `'hash' \| 'vocab'` | `'hash'` | Term-to-index mapping strategy. |
| `vocabSize` | `number` | `262144` | Vocabulary size for hash-based mapping. |
| `hashSeed` | `number` | `0` | MurmurHash3 seed value. |
| `oovStrategy` | `'ignore' \| 'hash' \| 'error'` | `'ignore'` | Out-of-vocabulary handling for vocabulary-based mapping. |
| `tokenizer` | `TokenizerFn` | built-in | Custom tokenizer function. Replaces the default pipeline. |
| `stemming` | `boolean` | `false` | Enable Porter stemming in the default tokenizer. |
| `stopwords` | `string[] \| false` | built-in list | Custom stopword list, or `false` to disable stopword removal. |
| `additionalStopwords` | `string[]` | `[]` | Extra stopwords appended to the built-in list. |
| `minTokenLength` | `number` | `2` | Minimum token length after tokenization. |

**Returns:** `TFIDFEncoder`

**Throws:** `SparseEncodeError` with code `'INVALID_OPTIONS'` if parameter values are invalid.

---

### `encoder.fit(documents)`

Computes IDF statistics from a corpus. Required before encoding.

```typescript
encoder.fit([
  'document one text',
  'document two text',
  'document three text',
]);
```

**Parameters:**

- `documents` (`string[]`) -- Array of document texts to fit on.

**Returns:** `void`

**Throws:**
- `SparseEncodeError` with code `'EMPTY_CORPUS'` if `documents` is an empty array.

Calling `fit()` a second time replaces the previous fit entirely. For incremental updates, use `partialFit()`.

---

### `encoder.partialFit(documents)`

Incrementally updates fitted statistics with additional documents without re-processing the entire corpus.

```typescript
encoder.fit(initialCorpus);
// Later, when new documents arrive:
encoder.partialFit(newDocuments);
```

**Parameters:**

- `documents` (`string[]`) -- Array of new document texts.

**Returns:** `void`

**Throws:**
- `SparseEncodeError` with code `'NOT_FITTED'` if the encoder has not been fitted yet (call `fit()` first).

Updates N, document frequencies, and average document length. For vocabulary-based indexing, new terms are assigned sequential IDs continuing from the last assigned ID.

---

### `encoder.encode(text)`

Encodes a document as a sparse vector using the full BM25 or TF-IDF formula with document length normalization.

```typescript
const sv: SparseVector = encoder.encode('the quick brown fox');
// { indices: [4821, 19203, 51847, 182033], values: [1.23, 0.87, 1.56, 2.01] }
```

**Parameters:**

- `text` (`string`) -- Document text to encode.

**Returns:** `SparseVector` -- `{ indices: number[], values: number[] }` with indices sorted ascending and all values positive.

**Throws:**
- `SparseEncodeError` with code `'NOT_FITTED'` if the encoder has not been fitted.

Returns `{ indices: [], values: [] }` for empty text or text consisting entirely of stopwords.

---

### `encoder.encodeBatch(texts)`

Encodes multiple documents as sparse vectors.

```typescript
const vectors: SparseVector[] = encoder.encodeBatch([
  'document one text',
  'document two text',
  'document three text',
]);
```

**Parameters:**

- `texts` (`string[]`) -- Array of document texts to encode.

**Returns:** `SparseVector[]`

---

### `encoder.encodeQuery(query)`

Encodes a query as a sparse vector using IDF-only weighting. No term frequency weighting and no document length normalization are applied. Duplicate terms in the query produce a single entry (no score increase for repeated terms).

```typescript
const querySv: SparseVector = encoder.encodeQuery('quick brown fox');
```

**Parameters:**

- `query` (`string`) -- Query text to encode.

**Returns:** `SparseVector`

**Throws:**
- `SparseEncodeError` with code `'NOT_FITTED'` if the encoder has not been fitted.

Use `encodeQuery()` for search queries and `encode()` for documents. Using document encoding for queries or vice versa degrades retrieval quality.

---

### `encoder.serialize()`

Serializes the fitted model to a JSON string for persistence and deployment.

```typescript
import * as fs from 'fs';

const json: string = encoder.serialize();
fs.writeFileSync('model.json', json);
```

**Returns:** `string` -- JSON string containing model type, version, parameters, corpus statistics, indexing configuration, and tokenizer settings.

**Throws:**
- `SparseEncodeError` with code `'NOT_FITTED'` if the encoder has not been fitted.

---

### `BM25Encoder.deserialize(json, options?)`

Restores a BM25 encoder from a serialized JSON string. The restored encoder is ready to encode without calling `fit()`.

```typescript
import * as fs from 'fs';

const json = fs.readFileSync('model.json', 'utf-8');
const encoder = BM25Encoder.deserialize(json);
```

When the original encoder used a custom tokenizer, provide the same tokenizer function during deserialization:

```typescript
const encoder = BM25Encoder.deserialize(json, {
  tokenizer: myCustomTokenizer,
});
```

**Parameters:**

- `json` (`string`) -- Serialized model JSON string.
- `options` (`{ tokenizer?: TokenizerFn }`) -- Optional. Custom tokenizer to restore.

**Returns:** `BM25Encoder`

**Throws:**
- `SparseEncodeError` with code `'SERIALIZATION_ERROR'` for corrupted or invalid JSON.

---

### `TFIDFEncoder.deserialize(json, options?)`

Restores a TF-IDF encoder from a serialized JSON string. Same interface as `BM25Encoder.deserialize()`.

```typescript
const encoder = TFIDFEncoder.deserialize(json);
```

**Parameters:**

- `json` (`string`) -- Serialized model JSON string.
- `options` (`{ tokenizer?: TokenizerFn }`) -- Optional. Custom tokenizer to restore.

**Returns:** `TFIDFEncoder`

**Throws:**
- `SparseEncodeError` with code `'SERIALIZATION_ERROR'` for corrupted or invalid JSON.

---

### `encoder.getStats()`

Returns statistics about the fitted corpus.

```typescript
const stats: FitStats = encoder.getStats();
// { N: 10000, avgdl: 142.5, vocabSize: 87432, totalTokens: 1425000 }
```

**Returns:** `FitStats`

**Throws:**
- `SparseEncodeError` with code `'NOT_FITTED'` if the encoder has not been fitted.

---

### `toMilvusSparse(sv)`

Converts a `SparseVector` to the dictionary format Milvus expects.

```typescript
import { toMilvusSparse } from 'sparse-encode';

const milvusFormat = toMilvusSparse(sparseVector);
// { 4821: 1.23, 19203: 0.87, 51847: 1.56, 182033: 2.01 }
```

**Parameters:**

- `sv` (`SparseVector`) -- Sparse vector to convert.

**Returns:** `Record<number, number>`

---

### Types

```typescript
/** Sparse vector in the format expected by Pinecone, Qdrant, and Milvus. */
interface SparseVector {
  /** Sorted ascending integer indices in [0, vocabSize). */
  indices: number[];
  /** Values corresponding to each index. Same length as indices. All positive. */
  values: number[];
}

/** Custom tokenizer function. Takes raw text, returns array of term strings. */
type TokenizerFn = (text: string) => string[];

/** Statistics about the fitted corpus. */
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

/** Options for createBM25(). */
interface BM25Options {
  k1?: number;
  b?: number;
  indexing?: 'hash' | 'vocab';
  vocabSize?: number;
  hashSeed?: number;
  oovStrategy?: 'ignore' | 'hash' | 'error';
  tokenizer?: TokenizerFn;
  stemming?: boolean;
  stopwords?: string[] | false;
  additionalStopwords?: string[];
  minTokenLength?: number;
}

/** Options for createTFIDF(). */
interface TFIDFOptions {
  tfVariant?: 'raw' | 'log' | 'augmented';
  normalize?: boolean;
  indexing?: 'hash' | 'vocab';
  vocabSize?: number;
  hashSeed?: number;
  oovStrategy?: 'ignore' | 'hash' | 'error';
  tokenizer?: TokenizerFn;
  stemming?: boolean;
  stopwords?: string[] | false;
  additionalStopwords?: string[];
  minTokenLength?: number;
}
```

---

## Configuration

### BM25 Parameter Tuning

| Parameter | Default | Description | Tuning guidance |
|-----------|---------|-------------|-----------------|
| `k1` | `1.2` | Term frequency saturation. At `k1 = 0`, term frequency is ignored (Boolean matching). Higher values allow more influence from repeated terms. | For short documents (tweets, titles): use default. For code search: increase to 1.5--2.0. |
| `b` | `0.75` | Document length normalization. At `b = 0`, no length normalization. At `b = 1`, full normalization. | For short documents: reduce to 0.3--0.5. For long documents: use default. |

### TF-IDF Variants

| Variant | Formula | Use case |
|---------|---------|----------|
| `'raw'` | `f(t, D)` | Raw term count. Simple but unbounded. Penalizes short documents. |
| `'log'` (default) | `1 + log(f(t, D))` | Sublinear scaling. Best balance of simplicity and quality. |
| `'augmented'` | `0.5 + 0.5 * f / max_f` | Normalized by max term frequency. Useful for comparing documents of very different lengths. |

### Term-to-Index Mapping

| Strategy | Description | Trade-off |
|----------|-------------|-----------|
| `'hash'` (default) | MurmurHash3 modulo `vocabSize`. No vocabulary file needed. | Rare hash collisions (~0.005 per document at default vocabSize). Compatible with Pinecone's `pinecone-text`. |
| `'vocab'` | Sequential IDs assigned during `fit()`. | Zero collisions. Compact index space. Requires vocabulary in serialized model. New terms need OOV handling. |

### OOV Strategies (vocabulary-based mapping only)

| Strategy | Behavior |
|----------|----------|
| `'ignore'` (default) | Skip out-of-vocabulary terms. They do not appear in the sparse vector. |
| `'hash'` | Fall back to hash-based mapping for unknown terms. |
| `'error'` | Throw `SparseEncodeError` for unknown terms. |

### Tokenizer Configuration

| Option | Default | Description |
|--------|---------|-------------|
| `stemming` | `false` | Enable Porter stemming. Increases recall at the cost of precision. |
| `stopwords` | built-in English list | Custom stopword list (`string[]`), or `false` to disable removal entirely. |
| `additionalStopwords` | `[]` | Extra stopwords appended to the built-in list. Ignored if custom `stopwords` are provided. |
| `minTokenLength` | `2` | Minimum token length. Tokens shorter than this are discarded. |

All tokenizer options are ignored when a custom `tokenizer` function is provided.

---

## Error Handling

All errors thrown by `sparse-encode` are instances of `SparseEncodeError`, which extends `Error` and includes a `code` property for programmatic error handling.

```typescript
import { createBM25, SparseEncodeError } from 'sparse-encode';

const encoder = createBM25();

try {
  encoder.encode('some text');
} catch (err) {
  if (err instanceof SparseEncodeError) {
    switch (err.code) {
      case 'NOT_FITTED':
        // Call encoder.fit() before encoding
        break;
      case 'INVALID_OPTIONS':
        // Check option values (k1, b, vocabSize, etc.)
        break;
      case 'EMPTY_CORPUS':
        // Provide a non-empty array to fit()
        break;
      case 'SERIALIZATION_ERROR':
        // Check the JSON string passed to deserialize()
        break;
    }
  }
}
```

### Error Codes

| Code | When thrown |
|------|------------|
| `NOT_FITTED` | `encode()`, `encodeQuery()`, `encodeBatch()`, `serialize()`, `getStats()`, or `partialFit()` called before `fit()`. |
| `INVALID_OPTIONS` | Invalid parameter values passed to `createBM25()` or `createTFIDF()`. |
| `EMPTY_CORPUS` | `fit()` called with an empty document array. |
| `SERIALIZATION_ERROR` | `deserialize()` called with corrupted, invalid, or incompatible JSON. |

---

## Advanced Usage

### Incremental Fitting

Update corpus statistics as new documents arrive without re-processing the entire corpus:

```typescript
const encoder = createBM25();
encoder.fit(initialCorpus);

// Later, when new documents arrive:
encoder.partialFit(newDocuments);

// Encoder now reflects both initial and new documents
const sv = encoder.encode('some text');
```

`partialFit()` updates N, document frequencies, and average document length. For vocabulary-based indexing, new terms receive sequential IDs continuing from the last assigned ID.

### Model Serialization and Deployment

Serialize a fitted model to avoid re-fitting on every application start:

```typescript
import * as fs from 'fs';
import { createBM25, BM25Encoder } from 'sparse-encode';

// Fit once, serialize
const encoder = createBM25();
encoder.fit(corpus);
fs.writeFileSync('bm25-model.json', encoder.serialize());

// Load in production -- no fit() needed
const json = fs.readFileSync('bm25-model.json', 'utf-8');
const prodEncoder = BM25Encoder.deserialize(json);
const sv = prodEncoder.encode('incoming document');
```

The serialized model stores parameters, corpus statistics (N, avgdl, document frequencies), and indexing configuration. For hash-based indexing, typical model size is 2--8 MB for a 100,000-document corpus. For vocabulary-based indexing, add 1--4 MB for the vocabulary map.

### Custom Tokenizer

Replace the built-in tokenizer for non-English text, code search, or domain-specific tokenization:

```typescript
const encoder = createBM25({
  tokenizer: (text: string): string[] => {
    // Domain-specific: split camelCase and snake_case, keep digits
    return text
      .replace(/([a-z])([A-Z])/g, '$1 $2')
      .replace(/_/g, ' ')
      .toLowerCase()
      .split(/\s+/)
      .filter(t => t.length >= 2);
  },
});
```

When a custom tokenizer is provided, all built-in tokenizer options (`stemming`, `stopwords`, `additionalStopwords`, `minTokenLength`) are ignored. The custom function is responsible for its own preprocessing.

When serializing a model that uses a custom tokenizer, the tokenizer function itself is not stored (functions are not serializable). Provide the same tokenizer during deserialization:

```typescript
const restored = BM25Encoder.deserialize(json, {
  tokenizer: myCustomTokenizer,
});
```

### Pinecone Integration

```typescript
import { createBM25 } from 'sparse-encode';
import { Pinecone } from '@pinecone-database/pinecone';

const bm25 = createBM25();
bm25.fit(allDocumentTexts);

const pinecone = new Pinecone();
const index = pinecone.index('hybrid-search');

// Index documents with both dense and sparse vectors
for (const doc of documents) {
  await index.upsert([{
    id: doc.id,
    values: denseVector,
    sparseValues: bm25.encode(doc.text),
    metadata: { title: doc.title },
  }]);
}

// Query with hybrid search
const results = await index.query({
  vector: denseQueryVector,
  sparseVector: bm25.encodeQuery(queryText),
  topK: 10,
  includeMetadata: true,
});
```

### Qdrant Integration

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

// Query with Reciprocal Rank Fusion
const results = await qdrant.query('my-collection', {
  query: { fusion: 'rrf' },
  prefetch: [
    { query: denseQueryVector, using: 'dense', limit: 20 },
    { query: bm25.encodeQuery(queryText), using: 'sparse', limit: 20 },
  ],
  limit: 10,
});
```

### Milvus Integration

```typescript
import { createBM25, toMilvusSparse } from 'sparse-encode';
import { MilvusClient } from '@zilliz/milvus2-sdk-node';

const bm25 = createBM25();
bm25.fit(corpus);

const milvus = new MilvusClient({ address: 'localhost:19530' });

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

### Document vs. Query Encoding

Document encoding and query encoding serve different purposes and produce different sparse vectors for the same input text:

- **`encode(text)`** -- Scores each term using the full BM25/TF-IDF formula with term frequency weighting and document length normalization. Use this when indexing documents.
- **`encodeQuery(query)`** -- Scores each term using IDF only. No term frequency weighting (repeated query terms do not increase the score) and no document length normalization. Use this when searching.

The vector database's dot product between query and document sparse vectors then naturally computes the correct BM25/TF-IDF retrieval score.

```typescript
// These produce DIFFERENT vectors -- use the right one for the right purpose
const docVector = encoder.encode('quick brown fox');
const queryVector = encoder.encodeQuery('quick brown fox');
```

### Integration with embed-cache and fusion-rank

Combine `sparse-encode` with `embed-cache` (dense embedding caching) and `fusion-rank` (score fusion) for a complete hybrid search pipeline:

```typescript
import { createBM25 } from 'sparse-encode';
import { createCache } from 'embed-cache';
import { fuse } from 'fusion-rank';

const bm25 = createBM25();
bm25.fit(corpus);

const embedCache = createCache({
  model: 'text-embedding-3-small',
  embedder: openaiEmbedder,
  storage: { type: 'sqlite', path: './embed-cache.db' },
});

// Encode documents with both dense and sparse vectors
async function encodeDocument(text: string) {
  const [dense, sparse] = await Promise.all([
    embedCache.embed(text),
    Promise.resolve(bm25.encode(text)),
  ]);
  return { dense, sparse };
}

// Application-level fusion
const denseResults = await vectorDb.searchDense(denseQueryVector, { topK: 50 });
const sparseResults = await vectorDb.searchSparse(bm25.encodeQuery(query), { topK: 50 });

const fusedResults = fuse([denseResults, sparseResults], {
  method: 'rrf',
  k: 60,
});
```

---

## CLI

The `sparse-encode` CLI provides commands for fitting corpora, encoding documents, encoding queries, and inspecting model statistics.

```bash
# Global install
npm install -g sparse-encode

# Or use npx
npx sparse-encode <command> [options]
```

### `sparse-encode fit`

Fits an encoder on a corpus and saves the model.

```bash
sparse-encode fit --input corpus.jsonl --output model.json

# With options
sparse-encode fit \
  --input corpus.jsonl \
  --output model.json \
  --algorithm tfidf \
  --tf-variant log \
  --indexing vocab \
  --stemming \
  --format json
```

| Option | Description |
|--------|-------------|
| `--input <path>` | Path to corpus file (JSONL or JSON array of strings). Required. |
| `--output <path>` | Path to write the fitted model JSON. Required. |
| `--algorithm <alg>` | Scoring algorithm: `bm25` (default) or `tfidf`. |
| `--k1 <n>` | BM25 k1 parameter. Default: 1.2. |
| `--b <n>` | BM25 b parameter. Default: 0.75. |
| `--tf-variant <v>` | TF-IDF TF variant: `raw`, `log`, or `augmented`. Default: `log`. |
| `--indexing <type>` | Index mapping: `hash` (default) or `vocab`. |
| `--vocab-size <n>` | Hash vocabulary size. Default: 262144. |
| `--stemming` | Enable Porter stemming. |
| `--format <fmt>` | Output format: `human` (default) or `json`. |

### `sparse-encode encode`

Encodes one or more texts as document sparse vectors.

```bash
sparse-encode encode --model model.json --text "the quick brown fox"
sparse-encode encode --model model.json --input documents.jsonl --output vectors.json
```

| Option | Description |
|--------|-------------|
| `--model <path>` | Path to fitted model JSON. Required. |
| `--text <text>` | Single text to encode. Mutually exclusive with `--input`. |
| `--input <path>` | Path to file with texts (one per line or JSONL). Mutually exclusive with `--text`. |
| `--output <path>` | Path to write output. Default: stdout. |
| `--format <fmt>` | Output format: `json` (default) or `jsonl`. |

### `sparse-encode query`

Encodes a query as a sparse vector using query-specific encoding (IDF-only weighting).

```bash
sparse-encode query --model model.json --text "search terms"
```

| Option | Description |
|--------|-------------|
| `--model <path>` | Path to fitted model JSON. Required. |
| `--text <text>` | Query text to encode. Required. |
| `--format <fmt>` | Output format: `json` (default) or `human`. |

### `sparse-encode stats`

Prints statistics about a fitted model.

```bash
sparse-encode stats --model model.json
```

| Option | Description |
|--------|-------------|
| `--model <path>` | Path to fitted model JSON. Required. |
| `--format <fmt>` | Output format: `human` (default) or `json`. |

### Exit Codes

| Code | Meaning |
|------|---------|
| `0` | Success. |
| `1` | Operation failed (IO error, invalid model file). |
| `2` | Configuration error (missing required options, invalid flags). |

---

## TypeScript

`sparse-encode` is written in TypeScript and ships type declarations alongside compiled JavaScript. All public interfaces, types, and function signatures are fully typed.

```typescript
import {
  createBM25,
  createTFIDF,
  toMilvusSparse,
  BM25Encoder,
  TFIDFEncoder,
  SparseEncodeError,
} from 'sparse-encode';

import type {
  SparseVector,
  BM25Options,
  TFIDFOptions,
  FitStats,
  TokenizerFn,
} from 'sparse-encode';
```

---

## License

MIT
