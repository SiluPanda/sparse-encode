# sparse-encode

Generate BM25 and TF-IDF sparse vectors in JavaScript/TypeScript. Designed for use with sparse vector search engines (e.g., Pinecone sparse indexes, Qdrant sparse vectors).

## Install

```bash
npm install sparse-encode
```

## Quick Start

```typescript
import { createBM25, createTFIDF } from 'sparse-encode'

const docs = [
  'the quick brown fox jumps over the lazy dog',
  'the dog barked loudly at the fox',
  'cats and dogs are common pets',
]

// BM25
const bm25 = createBM25({ k1: 1.5, b: 0.75 })
bm25.fit(docs)
const vec = bm25.encode('quick fox')
// { indices: [2, 5, ...], values: [0.82, 1.23, ...] }

const queryVec = bm25.encodeQuery('fox')

// TF-IDF
const tfidf = createTFIDF({ sublinearTf: false })
tfidf.fit(docs)
const tVec = tfidf.encode('quick fox')
// L2-normalized: sum of squares ≈ 1.0
```

## BM25 Encoder

```typescript
const enc = createBM25(options?)
enc.fit(documents: string[])           // build vocabulary + IDF statistics
enc.encode(text: string): SparseVector // encode a document
enc.encodeQuery(text: string): SparseVector // encode a query (no length norm)
enc.encodeBatch(texts: string[]): SparseVector[]
enc.getStats(): FitStats               // { N, avgdl, vocabSize, totalTokens }
enc.serialize(): string                // JSON snapshot (vocab + df + options)
```

### Options

| Option | Default | Description |
|--------|---------|-------------|
| `k1` | `1.5` | Term frequency saturation parameter |
| `b` | `0.75` | Length normalization parameter |
| `stem` | `true` | Apply Porter stemmer |
| `stopwords` | `[]` | Additional stopwords to remove |
| `tokenizer` | built-in | Custom tokenizer function |

## TF-IDF Encoder

```typescript
const enc = createTFIDF(options?)
enc.fit(documents: string[])
enc.encode(text: string): SparseVector  // L2-normalized TF-IDF vector
enc.encodeQuery(text: string): SparseVector
enc.encodeBatch(texts: string[]): SparseVector[]
enc.getStats(): FitStats
enc.serialize(): string
```

### Options

| Option | Default | Description |
|--------|---------|-------------|
| `stem` | `true` | Apply Porter stemmer |
| `stopwords` | `[]` | Additional stopwords to remove |
| `sublinearTf` | `false` | Use `1 + log(tf)` instead of `tf / doc_length` |
| `tokenizer` | built-in | Custom tokenizer function |

## SparseVector Format

```typescript
interface SparseVector {
  indices: number[]  // term IDs (sorted ascending)
  values: number[]   // corresponding scores
}
```

Each position `i` maps `indices[i]` → `values[i]`. Zero values are omitted.

## Tokenizer Pipeline

The built-in tokenizer:
1. Lowercases the input
2. Splits on non-word characters
3. Removes pure numbers
4. Removes common English stopwords
5. Applies the Porter stemmer (can be disabled with `stem: false`)

## License

MIT
