import { tokenize } from './tokenizer'
import { Vocabulary } from './vocab'
import type { BM25Options, BM25Encoder, SparseVector, FitStats } from './types'

export function createBM25(options?: BM25Options): BM25Encoder {
  const k1 = options?.k1 ?? 1.5
  const b = options?.b ?? 0.75
  const stemOpt = options?.stem ?? true
  const tokenizerFn = options?.tokenizer
  const extraStopwords = options?.stopwords ?? []

  const vocab = new Vocabulary()
  // df[termId] = number of documents containing that term
  const df = new Map<number, number>()
  let N = 0
  let avgdl = 0
  let totalTokens = 0
  let fitted = false

  function tokenizeText(text: string): string[] {
    return tokenize(text, { stem: stemOpt, tokenizer: tokenizerFn, stopwords: extraStopwords })
  }

  function fit(documents: string[]): void {
    N = documents.length
    let totalLen = 0
    const tokenizedDocs: string[][] = []

    for (const doc of documents) {
      const tokens = tokenizeText(doc)
      tokenizedDocs.push(tokens)
      totalLen += tokens.length
      totalTokens += tokens.length

      // Register all terms in vocab first
      const seen = new Set<number>()
      for (const token of tokens) {
        const id = vocab.getOrAdd(token)
        if (!seen.has(id)) {
          seen.add(id)
          df.set(id, (df.get(id) ?? 0) + 1)
        }
      }
    }

    avgdl = N > 0 ? totalLen / N : 0
    fitted = true
  }

  function computeBM25(tokens: string[], dl: number): SparseVector {
    if (!fitted) throw new Error('BM25Encoder must be fit() before encode()')

    // Count term frequencies
    const tf = new Map<number, number>()
    for (const token of tokens) {
      const id = vocab.getId(token)
      if (id !== undefined) {
        tf.set(id, (tf.get(id) ?? 0) + 1)
      }
    }

    const entries: { idx: number; val: number }[] = []
    for (const [termId, termTf] of tf) {
      const termDf = df.get(termId) ?? 0
      const idf = Math.log((N - termDf + 0.5) / (termDf + 0.5) + 1)
      const score = idf * (termTf * (k1 + 1)) / (termTf + k1 * (1 - b + b * dl / avgdl))
      if (score > 0) {
        entries.push({ idx: termId, val: score })
      }
    }

    entries.sort((a, c) => a.idx - c.idx)
    return {
      indices: entries.map(e => e.idx),
      values: entries.map(e => e.val),
    }
  }

  function encode(text: string): SparseVector {
    const tokens = tokenizeText(text)
    return computeBM25(tokens, tokens.length)
  }

  function encodeBatch(texts: string[]): SparseVector[] {
    return texts.map(t => encode(t))
  }

  function encodeQuery(text: string): SparseVector {
    // Query encoding: no length normalization (treat avgdl as doc length)
    if (!fitted) throw new Error('BM25Encoder must be fit() before encodeQuery()')
    const tokens = tokenizeText(text)
    return computeBM25(tokens, avgdl)
  }

  function serialize(): string {
    return JSON.stringify({
      N,
      avgdl,
      totalTokens,
      df: Object.fromEntries(df),
      vocab: vocab.serialize(),
      options: { k1, b, stem: stemOpt, stopwords: extraStopwords },
    })
  }

  function getStats(): FitStats {
    return { N, avgdl, vocabSize: vocab.size, totalTokens }
  }

  return { fit, encode, encodeBatch, encodeQuery, serialize, getStats }
}
