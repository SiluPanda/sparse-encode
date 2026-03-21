import { tokenize } from './tokenizer'
import { Vocabulary } from './vocab'
import type { TFIDFOptions, TFIDFEncoder, SparseVector, FitStats } from './types'

export function createTFIDF(options?: TFIDFOptions): TFIDFEncoder {
  const stemOpt = options?.stem ?? true
  const tokenizerFn = options?.tokenizer
  const extraStopwords = options?.stopwords ?? []
  const sublinearTf = options?.sublinearTf ?? false

  const vocab = new Vocabulary()
  const df = new Map<number, number>()
  let N = 0
  let totalTokens = 0
  let avgdl = 0
  let fitted = false

  function tokenizeText(text: string): string[] {
    return tokenize(text, { stem: stemOpt, tokenizer: tokenizerFn, stopwords: extraStopwords })
  }

  function fit(documents: string[]): void {
    N = documents.length
    let totalLen = 0

    for (const doc of documents) {
      const tokens = tokenizeText(doc)
      totalLen += tokens.length
      totalTokens += tokens.length

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

  function encode(text: string): SparseVector {
    if (!fitted) throw new Error('TFIDFEncoder must be fit() before encode()')

    const tokens = tokenizeText(text)
    const dl = tokens.length
    if (dl === 0) return { indices: [], values: [] }

    // Count raw term frequencies
    const rawTf = new Map<number, number>()
    for (const token of tokens) {
      const id = vocab.getId(token)
      if (id !== undefined) {
        rawTf.set(id, (rawTf.get(id) ?? 0) + 1)
      }
    }

    const entries: { idx: number; val: number }[] = []
    for (const [termId, count] of rawTf) {
      const tf = sublinearTf ? 1 + Math.log(count) : count / dl
      const termDf = df.get(termId) ?? 0
      const idf = Math.log((N + 1) / (termDf + 1)) + 1
      const score = tf * idf
      if (score > 0) {
        entries.push({ idx: termId, val: score })
      }
    }

    // L2 normalize
    const norm = Math.sqrt(entries.reduce((acc, e) => acc + e.val * e.val, 0))
    if (norm > 0) {
      for (const e of entries) e.val /= norm
    }

    entries.sort((a, c) => a.idx - c.idx)
    return {
      indices: entries.map(e => e.idx),
      values: entries.map(e => e.val),
    }
  }

  function encodeBatch(texts: string[]): SparseVector[] {
    return texts.map(t => encode(t))
  }

  function encodeQuery(text: string): SparseVector {
    // For queries, same as encode
    return encode(text)
  }

  function serialize(): string {
    return JSON.stringify({
      N,
      avgdl,
      totalTokens,
      df: Object.fromEntries(df),
      vocab: vocab.serialize(),
      options: { stem: stemOpt, sublinearTf, stopwords: extraStopwords },
    })
  }

  function getStats(): FitStats {
    return { N, avgdl, vocabSize: vocab.size, totalTokens }
  }

  return { fit, encode, encodeBatch, encodeQuery, serialize, getStats }
}
