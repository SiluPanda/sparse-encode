export interface SparseVector {
  indices: number[]
  values: number[]
}

export interface FitStats {
  N: number
  avgdl: number
  vocabSize: number
  totalTokens: number
}

export type TokenizerFn = (text: string) => string[]

export interface BM25Options {
  k1?: number
  b?: number
  tokenizer?: TokenizerFn
  stopwords?: string[]
  stem?: boolean
}

export interface TFIDFOptions {
  tokenizer?: TokenizerFn
  stopwords?: string[]
  stem?: boolean
  sublinearTf?: boolean
}

export interface BM25Encoder {
  fit(documents: string[]): void
  encode(text: string): SparseVector
  encodeBatch(texts: string[]): SparseVector[]
  encodeQuery(text: string): SparseVector
  serialize(): string
  getStats(): FitStats
}

export interface TFIDFEncoder {
  fit(documents: string[]): void
  encode(text: string): SparseVector
  encodeBatch(texts: string[]): SparseVector[]
  encodeQuery(text: string): SparseVector
  serialize(): string
  getStats(): FitStats
}
