import { stem } from './porter-stemmer'
import type { TokenizerFn } from './types'

const DEFAULT_STOPWORDS = new Set([
  'a', 'an', 'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
  'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
  'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
  'should', 'may', 'might', 'shall', 'can',
])

export function defaultTokenizer(text: string): string[] {
  return text
    .toLowerCase()
    .split(/[^\w]+/)
    .filter(t => t.length > 0 && !/^\d+$/.test(t))
    .filter(t => !DEFAULT_STOPWORDS.has(t))
    .map(t => stem(t))
}

export function tokenize(
  text: string,
  options?: {
    stopwords?: string[]
    stem?: boolean
    tokenizer?: TokenizerFn
  }
): string[] {
  const { tokenizer, stopwords, stem: doStem = true } = options ?? {}

  let tokens: string[]
  if (tokenizer) {
    tokens = tokenizer(text)
  } else {
    // Manual pipeline so we can honour stem=false
    tokens = text
      .toLowerCase()
      .split(/[^\w]+/)
      .filter(t => t.length > 0 && !/^\d+$/.test(t))
      .filter(t => !DEFAULT_STOPWORDS.has(t))

    if (doStem) {
      tokens = tokens.map(t => stem(t))
    }
  }

  if (stopwords && stopwords.length > 0) {
    const extra = new Set(stopwords.map(s => s.toLowerCase()))
    tokens = tokens.filter(t => !extra.has(t))
  }

  return tokens
}
