import { describe, it, expect } from 'vitest'
import { defaultTokenizer, tokenize } from '../tokenizer'

describe('defaultTokenizer', () => {
  it('lowercases input', () => {
    const tokens = defaultTokenizer('Hello World')
    expect(tokens.every(t => t === t.toLowerCase())).toBe(true)
  })

  it('removes stopwords', () => {
    const tokens = defaultTokenizer('the cat sat on the mat')
    expect(tokens).not.toContain('the')
    expect(tokens).not.toContain('on')
  })

  it('removes pure numbers', () => {
    const tokens = defaultTokenizer('I have 42 cats')
    expect(tokens).not.toContain('42')
  })

  it('splits on non-word characters', () => {
    const tokens = defaultTokenizer('hello-world foo.bar')
    expect(tokens.length).toBeGreaterThanOrEqual(2)
  })

  it('applies Porter stemmer (running → run)', () => {
    const tokens = defaultTokenizer('running')
    // porter stemmer maps running → run
    expect(tokens).toContain('run')
  })

  it('applies Porter stemmer (cats → cat)', () => {
    const tokens = defaultTokenizer('cats')
    expect(tokens).toContain('cat')
  })
})

describe('tokenize', () => {
  it('uses custom tokenizer when provided', () => {
    const custom = (t: string) => t.split(',')
    const tokens = tokenize('a,b,c', { tokenizer: custom })
    expect(tokens).toEqual(['a', 'b', 'c'])
  })

  it('applies extra stopwords', () => {
    const tokens = tokenize('cat dog bird', { stem: false, stopwords: ['dog'] })
    expect(tokens).not.toContain('dog')
    expect(tokens).toContain('cat')
    expect(tokens).toContain('bird')
  })

  it('skips stemming when stem=false', () => {
    const tokens = tokenize('running cats', { stem: false })
    expect(tokens).toContain('running')
    expect(tokens).toContain('cats')
  })
})
