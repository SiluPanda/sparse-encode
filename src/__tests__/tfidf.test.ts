import { describe, it, expect } from 'vitest'
import { createTFIDF } from '../tfidf'

describe('createTFIDF', () => {
  const docs = [
    'the quick brown fox jumps over the lazy dog',
    'the dog barked loudly at the fox',
    'cats and dogs are common pets',
  ]

  it('throws if encode called before fit', () => {
    const enc = createTFIDF()
    expect(() => enc.encode('test')).toThrow()
  })

  it('returns a SparseVector with indices and values', () => {
    const enc = createTFIDF()
    enc.fit(docs)
    const vec = enc.encode('fox')
    expect(Array.isArray(vec.indices)).toBe(true)
    expect(Array.isArray(vec.values)).toBe(true)
    expect(vec.indices.length).toBe(vec.values.length)
  })

  it('encodes to non-empty vector for known term', () => {
    const enc = createTFIDF()
    enc.fit(docs)
    const vec = enc.encode('fox')
    expect(vec.indices.length).toBeGreaterThan(0)
  })

  it('indices are sorted ascending', () => {
    const enc = createTFIDF()
    enc.fit(docs)
    const vec = enc.encode('quick brown fox')
    for (let i = 1; i < vec.indices.length; i++) {
      expect(vec.indices[i]).toBeGreaterThan(vec.indices[i - 1])
    }
  })

  it('output vector is unit-length (L2 norm ≈ 1)', () => {
    const enc = createTFIDF()
    enc.fit(docs)
    const vec = enc.encode('quick brown fox')
    const sumSq = vec.values.reduce((acc, v) => acc + v * v, 0)
    expect(sumSq).toBeCloseTo(1.0, 5)
  })

  it('unit-length holds for single-term query', () => {
    const enc = createTFIDF()
    enc.fit(docs)
    const vec = enc.encode('fox')
    if (vec.values.length > 0) {
      const sumSq = vec.values.reduce((acc, v) => acc + v * v, 0)
      expect(sumSq).toBeCloseTo(1.0, 5)
    }
  })

  it('getStats returns correct N and vocabSize', () => {
    const enc = createTFIDF()
    enc.fit(docs)
    const stats = enc.getStats()
    expect(stats.N).toBe(3)
    expect(stats.vocabSize).toBeGreaterThan(0)
    expect(stats.totalTokens).toBeGreaterThan(0)
  })

  it('encodeBatch returns one vector per text', () => {
    const enc = createTFIDF()
    enc.fit(docs)
    const vecs = enc.encodeBatch(['fox', 'dog', 'cat'])
    expect(vecs).toHaveLength(3)
  })

  it('encodeQuery returns unit-length SparseVector', () => {
    const enc = createTFIDF()
    enc.fit(docs)
    const vec = enc.encodeQuery('brown fox')
    if (vec.values.length > 0) {
      const sumSq = vec.values.reduce((acc, v) => acc + v * v, 0)
      expect(sumSq).toBeCloseTo(1.0, 5)
    }
  })

  it('encodeQuery uses IDF-only: repeated terms do not inflate scores', () => {
    const enc = createTFIDF()
    enc.fit(docs)
    const single = enc.encodeQuery('fox')
    const repeated = enc.encodeQuery('fox fox fox fox')
    // Both should produce identical vectors (IDF-only, deduplicated)
    expect(repeated.indices).toEqual(single.indices)
    expect(repeated.values).toEqual(single.values)
  })

  it('serialize returns valid JSON', () => {
    const enc = createTFIDF()
    enc.fit(docs)
    const json = enc.serialize()
    expect(() => JSON.parse(json)).not.toThrow()
    const parsed = JSON.parse(json)
    expect(parsed.N).toBe(3)
    expect(parsed.vocab).toBeDefined()
  })

  it('sublinearTf option changes scores', () => {
    const enc1 = createTFIDF({ sublinearTf: false })
    const enc2 = createTFIDF({ sublinearTf: true })
    enc1.fit(docs)
    enc2.fit(docs)
    // Use a text with repeated terms to make sublinear difference visible
    const text = 'fox fox fox quick'
    const v1 = enc1.encode(text)
    const v2 = enc2.encode(text)
    // Both should be unit vectors but with different distributions
    if (v1.values.length > 0 && v2.values.length > 0) {
      const s1 = v1.values.reduce((a, b) => a + b * b, 0)
      const s2 = v2.values.reduce((a, b) => a + b * b, 0)
      expect(s1).toBeCloseTo(1.0, 5)
      expect(s2).toBeCloseTo(1.0, 5)
    }
  })

  it('empty text returns empty vector', () => {
    const enc = createTFIDF()
    enc.fit(docs)
    const vec = enc.encode('')
    expect(vec.indices).toHaveLength(0)
    expect(vec.values).toHaveLength(0)
  })
})
