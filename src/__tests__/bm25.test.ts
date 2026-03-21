import { describe, it, expect } from 'vitest'
import { createBM25 } from '../bm25'

describe('createBM25', () => {
  const docs = [
    'the quick brown fox jumps over the lazy dog',
    'the dog barked loudly at the fox',
    'cats and dogs are common pets',
  ]

  it('throws if encode called before fit', () => {
    const enc = createBM25()
    expect(() => enc.encode('test')).toThrow()
  })

  it('returns a SparseVector with indices and values', () => {
    const enc = createBM25()
    enc.fit(docs)
    const vec = enc.encode('fox')
    expect(Array.isArray(vec.indices)).toBe(true)
    expect(Array.isArray(vec.values)).toBe(true)
    expect(vec.indices.length).toBe(vec.values.length)
  })

  it('encodes to non-empty vector for known term', () => {
    const enc = createBM25()
    enc.fit(docs)
    const vec = enc.encode('fox')
    expect(vec.indices.length).toBeGreaterThan(0)
  })

  it('indices are sorted ascending', () => {
    const enc = createBM25()
    enc.fit(docs)
    const vec = enc.encode('quick brown fox')
    for (let i = 1; i < vec.indices.length; i++) {
      expect(vec.indices[i]).toBeGreaterThan(vec.indices[i - 1])
    }
  })

  it('term in doc scores higher than term not in corpus', () => {
    const enc = createBM25()
    enc.fit(docs)
    const vec = enc.encode('fox')
    const foxIdx = vec.indices.findIndex(i => i >= 0)
    // vector should have at least one positive value
    expect(vec.values[foxIdx]).toBeGreaterThan(0)
  })

  it('getStats returns correct N and vocabSize', () => {
    const enc = createBM25()
    enc.fit(docs)
    const stats = enc.getStats()
    expect(stats.N).toBe(3)
    expect(stats.vocabSize).toBeGreaterThan(0)
    expect(stats.avgdl).toBeGreaterThan(0)
    expect(stats.totalTokens).toBeGreaterThan(0)
  })

  it('encodeBatch returns one vector per text', () => {
    const enc = createBM25()
    enc.fit(docs)
    const vecs = enc.encodeBatch(['fox', 'dog', 'cat'])
    expect(vecs).toHaveLength(3)
  })

  it('encodeQuery returns a SparseVector', () => {
    const enc = createBM25()
    enc.fit(docs)
    const vec = enc.encodeQuery('brown fox')
    expect(Array.isArray(vec.indices)).toBe(true)
    expect(Array.isArray(vec.values)).toBe(true)
  })

  it('serialize returns valid JSON', () => {
    const enc = createBM25()
    enc.fit(docs)
    const json = enc.serialize()
    expect(() => JSON.parse(json)).not.toThrow()
    const parsed = JSON.parse(json)
    expect(parsed.N).toBe(3)
    expect(parsed.vocab).toBeDefined()
  })

  it('custom k1/b options affect scores', () => {
    const enc1 = createBM25({ k1: 1.0, b: 0.5 })
    const enc2 = createBM25({ k1: 2.0, b: 0.9 })
    enc1.fit(docs)
    enc2.fit(docs)
    const v1 = enc1.encode('fox')
    const v2 = enc2.encode('fox')
    // Scores differ with different parameters
    const sum1 = v1.values.reduce((a, b) => a + b, 0)
    const sum2 = v2.values.reduce((a, b) => a + b, 0)
    expect(sum1).not.toBeCloseTo(sum2, 5)
  })
})
