export class Vocabulary {
  private termToId = new Map<string, number>()
  private idToTerm: string[] = []

  getOrAdd(term: string): number {
    let id = this.termToId.get(term)
    if (id === undefined) {
      id = this.idToTerm.length
      this.termToId.set(term, id)
      this.idToTerm.push(term)
    }
    return id
  }

  getId(term: string): number | undefined {
    return this.termToId.get(term)
  }

  get size(): number {
    return this.idToTerm.length
  }

  terms(): string[] {
    return [...this.idToTerm]
  }

  serialize(): object {
    return { terms: this.idToTerm }
  }

  static deserialize(data: object): Vocabulary {
    const v = new Vocabulary()
    const d = data as { terms: string[] }
    for (const term of d.terms) {
      v.getOrAdd(term)
    }
    return v
  }
}
