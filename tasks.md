# sparse-encode -- Task Breakdown

## Phase 1: Project Scaffolding and Type Definitions

- [ ] **Install dev dependencies** -- Add `typescript`, `vitest`, `eslint`, and `@types/node` as devDependencies in `package.json`. Run `npm install` to generate `node_modules` and `package-lock.json`. | Status: not_done

- [ ] **Define all TypeScript types in `src/types.ts`** -- Create the `types.ts` file containing all public type definitions from the spec: `SparseVector` interface (`indices: number[]`, `values: number[]`), `TokenizerFn` type (`(text: string) => string[]`), `BM25Options` interface (k1, b, indexing, vocabSize, hashSeed, oovStrategy, tokenizer, stemming, stopwords, additionalStopwords, minTokenLength), `TFIDFOptions` interface (tfVariant, normalize, plus shared options), `FitStats` interface (N, avgdl, vocabSize, totalTokens), `BM25Encoder` interface (fit, partialFit, encode, encodeBatch, encodeQuery, serialize, getStats), `TFIDFEncoder` interface (same methods). | Status: not_done

- [ ] **Create custom error class in `src/errors.ts`** -- Implement `SparseEncodeError` extending `Error` with a `code` property typed as `'NOT_FITTED' | 'INVALID_OPTIONS' | 'EMPTY_CORPUS' | 'SERIALIZATION_ERROR'`. Include descriptive error messages for each code. | Status: not_done

- [ ] **Set up the public API barrel file `src/index.ts`** -- Export `createBM25`, `createTFIDF`, `BM25Encoder`, `TFIDFEncoder`, `toMilvusSparse`, `SparseVector`, `BM25Options`, `TFIDFOptions`, `FitStats`, `TokenizerFn`, `SparseEncodeError`. Initially these can be placeholder re-exports that are filled in as modules are built. | Status: not_done

- [ ] **Create test fixtures directory and corpus file** -- Create `src/__tests__/fixtures/corpus-small.json` containing a ~50-document test corpus of short English texts suitable for verifying BM25/TF-IDF scoring. Include documents with known term distributions so expected values can be hand-computed. | Status: not_done

---

## Phase 2: Tokenizer Pipeline

- [ ] **Implement built-in English stopword list in `src/tokenizer/stopwords.ts`** -- Export a `Set<string>` containing the ~175 English stopwords listed in the spec (a, about, above, after, again, against, all, am, an, and, any, are, aren't, as, at, be, because, been, before, being, below, between, both, but, by, can, can't, cannot, could, couldn't, did, didn't, do, does, doesn't, doing, don't, down, during, each, few, for, from, further, get, got, had, hadn't, has, hasn't, have, haven't, having, he, her, here, hers, herself, him, himself, his, how, i, if, in, into, is, isn't, it, its, itself, just, let, me, might, more, most, mustn't, my, myself, no, nor, not, of, off, on, once, only, or, other, ought, our, ours, ourselves, out, over, own, same, she, should, shouldn't, so, some, such, than, that, the, their, theirs, them, themselves, then, there, these, they, this, those, through, to, too, under, until, up, very, was, wasn't, we, were, weren't, what, when, where, which, while, who, whom, why, will, with, won't, would, wouldn't, you, your, yours, yourself, yourselves). | Status: not_done

- [ ] **Implement Porter stemmer in `src/tokenizer/stemmer.ts`** -- Implement the full Porter stemmer algorithm from scratch (no dependencies). Must handle all 5 steps: Step 1 (plural/verb suffixes), Step 2 (double suffixes), Step 3 (derivational suffixes), Step 4 (further suffixes), Step 5 (cleanup). Verify against known test cases: "running" -> "run", "connections" -> "connect", "effectively" -> "effect", "computational" -> "comput". Export a `porterStem(word: string): string` function. | Status: not_done

- [ ] **Implement default tokenizer in `src/tokenizer/tokenizer.ts`** -- Implement the 5-step default tokenizer pipeline: (1) lowercase the entire text, (2) replace all characters matching `/[^\p{L}\p{N}\-]/gu` with spaces (preserving hyphens within words), (3) split on `/\s+/` and filter empty strings, (4) remove tokens in the stopword set, (5) filter tokens shorter than `minTokenLength` (default 2). If `stemming: true`, add step 6: apply Porter stemmer to each token. Export a `createDefaultTokenizer(options)` function that returns a `TokenizerFn`. Accept options: `stemming`, `stopwords` (custom list or `false`), `additionalStopwords`, `minTokenLength`. | Status: not_done

- [ ] **Support custom tokenizer passthrough** -- When a custom `tokenizer` function is provided in encoder options, use it directly and skip the default pipeline entirely (no stopword removal, no stemming applied on top). The custom function is `(text: string) => string[]` and replaces the entire pipeline. | Status: not_done

- [ ] **Write tokenizer unit tests in `src/__tests__/tokenizer.test.ts`** -- Test cases: lowercase conversion, punctuation removal (verify "state-of-the-art" is preserved, "hello, world!" becomes two tokens), stopword removal (verify "the", "is", "a" are removed), minimum length filtering (single chars removed), empty input returns empty array, text with only stopwords returns empty array, custom stopword list replaces default, `stopwords: false` disables removal, `additionalStopwords` are appended to default list, `minTokenLength` is respected. | Status: not_done

- [ ] **Write Porter stemmer unit tests in `src/__tests__/stemmer.test.ts`** -- Test against known stem pairs: "running"/"run", "connections"/"connect", "effectively"/"effect", "computational"/"comput", "caresses"/"caress", "ponies"/"poni", "cats"/"cat", "feed"/"feed", "agreed"/"agre", "disabled"/"disabl". Test that already-stemmed words are unchanged. Test single-character and two-character words. | Status: not_done

---

## Phase 3: MurmurHash3 and Term-to-Index Mapping

- [ ] **Implement MurmurHash3 (32-bit x86) in `src/indexing/hash.ts`** -- Implement the MurmurHash3_x86_32 hash function from scratch following the Austin Appleby 2008 reference spec. Accept `(key: string, seed: number) => number`. Must produce identical output to the reference C implementation for all inputs. Also export a `hashTermToIndex(term: string, vocabSize: number, seed: number): number` function that returns `MurmurHash3(term, seed) >>> 0) % vocabSize` (unsigned modulo). | Status: not_done

- [ ] **Implement vocabulary-based mapping in `src/indexing/vocab.ts`** -- Implement a `VocabMapper` class or set of functions that: (1) during `fit()`, assign sequential integer IDs starting from 0 to each unique term encountered, (2) during encoding, look up term IDs from the vocabulary `Map<string, number>`, (3) handle OOV terms according to `oovStrategy`: `'ignore'` skips the term (returns `undefined`), `'hash'` falls back to hash-based mapping, `'error'` throws `SparseEncodeError` with code `INVALID_OPTIONS`. Support extending the vocabulary during `partialFit()` with IDs continuing from the last assigned ID. | Status: not_done

- [ ] **Write MurmurHash3 unit tests in `src/__tests__/murmurhash.test.ts`** -- Test against reference test vectors from the C implementation (known input/output pairs). Verify determinism (same input always same output). Verify with different seeds. Verify distribution: hash 10,000 unique English words into 262,144 buckets and verify fewer than 5 collisions (probabilistic test). | Status: not_done

- [ ] **Write term-to-index mapping tests in `src/__tests__/indexing.test.ts`** -- Test hash-based: verify `index = MurmurHash3(term, seed) % vocabSize`. Test hash collision handling: given two terms that map to the same index, verify scores are summed. Test vocabulary-based: verify sequential ID assignment from 0. Test OOV `'ignore'` skips unknown terms. Test OOV `'hash'` falls back to hash mapping for unknown terms. Test OOV `'error'` throws for unknown terms. Test vocabulary extension during partialFit. | Status: not_done

---

## Phase 4: IDF and Scoring Functions

- [ ] **Implement IDF computation in `src/scoring/idf.ts`** -- Export two IDF functions: (1) `bm25IDF(df: number, N: number): number` implementing `log((N - df + 0.5) / (df + 0.5) + 1)` (Robertson-Sparck Jones variant, always non-negative); (2) `tfidfIDF(df: number, N: number): number` implementing `log(1 + N / df)` with smoothing. Handle edge cases: df = 0 (return max IDF `log(N + 1)` for BM25, `log(1 + N)` for TF-IDF), df = N (return near-zero positive value). | Status: not_done

- [ ] **Implement BM25 per-term scoring in `src/scoring/bm25-score.ts`** -- Export a `bm25Score(tf: number, idf: number, docLength: number, avgdl: number, k1: number, b: number): number` function implementing: `idf * (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * docLength / avgdl))`. Return 0 when tf = 0. | Status: not_done

- [ ] **Implement TF-IDF per-term scoring in `src/scoring/tfidf-score.ts`** -- Export functions for all three TF variants: `rawTF(f: number): number` returning `f`; `logTF(f: number): number` returning `f > 0 ? 1 + Math.log(f) : 0`; `augmentedTF(f: number, maxF: number): number` returning `0.5 + 0.5 * f / maxF`. Export a `tfidfScore(tf: number, idf: number): number` returning `tf * idf`. | Status: not_done

- [ ] **Write scoring function tests in `src/__tests__/scoring.test.ts`** -- BM25 scoring: verify hand-computed values for known inputs (f, df, N, avgdl, docLength, k1, b); verify score is 0 when f=0; verify score increases with tf but saturates; verify score decreases for long docs when b>0; verify score is independent of doc length when b=0; verify IDF decreases as df increases; verify IDF is always positive. TF-IDF scoring: verify raw TF returns raw count; verify log TF returns `1 + log(f)`; verify augmented TF returns `0.5 + 0.5 * f/maxF`; verify TF-IDF product is correct. | Status: not_done

---

## Phase 5: Sparse Vector Construction

- [ ] **Implement sparse vector builder in `src/sparse-vector.ts`** -- Export a `buildSparseVector(entries: Map<number, number>): SparseVector` function that: (1) filters out entries with value <= 0, (2) sorts by index ascending, (3) returns `{ indices: number[], values: number[] }` with parallel arrays. Also export `l2Normalize(sv: SparseVector): SparseVector` that computes `norm = sqrt(sum(v^2))` and divides each value by norm (returns the original if norm is 0). Both arrays must have the same length. Empty input produces `{ indices: [], values: [] }`. | Status: not_done

- [ ] **Implement `toMilvusSparse` utility in `src/utils.ts`** -- Export `toMilvusSparse(sv: SparseVector): Record<number, number>` that converts a sparse vector to Milvus's dictionary format `{ [index]: value }`. | Status: not_done

- [ ] **Write sparse vector format tests in `src/__tests__/sparse-vector.test.ts`** -- Verify indices are sorted ascending. Verify all values are positive (no zeros, no negatives). Verify indices are in range [0, vocabSize). Verify indices and values have the same length. Verify empty text produces `{ indices: [], values: [] }`. Verify L2 normalization produces unit-length vector (magnitude ~1.0). Verify L2 normalization can be disabled (values unchanged). Verify `toMilvusSparse` produces correct `Record<number, number>` output. | Status: not_done

---

## Phase 6: BM25 Encoder

- [ ] **Implement `BM25Encoder` class in `src/bm25.ts`** -- Implement the full `BM25Encoder` class with: constructor accepting `BM25Options` with defaults (k1=1.2, b=0.75, indexing='hash', vocabSize=262144, hashSeed=0, oovStrategy='ignore', stemming=false, minTokenLength=2); `fit(documents: string[]): void` that computes N, df map, avgdl, totalTokens, and optionally builds vocabulary; `encode(text: string): SparseVector` that tokenizes, counts term frequencies, computes BM25 score per term, maps terms to indices (summing on collision), and builds sparse vector; `encodeQuery(query: string): SparseVector` that uses IDF-only scoring (no TF, no length normalization), deduplicating query terms; `encodeBatch(texts: string[]): SparseVector[]` that maps encode over each text; `partialFit(documents: string[]): void` that incrementally updates N, df, avgdl; `getStats(): FitStats` returning fitted statistics. Throw `SparseEncodeError('NOT_FITTED')` if encode/encodeQuery/serialize/getStats called before fit. Throw `SparseEncodeError('EMPTY_CORPUS')` if fit called with empty array. Throw `SparseEncodeError('NOT_FITTED')` if partialFit called before fit. | Status: not_done

- [ ] **Implement `createBM25` factory function** -- Export a `createBM25(options?: BM25Options): BM25Encoder` function that validates options (k1 in [0,3], b in [0,1], vocabSize > 0) and returns a new `BM25Encoder` instance. Throw `SparseEncodeError('INVALID_OPTIONS')` for invalid parameter values. | Status: not_done

- [ ] **Write BM25 encoder unit tests in `src/__tests__/bm25.test.ts`** -- Test fit computes correct N, avgdl, df for a small known corpus. Test encode produces correct BM25 scores (hand-verified). Test BM25 score increases with tf but saturates. Test BM25 score decreases for longer documents (b > 0). Test BM25 score is length-independent when b = 0. Test encodeQuery uses IDF-only (no TF weighting). Test encodeQuery with repeated terms does not increase score. Test encode and encodeQuery produce different vectors for same text. Test encodeBatch produces same results as individual encode calls. Test encoding before fitting throws NOT_FITTED. Test fitting with empty corpus throws EMPTY_CORPUS. Test partialFit updates statistics correctly. Test partialFit before fit throws NOT_FITTED. Test custom k1 and b values are applied. Test hash-based indexing produces indices in [0, vocabSize). Test vocabulary-based indexing assigns sequential IDs. Test OOV strategies (ignore, hash, error). Test with custom tokenizer. Test with stemming enabled. Test with custom stopwords. Test with stopwords disabled (false). Test empty text returns empty sparse vector. | Status: not_done

---

## Phase 7: TF-IDF Encoder

- [ ] **Implement `TFIDFEncoder` class in `src/tfidf.ts`** -- Implement the full `TFIDFEncoder` class with: constructor accepting `TFIDFOptions` with defaults (tfVariant='log', normalize=true, indexing='hash', vocabSize=262144, hashSeed=0, oovStrategy='ignore', stemming=false, minTokenLength=2); `fit(documents: string[]): void` computing N, df, avgdl, totalTokens, and optionally vocabulary; `encode(text: string): SparseVector` using selected TF variant and IDF, with optional L2 normalization; `encodeQuery(query: string): SparseVector` using IDF-only scoring; `encodeBatch(texts: string[]): SparseVector[]`; `partialFit(documents: string[]): void`; `getStats(): FitStats`. Same error handling as BM25Encoder. | Status: not_done

- [ ] **Implement `createTFIDF` factory function** -- Export `createTFIDF(options?: TFIDFOptions): TFIDFEncoder` that validates options (tfVariant is one of 'raw'/'log'/'augmented', normalize is boolean, vocabSize > 0) and returns a new `TFIDFEncoder`. Throw `SparseEncodeError('INVALID_OPTIONS')` for invalid values. | Status: not_done

- [ ] **Write TF-IDF encoder unit tests in `src/__tests__/tfidf.test.ts`** -- Test fit computes correct statistics. Test raw TF variant returns raw counts as TF component. Test log TF variant returns `1 + log(f)`. Test augmented TF variant returns `0.5 + 0.5 * f / maxF`. Test L2 normalization produces unit-length vector. Test `normalize: false` skips L2 normalization. Test encodeQuery uses IDF-only. Test encodeBatch matches individual encode calls. Test all error cases (not fitted, empty corpus). Test with each TF variant. Test with custom tokenizer. Test with stemming. | Status: not_done

---

## Phase 8: Serialization

- [ ] **Implement serialize/deserialize in `src/serialization.ts`** -- Implement `serializeBM25(encoder): string` that produces JSON with structure: `{ type: "bm25", version: 1, params: { k1, b }, corpus: { N, avgdl, df: object }, indexing: { type, vocabSize, hashSeed } or { type: "vocab", vocab: object }, tokenizer: "default" | "custom", tokenizerConfig: { stemming, stopwords, minTokenLength, additionalStopwords } }`. Implement `deserializeBM25(json: string, options?: { tokenizer?: TokenizerFn }): BM25Encoder` that restores the encoder from JSON. Implement equivalent `serializeTFIDF`/`deserializeTFIDF` with additional params (tfVariant, normalize). When `tokenizer: "custom"` is recorded, require the caller to pass the tokenizer function during deserialization. Throw `SparseEncodeError('SERIALIZATION_ERROR')` for corrupted/invalid JSON. Verify version field for forward compatibility. | Status: not_done

- [ ] **Wire serialize/deserialize into encoder classes** -- Add `serialize(): string` instance method on both `BM25Encoder` and `TFIDFEncoder` (throws if not fitted). Add `static deserialize(json: string, options?): BM25Encoder` and `static deserialize(json: string, options?): TFIDFEncoder` static methods. | Status: not_done

- [ ] **Write serialization tests in `src/__tests__/serialization.test.ts`** -- Test round-trip: `deserialize(serialize())` produces an encoder giving identical output for the same input text. Test deserialized encoder produces same sparse vectors as original for multiple inputs. Test deserialization with custom tokenizer provided. Test deserialization without custom tokenizer when model was fitted with custom tokenizer throws or warns. Test corrupted JSON throws SERIALIZATION_ERROR. Test model with unknown version throws. Test both BM25 and TF-IDF serialization round-trips. Test that vocabulary-based indexing serializes and restores the vocabulary. Test partialFit state survives serialization round-trip. | Status: not_done

---

## Phase 9: Query Encoding

- [ ] **Verify query encoding implementation** -- Ensure `encodeQuery` on both BM25Encoder and TFIDFEncoder uses IDF-only weighting: for each unique term in the query, the sparse vector value is `IDF(term)`. Term frequency within the query is NOT used (repeated query terms do not get extra weight). Document length normalization is NOT applied. Duplicate terms in the query produce a single entry. | Status: not_done

- [ ] **Write query encoding tests in `src/__tests__/query-encoding.test.ts`** -- Verify query encoding uses IDF-only weighting. Verify query and document encoding for the same text produce different sparse vectors. Verify repeated terms in a query do not increase the score (deduplicated). Verify empty query returns empty sparse vector. Verify query terms not in corpus still get IDF scores (unseen terms get max IDF). Test query encoding with BM25. Test query encoding with TF-IDF. | Status: not_done

---

## Phase 10: CLI

- [ ] **Implement CLI entry point in `src/cli.ts`** -- Implement CLI using Node.js built-in `process.argv` parsing (no dependency like commander/yargs, keeping zero-dependency constraint). Support four commands: `fit`, `encode`, `query`, `stats`. Parse global flags and per-command options. Display version from package.json. Handle `--help` for each command. | Status: not_done

- [ ] **Implement `sparse-encode fit` command** -- Accept options: `--input <path>` (required, JSONL or JSON array), `--output <path>` (required, path to write model JSON), `--algorithm <bm25|tfidf>` (default bm25), `--k1 <n>`, `--b <n>`, `--tf-variant <raw|log|augmented>`, `--indexing <hash|vocab>`, `--vocab-size <n>`, `--stemming` (flag), `--format <human|json>` (default human). Read corpus from input file, create encoder, call fit, write serialized model to output. Print summary (algorithm, indexing, stemming, corpus stats, model file size) in human or JSON format. Exit code 0 on success, 1 on IO error, 2 on config error. | Status: not_done

- [ ] **Implement `sparse-encode encode` command** -- Accept options: `--model <path>` (required), `--text <text>` (single text, mutually exclusive with --input), `--input <path>` (file with texts, one per line or JSONL, mutually exclusive with --text), `--output <path>` (default stdout), `--format <json|jsonl>` (default json). Deserialize model, encode text(s) as documents, write output. | Status: not_done

- [ ] **Implement `sparse-encode query` command** -- Accept options: `--model <path>` (required), `--text <text>` (required), `--format <json|human>` (default json). Deserialize model, encode query text using `encodeQuery`, write output. | Status: not_done

- [ ] **Implement `sparse-encode stats` command** -- Accept options: `--model <path>` (required), `--format <human|json>` (default human). Deserialize model, call `getStats()`, print statistics (N, avgdl, vocabSize, totalTokens). | Status: not_done

- [ ] **Add CLI binary configuration to `package.json`** -- Add `"bin": { "sparse-encode": "dist/cli.js" }` to package.json. Ensure `cli.ts` has a hashbang (`#!/usr/bin/env node`) at the top. | Status: not_done

- [ ] **Write CLI integration tests** -- Create tests that spawn the CLI as a subprocess and verify: `fit` command reads corpus, writes model file, prints summary, exits 0; `encode` command reads model, encodes text, outputs JSON; `query` command reads model, encodes query, outputs JSON; `stats` command reads model, prints stats; missing required options exit with code 2; invalid model file exits with code 1; `--help` prints usage; `--format json` produces valid JSON output. | Status: not_done

---

## Phase 11: Integration Tests

- [ ] **Write end-to-end BM25 integration test in `src/__tests__/integration/bm25-e2e.test.ts`** -- Fit BM25 on a 100-document corpus, encode a document, verify the highest-scoring terms in the sparse vector are the most distinctive terms (high IDF, moderate TF). Verify the sparse vector format is valid. Verify encodeQuery produces different results from encode for the same text. Verify encodeBatch matches individual encodes. | Status: not_done

- [ ] **Write end-to-end TF-IDF integration test in `src/__tests__/integration/tfidf-e2e.test.ts`** -- Same structure as BM25 E2E test but with TF-IDF encoder. Test all three TF variants (raw, log, augmented). Test with and without L2 normalization. | Status: not_done

- [ ] **Write Pinecone format compatibility test in `src/__tests__/integration/pinecone-compat.test.ts`** -- Encode a document and verify the output `SparseVector` has the structure Pinecone expects: `{ indices: number[], values: number[] }` with indices as integers, values as floats, indices sorted ascending, all values positive. Verify it can be used directly in a Pinecone upsert payload structure (type-level check). | Status: not_done

- [ ] **Create reference vectors fixture for regression testing** -- Create `src/__tests__/fixtures/reference-vectors.json` containing known-good sparse vectors for specific inputs with specific encoder configurations. Use these in regression tests to detect unintentional changes to scoring or hashing behavior. | Status: not_done

- [ ] **Write batch encoding consistency test** -- Verify `encodeBatch(texts)` returns identical results to `texts.map(t => encoder.encode(t))` for both BM25 and TF-IDF encoders across various configurations. | Status: not_done

---

## Phase 12: Performance Benchmarks

- [ ] **Create benchmark script in `src/__benchmarks__/encode-throughput.ts`** -- Implement benchmarks measuring: (1) fit throughput -- time to fit on 10,000 documents of ~500 tokens each, target under 2 seconds; (2) encode throughput -- time to encode 10,000 documents after fitting, target under 1 second (>10K docs/sec); (3) query encode throughput -- time to encode 10,000 queries, target under 500ms (>20K queries/sec); (4) serialization round-trip -- time to serialize and deserialize a model fitted on 100,000 documents, target under 500ms. Print results in a human-readable format. | Status: not_done

- [ ] **Add benchmark npm script** -- Add `"bench": "npx tsx src/__benchmarks__/encode-throughput.ts"` script to package.json for easy benchmark execution. | Status: not_done

---

## Phase 13: Option Validation and Edge Cases

- [ ] **Validate BM25Options on construction** -- Verify k1 is a number in [0, 3] (throw INVALID_OPTIONS otherwise). Verify b is a number in [0, 1]. Verify vocabSize is a positive integer. Verify indexing is 'hash' or 'vocab'. Verify oovStrategy is 'ignore', 'hash', or 'error'. Verify minTokenLength is a positive integer. Verify hashSeed is a number. | Status: not_done

- [ ] **Validate TFIDFOptions on construction** -- Same as BM25 validation plus: verify tfVariant is 'raw', 'log', or 'augmented'. Verify normalize is a boolean. | Status: not_done

- [ ] **Handle edge case: document with only stopwords** -- When a document consists entirely of stopwords (all tokens removed), `encode` should return `{ indices: [], values: [] }` (empty sparse vector), not throw. | Status: not_done

- [ ] **Handle edge case: very long documents** -- Ensure encoding works correctly for documents with thousands of tokens. No memory leaks, no stack overflows. BM25 length normalization should handle documents much longer than avgdl gracefully. | Status: not_done

- [ ] **Handle edge case: single-document corpus** -- When fit is called with exactly one document, N=1, avgdl = document length. IDF for all terms in that document should be well-defined (df=1, N=1). BM25 scores should still be valid. | Status: not_done

- [ ] **Handle edge case: duplicate documents in corpus** -- When the corpus contains identical documents, fit should count each occurrence separately (N includes duplicates, df counts are correct). | Status: not_done

- [ ] **Handle edge case: hash collision summing** -- When two different terms hash to the same index via MurmurHash3, their BM25 or TF-IDF scores should be summed at that index. Verify this with a test that forces a collision. | Status: not_done

- [ ] **Handle edge case: unseen terms during encoding** -- Terms that appear in a document being encoded but were never seen during fit should get maximum IDF (treated as maximally informative) for hash-based indexing. For vocabulary-based indexing, they follow the oovStrategy. | Status: not_done

---

## Phase 14: Documentation

- [ ] **Write README.md** -- Create a comprehensive README with: package description, installation instructions (`npm install sparse-encode`), quickstart examples (BM25 and TF-IDF), API reference for all public functions and types, configuration options tables (BM25 and TF-IDF), CLI usage with all commands and options, integration examples (Pinecone, Qdrant, Milvus), integration with `embed-cache` and `fusion-rank`, explanation of document vs query encoding, explanation of hash vs vocabulary indexing, performance characteristics, and license. | Status: not_done

- [ ] **Add JSDoc comments to all public API exports** -- Add comprehensive JSDoc comments to: `createBM25`, `createTFIDF`, `BM25Encoder` (all methods), `TFIDFEncoder` (all methods), `toMilvusSparse`, all interfaces and types in `types.ts`, `SparseEncodeError`. Include `@param`, `@returns`, `@throws`, and `@example` tags where appropriate. | Status: not_done

---

## Phase 15: Build, Lint, and Publish Preparation

- [ ] **Verify TypeScript compilation** -- Run `npm run build` (`tsc`) and verify it compiles without errors. Verify `dist/` output contains `index.js`, `index.d.ts`, `bm25.js`, `tfidf.js`, `cli.js`, and all other modules with corresponding `.d.ts` and `.js.map` files. | Status: not_done

- [ ] **Configure ESLint** -- Set up ESLint configuration for the project (e.g., `.eslintrc.json` or `eslint.config.js`). Run `npm run lint` and fix any issues. Ensure lint passes cleanly. | Status: not_done

- [ ] **Run full test suite** -- Run `npm run test` (`vitest run`) and verify all unit tests, integration tests, and CLI tests pass. Fix any failures. | Status: not_done

- [ ] **Verify package.json metadata** -- Ensure `name`, `version`, `description`, `main`, `types`, `files`, `bin`, `engines`, `keywords`, `license`, `publishConfig` are all correctly set. Add relevant keywords: `["bm25", "tfidf", "sparse-vector", "hybrid-search", "pinecone", "qdrant", "milvus", "information-retrieval", "nlp"]`. | Status: not_done

- [ ] **Bump version to target version** -- Following the implementation roadmap, ensure version is bumped appropriately as phases complete (0.1.0 for Phase 1 core, incrementing through to 0.5.0 for the final polished release). | Status: not_done

- [ ] **Add `.gitignore` entries** -- Ensure `dist/`, `node_modules/`, and any other build artifacts are gitignored. | Status: not_done

- [ ] **Verify `prepublishOnly` hook** -- Confirm `npm run build` runs automatically before `npm publish` via the `prepublishOnly` script already in package.json. | Status: not_done
