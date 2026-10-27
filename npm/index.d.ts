/* auto-generated TS types — mirrors crates/leann-node/src/lib.rs exactly */

export interface EmbedderOptions {
  /** "openai" (default) or "local-hash" */
  kind?: string
  baseUrl?: string
  apiKey?: string
  model?: string
  dim?: number
  batchSize?: number
  concurrency?: number
  maxRetries?: number
  sendDimensionsParam?: boolean
}

export interface LlmOptions {
  /** "openai" (default) or "anthropic" */
  kind?: string
  baseUrl?: string
  apiKey?: string
  model?: string
  temperature?: number
  maxTokens?: number
}

export interface TranscriberOptions {
  baseUrl?: string
  apiKey?: string
  model?: string
  language?: string
}

export interface ChunkerOptions {
  chunkTokens?: number
  overlapTokens?: number
  minTokens?: number
  maxTokens?: number
}

export interface LeannOptions {
  embedder: EmbedderOptions
  llm?: LlmOptions
  transcriber?: TranscriberOptions
  /** "cosine" (default) | "l2" | "ip" */
  metric?: string
  chunker?: ChunkerOptions
  cacheCapacity?: number
  extraMetadata?: unknown
}

export interface SearchArgs {
  k?: number
  ef?: number
  alpha?: number
  minBatch?: number
  maxBatch?: number
  relax?: number
  exact?: boolean
  withText?: boolean
  withMetadata?: boolean
  /** JSON filter, e.g. { op: "eq", field: "format", value: "pdf" } */
  filter?: unknown
}

export interface HybridArgs {
  k?: number
  /** "rrf" (default) or "weighted" */
  fusion?: string
  vectorWeight?: number
  lexicalWeight?: number
  vectorOverfetch?: number
  lexicalK?: number
  expandBefore?: number
  expandAfter?: number
  filter?: unknown
  withText?: boolean
}

export interface ChatArgs {
  topK?: number
  useHybrid?: boolean
  expandBefore?: number
  expandAfter?: number
  filter?: unknown
  systemPrompt?: string
  temperature?: number
  maxTokens?: number
  maxContextChars?: number
  /** verify the answer with the Guardian (IBM Granite Guardian method) */
  guard?: GuardArgs
  /** Reset conversation memory before this question */
  fresh?: boolean
}

export interface GuardArgs {
  /** minimum groundedness confidence (0-1) to accept; default 0.7 */
  groundednessThreshold?: number
  /** repair loop attempts when checks fail; default 1 */
  repairAttempts?: number
  /** reject/flag harm on prompt and answer; default true */
  harmCheck?: boolean
  /** policy: "abstain" (default) | "annotate" | "log" */
  policy?: string
}

export interface Hit {
  id: number
  distance: number
  similarity: number
  text?: string
  metadata?: unknown
}

export interface SearchStatsJs {
  hops: number
  pqScored: number
  promoted: number
  embedCalls: number
  vectorsEmbedded: number
  embedSecs: number
  tookMs: number
}

export interface SearchResultJs {
  hits: Array<Hit>
  stats: SearchStatsJs
}

export interface HybridHit {
  id: number
  fusedScore: number
  vectorRank?: number
  lexicalRank?: number
  vectorSimilarity?: number
  lexicalScore?: number
  text?: string
  metadata?: unknown
}

export interface ExpansionJs {
  primary: number
  neighbors: Array<number>
  offsets: Array<number>
}

export interface HybridResultJs {
  hits: Array<HybridHit>
  expanded: Array<ExpansionJs>
  tookMs: number
  stats: SearchStatsJs
}

export interface SourceRefJs {
  chunkId: number
  source?: string
  score?: number
  text?: string
  metadata?: unknown
}

export interface ChatResultJs {
  answer: string
  sources: Array<SourceRefJs>
  query: string
  /** present when guard verification ran */
  guard?: GuardReportJs
}

export interface GuardReportJs {
  riskTriggered: boolean
  risks: Array<string>
  groundedness?: number
  contextRelevance?: number
  answerRelevance?: number
  harm?: boolean
  trace?: string
  action: string
  repaired: boolean
}

export interface AddResult {
  chunkIds: Array<number>
  chunksAdded: number
  totalChunks: number
}

export interface StatsJs {
  chunks: number
  alive: number
  deleted: number
  dim: number
  metric: string
  pqM: number
  pqCompression: number
  textBytes: number
  codesBytes: number
  graphEdges: number
  avgDegree: number
  conventionalVectorBytes: number
  storageSavedPercent: number
  bm25Vocab: number
  embedder: string
  fileBytes?: number
}

export interface VerifiedSentenceJs {
  text: string
  grounded: boolean
  confidence: number
  supporting: Array<number>
}

export interface GuardianVerdictJs {
  riskId: string
  triggered: boolean
  confidence: number
  trace?: string
  explanation?: string
}

export declare class Leann {
  constructor(opts: LeannOptions)
  static open(path: string, opts: LeannOptions): Leann
  addFiles(paths: Array<string>, args?: IngestArgs): Promise<AddResult>
  addTexts(texts: Array<string>, metadatas?: Array<any>): Promise<AddResult>
  search(query: string, args?: SearchArgs): Promise<SearchResultJs>
  hybridSearch(query: string, args?: HybridArgs): Promise<HybridResultJs>
  chat(query: string, args?: ChatArgs): Promise<ChatResultJs>
  clearHistory(): void
  expandContext(ids: Array<number>, before?: number, after?: number): Array<ExpansionJs>
  deleteChunks(ids: Array<number>): number
  deleteWhere(filter: any): number
  getChunk(id: number): any
  save(path: string): void
  stats(): StatsJs
  /** IBM Granite Guardian sentence-level grounding audit of any answer */
  verifyAnswer(answer: string, context: Array<string>): Array<VerifiedSentenceJs>
}

export interface IngestArgs {
  chunker?: ChunkerOptions
  extraMetadata?: unknown
}

export declare class Collections {
  constructor(root: string)
  open(user: string, name: string, opts: LeannOptions): Leann
  list(user: string): Array<string>
  save(user: string, name: string, leann: Leann): void
  delete(user: string, name: string): boolean
}

/** Standalone IBM Granite Guardian judge (works with any OpenAI-compatible endpoint, incl. served granite-guardian models). */
export declare class Guardian {
  constructor(opts: GuardianOptions)
  checkGroundedness(response: string, context: Array<string>): Promise<GuardianVerdictJs>
  checkContextRelevance(query: string, context: Array<string>): Promise<GuardianVerdictJs>
  checkAnswerRelevance(query: string, response: string): Promise<GuardianVerdictJs>
  checkHarm(text: string): Promise<GuardianVerdictJs>
  checkJailbreak(prompt: string): Promise<GuardianVerdictJs>
  checkCustom(text: string, criteria: string, context?: Array<string>): Promise<GuardianVerdictJs>
}

export interface GuardianOptions {
  baseUrl?: string
  apiKey?: string
  /** e.g. "ibm-granite/granite-guardian-3.3-8b" served via vLLM/Ollama, or "gpt-4o-mini" as a generic judge */
  model?: string
  /** reasoning mode: true = IMD trace, false (default) = fast no-think */
  think?: boolean
  temperature?: number
  maxRetries?: number
}

export declare function version(): string
export declare function selfTest(): Promise<boolean>
