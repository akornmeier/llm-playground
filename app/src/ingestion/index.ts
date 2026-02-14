/**
 * Document ingestion module — barrel export.
 */

export { chunkDocument, type Chunk, type ChunkOptions } from "./chunker.js";
export { embed } from "./embedder.js";
export { VectorStore, type StoreDocument, type QueryResult } from "./store.js";
export { ingestDocuments } from "./ingest.js";
