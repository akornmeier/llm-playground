/**
 * Document ingestion pipeline.
 *
 * Wires the chunker and vector store together: accepts raw documents,
 * chunks them, and stores the resulting chunks for later retrieval.
 */

import { chunkDocument, type ChunkOptions } from "./chunker.js";
import { type VectorStore } from "./store.js";

/**
 * Ingest one or more documents into the vector store.
 *
 * Each document is chunked according to `options` (defaults to
 * section-aware splitting) and the resulting chunks are added to `store`.
 *
 * @returns The total number of chunks ingested.
 */
export async function ingestDocuments(
  documents: Array<{ text: string; metadata: Record<string, string> }>,
  store: VectorStore,
  options?: ChunkOptions,
): Promise<number> {
  const defaultOptions: ChunkOptions = {
    strategy: "section-aware",
    ...options,
  };

  let totalChunks = 0;

  for (const doc of documents) {
    const chunks = chunkDocument(doc.text, {
      ...defaultOptions,
      metadata: { ...doc.metadata, ...defaultOptions.metadata },
    });

    await store.add(
      chunks.map((chunk) => ({
        content: chunk.content,
        metadata: chunk.metadata,
      })),
    );

    totalChunks += chunks.length;
  }

  return totalChunks;
}
