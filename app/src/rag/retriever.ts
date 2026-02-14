/**
 * Retriever — wraps a VectorStore query with optional score filtering.
 */

import type { VectorStore, QueryResult } from "../ingestion/store.js";

export interface RetrieveOptions {
  topK?: number;
  minScore?: number;
}

/**
 * Retrieve relevant chunks from the vector store.
 *
 * Queries the store for the top-K most similar documents to `query`,
 * optionally filtering out results below `minScore`. Results are
 * returned sorted by score descending.
 */
export async function retrieve(
  query: string,
  store: VectorStore,
  options: RetrieveOptions = {},
): Promise<QueryResult[]> {
  const { topK = 5, minScore } = options;

  const results = await store.query(query, { topK });

  const filtered =
    minScore !== undefined
      ? results.filter((r) => r.score >= minScore)
      : results;

  return filtered.sort((a, b) => b.score - a.score);
}
