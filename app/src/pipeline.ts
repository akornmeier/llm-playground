/**
 * LegalAIPipeline — wires all app stages together into a single entry point.
 *
 * Stages:
 *   1. Input guardrails (PII detection, scope filtering)
 *   2. Document retrieval from the vector store
 *   3. Answer generation via an LLM provider
 *   4. Citation grounding verification
 *   5. Output guardrails (legal advice detection)
 *   6. Evaluation metrics (citation accuracy, hallucination rate)
 *   7. Conversation history tracking
 */

import { VectorStore, ingestDocuments } from "./ingestion/index.js";
import { retrieve } from "./rag/retriever.js";
import { generateAnswer } from "./rag/generate.js";
import { checkGrounding } from "./verification/groundingChecker.js";
import { filterInput } from "./guardrails/inputFilter.js";
import { filterOutput } from "./guardrails/outputFilter.js";
import { citationAccuracy, hallucinationRate } from "./eval/metrics.js";
import { ConversationManager } from "./conversation/manager.js";
import type { LLMProvider } from "./rag/generate.js";
import type { InputFilterResult } from "./guardrails/inputFilter.js";
import type { OutputFilterResult } from "./guardrails/outputFilter.js";

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

export interface PipelineOptions {
  provider: LLMProvider;
  maxTokens?: number;
  systemMessage?: string;
}

export interface PipelineResult {
  answer: string;
  sources: Array<{ content: string; metadata: Record<string, string> }>;
  groundingReport: {
    citations: Array<{
      citation: { type: string; text: string };
      grounded: boolean;
    }>;
    hasUnverifiedCitations: boolean;
    citationAccuracy: number;
    hallucinationRate: number;
  };
  guardrailReport: {
    inputFilter: InputFilterResult;
    outputFilter: OutputFilterResult;
  };
}

// ---------------------------------------------------------------------------
// Pipeline
// ---------------------------------------------------------------------------

export class LegalAIPipeline {
  private store: VectorStore;
  private conversation: ConversationManager;
  private provider: LLMProvider;

  constructor(options: PipelineOptions) {
    this.store = new VectorStore();
    this.provider = options.provider;
    this.conversation = new ConversationManager({
      maxTokens: options.maxTokens ?? 4096,
      systemMessage: options.systemMessage,
    });
  }

  /**
   * Ingest documents into the vector store for later retrieval.
   *
   * @returns The total number of chunks stored.
   */
  async ingest(
    documents: Array<{ text: string; metadata: Record<string, string> }>,
  ): Promise<number> {
    return ingestDocuments(documents, this.store);
  }

  /**
   * Process a question through the full pipeline:
   *
   *   input guardrails -> retrieve -> generate -> grounding -> output guardrails -> metrics
   *
   * Even if the input is out-of-scope the question is still forwarded to the
   * provider; the guardrail result is included in the report so callers can
   * decide how to handle it.
   */
  async ask(question: string): Promise<PipelineResult> {
    // 1. Input guardrails
    const inputFilter = filterInput(question);

    // 2. Retrieve relevant chunks
    const chunks = await retrieve(question, this.store);

    // 3. Generate answer
    const sourceChunks = chunks.map((c) => ({
      content: c.content,
      metadata: c.metadata,
      score: c.score,
    }));
    const generated = await generateAnswer(
      question,
      sourceChunks,
      this.provider,
    );

    // 4. Check citation grounding
    const grounding = await checkGrounding(generated.text, this.store);

    // 5. Output guardrails
    const outputFilter = filterOutput(generated.text);

    // 6. Compute evaluation metrics
    const accuracy = citationAccuracy(grounding);
    const hallucination = hallucinationRate(grounding);

    // 7. Track conversation history
    this.conversation.addMessage({ role: "user", content: question });
    this.conversation.addMessage({
      role: "assistant",
      content: generated.text,
    });

    // 8. Build and return the full result
    return {
      answer: generated.text,
      sources: generated.sources,
      groundingReport: {
        citations: grounding.citations,
        hasUnverifiedCitations: grounding.hasUnverifiedCitations,
        citationAccuracy: accuracy,
        hallucinationRate: hallucination,
      },
      guardrailReport: {
        inputFilter,
        outputFilter,
      },
    };
  }
}
