/**
 * CLI entry point — minimal REPL for testing the LegalAIPipeline end-to-end.
 *
 * Usage:
 *   pnpm dev
 *
 * Loads sample court opinions from datasets/sample/court_opinions.jsonl,
 * ingests them into the pipeline, then accepts questions interactively.
 */

import { LegalAIPipeline } from "./pipeline.js";
import * as readline from "node:readline";
import * as fs from "node:fs";
import * as path from "node:path";

/**
 * A mock LLM provider that echoes the user prompt back.
 * Replace with a real provider (e.g. @ai-sdk/anthropic) for production use.
 */
function mockProvider(prompt: {
  system: string;
  user: string;
}): Promise<{ text: string }> {
  return Promise.resolve({
    text: `[Mock LLM] Received question: "${prompt.user}". Based on the provided sources, this is a placeholder answer. Please configure a real LLM provider for meaningful responses.`,
  });
}

async function main(): Promise<void> {
  const pipeline = new LegalAIPipeline({ provider: mockProvider });

  // Load sample data
  const dataPath = path.resolve(
    import.meta.dirname ?? ".",
    "../../datasets/sample/court_opinions.jsonl",
  );

  if (fs.existsSync(dataPath)) {
    const lines = fs
      .readFileSync(dataPath, "utf-8")
      .split("\n")
      .filter((line) => line.trim().length > 0);

    const documents = lines.map((line) => {
      const parsed = JSON.parse(line) as {
        id: number;
        case_name: string;
        court: string;
        date_filed: string;
        text: string;
      };
      return {
        text: parsed.text,
        metadata: {
          id: String(parsed.id),
          case_name: parsed.case_name,
          court: parsed.court,
          date_filed: parsed.date_filed,
        },
      };
    });

    const chunks = await pipeline.ingest(documents);
    console.log(`Ingested ${documents.length} documents (${chunks} chunks).`);
  } else {
    console.log(
      `Sample data not found at ${dataPath}. Starting with empty store.`,
    );
  }

  // REPL
  const rl = readline.createInterface({
    input: process.stdin,
    output: process.stdout,
  });

  console.log("\nLegal AI Pipeline — type a question or 'quit' to exit.\n");

  const prompt = (): void => {
    rl.question("> ", async (input) => {
      const trimmed = input.trim();
      if (
        trimmed.toLowerCase() === "quit" ||
        trimmed.toLowerCase() === "exit"
      ) {
        rl.close();
        return;
      }

      if (trimmed.length === 0) {
        prompt();
        return;
      }

      try {
        const result = await pipeline.ask(trimmed);

        console.log(`\nAnswer: ${result.answer}`);
        console.log(`\nSources: ${result.sources.length}`);
        console.log(
          `Grounding: ${result.groundingReport.citations.length} citations, ` +
            `accuracy=${result.groundingReport.citationAccuracy.toFixed(2)}, ` +
            `hallucination=${result.groundingReport.hallucinationRate.toFixed(2)}`,
        );
        console.log(
          `Guardrails: inScope=${result.guardrailReport.inputFilter.inScope}, ` +
            `warnings=[${result.guardrailReport.outputFilter.warnings.join(", ")}]`,
        );
        console.log();
      } catch (err) {
        console.error("Error:", err);
      }

      prompt();
    });
  };

  prompt();
}

main().catch(console.error);
