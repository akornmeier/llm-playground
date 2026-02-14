import { describe, it, expect } from "vitest";
import { ConversationHistory } from "../history.js";

describe("ConversationHistory", () => {
  it("tracks messages in order", () => {
    const history = new ConversationHistory();
    history.add({ role: "user", content: "What is habeas corpus?" });
    history.add({
      role: "assistant",
      content: "Habeas corpus is a legal remedy...",
    });
    expect(history.messages).toHaveLength(2);
    expect(history.messages[0].role).toBe("user");
    expect(history.messages[1].role).toBe("assistant");
  });

  it("estimates token count using word-based heuristic", () => {
    const history = new ConversationHistory();
    history.add({ role: "user", content: "Short question" });
    expect(history.estimatedTokens).toBeGreaterThan(0);
  });

  it("provides token estimate proportional to content length", () => {
    const short = new ConversationHistory();
    short.add({ role: "user", content: "Hi" });

    const long = new ConversationHistory();
    long.add({
      role: "user",
      content:
        "This is a much longer message about legal precedent and constitutional law",
    });

    expect(long.estimatedTokens).toBeGreaterThan(short.estimatedTokens);
  });

  it("clears history", () => {
    const history = new ConversationHistory();
    history.add({ role: "user", content: "test" });
    history.clear();
    expect(history.messages).toHaveLength(0);
    expect(history.estimatedTokens).toBe(0);
  });
});
