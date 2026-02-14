import { describe, it, expect } from "vitest";
import { ConversationManager } from "../manager.js";

describe("ConversationManager", () => {
  it("returns all messages when within token budget", () => {
    const manager = new ConversationManager({ maxTokens: 10000 });
    manager.addMessage({ role: "user", content: "What is habeas corpus?" });
    manager.addMessage({ role: "assistant", content: "A legal remedy." });
    const context = manager.getContext();
    expect(context.messages).toHaveLength(2);
  });

  it("truncates older messages when exceeding token budget", () => {
    const manager = new ConversationManager({ maxTokens: 20 });
    for (let i = 0; i < 10; i++) {
      manager.addMessage({
        role: "user",
        content: `Question ${i} about legal precedent and detailed case law analysis`,
      });
      manager.addMessage({
        role: "assistant",
        content: `Answer ${i} with comprehensive legal analysis and multiple citations`,
      });
    }
    const context = manager.getContext();
    expect(context.estimatedTokens).toBeLessThanOrEqual(20);
    // Most recent messages should be preserved
    const lastMessage = context.messages[context.messages.length - 1];
    expect(lastMessage.content).toContain("9");
  });

  it("includes system message in every context", () => {
    const manager = new ConversationManager({
      maxTokens: 500,
      systemMessage: "You are a legal assistant.",
    });
    manager.addMessage({ role: "user", content: "Hello" });
    const context = manager.getContext();
    expect(context.messages[0].role).toBe("system");
    expect(context.messages[0].content).toContain("legal assistant");
  });

  it("never drops the system message even with tight budget", () => {
    const manager = new ConversationManager({
      maxTokens: 50,
      systemMessage:
        "You are a legal assistant specializing in constitutional law.",
    });
    for (let i = 0; i < 20; i++) {
      manager.addMessage({ role: "user", content: `Long question ${i}` });
      manager.addMessage({ role: "assistant", content: `Long answer ${i}` });
    }
    const context = manager.getContext();
    expect(context.messages[0].role).toBe("system");
  });

  it("preserves most recent messages when truncating", () => {
    const manager = new ConversationManager({ maxTokens: 30 });
    manager.addMessage({
      role: "user",
      content: "Old question from earlier in the conversation",
    });
    manager.addMessage({
      role: "assistant",
      content: "Old answer from earlier",
    });
    manager.addMessage({ role: "user", content: "Recent question" });
    manager.addMessage({ role: "assistant", content: "Recent answer" });
    const context = manager.getContext();
    // Most recent should be present
    expect(context.messages[context.messages.length - 1].content).toBe(
      "Recent answer",
    );
  });
});
