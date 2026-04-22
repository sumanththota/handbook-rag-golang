package rag

import (
	"context"
	"encoding/json"
	"fmt"
	"strings"

	"handbook-rag/internal/llm"
)


const queryRewritePrompt = `You are a Query Rewriter for a University Handbook RAG assistant.
Your job is to TRIAGE the user query into one of three actions:
1) rewrite_for_retrieval
2) graceful_reply
3) ask_better_question

═══════════════════════════════════
WHAT YOU MUST DO
═══════════════════════════════════
1. For rewrite_for_retrieval:
   - Preserve the user's exact intent — do not change what they are asking.
   - Expand abbreviations and shorthand into their full handbook-standard form.
   - Add only synonyms that a handbook would actually use for the same concept.
   Safe additions: policy, procedure, requirement, eligibility, deadline,
                   documentation, approval, guideline, process, criteria
2. Keep all named entities exactly as written:
   course codes, office names, program names, person names, dates, acronyms.
3. For greetings or casual social input (e.g. "hi", "hello", "thanks"), return
   action=graceful_reply with a short friendly response.
4. For non-handbook or vague/teasing/low-information queries, return
   action=ask_better_question with a concise prompt asking for a clearer,
   handbook-related question.

═══════════════════════════════════
WHAT YOU MUST NOT DO
═══════════════════════════════════
- Do NOT invent handbook facts, offices, rules, dates, or section references.
- Do NOT output multiple actions.
- Do NOT include markdown or code fences.

═══════════════════════════════════
OUTPUT — valid JSON only, no markdown
═══════════════════════════════════
{
  "action": "rewrite_for_retrieval" | "graceful_reply" | "ask_better_question",
  "rewritten_query": string,
  "assistant_message": string
}

- rewrite_for_retrieval:
  - rewritten_query must be non-empty
  - assistant_message must be empty
- graceful_reply:
  - assistant_message must be non-empty
  - rewritten_query must be empty
- ask_better_question:
  - assistant_message must be non-empty and ask for a clearer handbook question
  - rewritten_query must be empty

═══════════════════════════════════
EXAMPLES
═══════════════════════════════════

Input:  "when is add drop?"
Output: {"action":"rewrite_for_retrieval","rewritten_query":"What is the add/drop period deadline, registration change procedure, and timeline for dropping or adding a course?","assistant_message":""}

Input: "hi"
Output: {"action":"graceful_reply","rewritten_query":"","assistant_message":"Hi! I can help with questions from the university handbook. Ask me about policies, deadlines, requirements, or procedures."}

Input: "say a joke about cats"
Output: {"action":"ask_better_question","rewritten_query":"","assistant_message":"I can help with university handbook topics. Please ask a specific question about policies, procedures, deadlines, or requirements."}`

type RewriteAction string

const (
	RewriteForRetrieval RewriteAction = "rewrite_for_retrieval"
	GracefulReply       RewriteAction = "graceful_reply"
	AskBetterQuestion   RewriteAction = "ask_better_question"
)

type RewriteResult struct {
	Action           RewriteAction `json:"action"`
	RewrittenQuery   string        `json:"rewritten_query"`
	AssistantMessage string        `json:"assistant_message"`
}

func rewriteQueryWithLLM(ctx context.Context, client *llm.OpenAICompatibleClient, apiKey, model, question string) (RewriteResult, error) {
	var parseErr error
	for attempt := 1; attempt <= 2; attempt++ {
		content, err := client.Complete(
			ctx,
			apiKey,
			model,
			0.1,
			[]llm.ChatMessage{
				{Role: "system", Content: queryRewritePrompt},
				{Role: "user", Content: question},
			},
		)
		if err != nil {
			return RewriteResult{}, fmt.Errorf("rewrite completion: %w", err)
		}

		var parsed RewriteResult
		if err := json.Unmarshal([]byte(content), &parsed); err != nil {
			parseErr = fmt.Errorf("rewrite parse json attempt=%d: %w", attempt, err)
			continue
		}

		parsed.RewrittenQuery = strings.TrimSpace(parsed.RewrittenQuery)
		parsed.AssistantMessage = strings.TrimSpace(parsed.AssistantMessage)

		switch parsed.Action {
		case RewriteForRetrieval:
			if parsed.RewrittenQuery == "" {
				parseErr = fmt.Errorf("rewrite parse validation attempt=%d: empty rewritten_query", attempt)
				continue
			}
			return parsed, nil
		case GracefulReply, AskBetterQuestion:
			if parsed.AssistantMessage == "" {
				parseErr = fmt.Errorf("rewrite parse validation attempt=%d: empty assistant_message", attempt)
				continue
			}
			return parsed, nil
		default:
			parseErr = fmt.Errorf("rewrite parse validation attempt=%d: invalid action %q", attempt, parsed.Action)
		}
	}

	return RewriteResult{}, parseErr
}