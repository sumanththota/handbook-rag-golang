package rag

import (
	"context"
	"encoding/json"
	"fmt"
	"strings"

	"handbook-rag/internal/llm"
)

const queryRewritePrompt = `You are a Query Rewriter for retrieval augmentation in a University Handbook assistant.

Your only job is to rewrite the user query for better semantic retrieval quality, with semantic clarification and zero intent drift.

STRICT RULES (must follow):
1) Keep user intent unchanged.
2) Do not answer the question.
3) Do not add new requirements, constraints, or assumptions.
4) Expand abbreviations, shorthand, and ambiguous references when possible.
5) Add closely related handbook-style synonyms that improve retrieval match, especially terms like:
   - policy, procedure, requirement, eligibility, deadline, timeline, documentation, approval
6) Preserve all named entities exactly as written (course codes, office names, program names, person names, dates, IDs, acronyms). Do not alter spelling/case for these entities.
7) Do not invent facts, dates, numbers, offices, sections, or rules.
8) Keep output concise and retrieval-focused.
9) If the original query is already clear and retrieval-ready, do not rewrite and set no_rewrite=true.
10) Output valid JSON only (no markdown, no extra text).

Output schema:
{
  "no_rewrite": boolean,
  "rewritten_query": string,
  "notes": string
}`

type rewriteResult struct {
	NoRewrite      bool   `json:"no_rewrite"`
	RewrittenQuery string `json:"rewritten_query"`
}

func rewriteQueryWithLLM(ctx context.Context, client *llm.OpenAICompatibleClient, apiKey, model, question string) (string, bool, error) {
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
		return "", false, fmt.Errorf("rewrite completion: %w", err)
	}

	var parsed rewriteResult
	if err := json.Unmarshal([]byte(content), &parsed); err != nil {
		return "", false, fmt.Errorf("rewrite parse json: %w", err)
	}

	if parsed.NoRewrite {
		return question, true, nil
	}

	rewritten := strings.TrimSpace(parsed.RewrittenQuery)
	if rewritten == "" {
		return "", false, fmt.Errorf("rewrite returned empty rewritten_query")
	}

	return rewritten, false, nil
}