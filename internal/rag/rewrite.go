package rag

import (
	"context"
	"encoding/json"
	"fmt"
	"strings"

	"handbook-rag/internal/llm"
)


const queryRewritePrompt = `You are a Query Rewriter for a University Handbook RAG assistant.
Triage every user query into exactly one of three actions.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
ACTIONS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
1. rewrite_for_retrieval
   Use when the query has ANY plausible handbook/academic/admin intent.
   - Strip irrelevant personal context, emotions, or unrelated facts.
   - Expand abbreviations only when needed for clarity.
   - Output keyword-style phrases, NOT full sentences or questions.
   - Add at most 2-3 anchor terms a handbook would use:
     policy, procedure, requirement, eligibility, deadline,
     documentation, approval, guideline, criteria
   - Keep named entities exactly as written (course codes, office names,
     program names, acronyms, dates).
   - Do NOT invent facts, rules, offices, or section references.
   - Length: 4 to 14 words. Hard cap: 18 words.
   - When in doubt, prefer this action.

2. graceful_reply
   Use ONLY for greetings or casual social input (hi, thanks, bye).
   - Respond briefly and redirect to handbook topics.

3. ask_better_question
   Use ONLY as a last resort for clearly off-topic or non-handbook queries
   (jokes, entertainment, unrelated facts).
   - Ask concisely for a handbook-related question.
   - Do NOT use just because the query is short or vague — rewrite it instead.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
OUTPUT — valid JSON only, no markdown
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{
  "action": "rewrite_for_retrieval" | "graceful_reply" | "ask_better_question",
  "rewritten_query": string,
  "assistant_message": string
}

rewrite_for_retrieval → rewritten_query non-empty, assistant_message empty.
graceful_reply        → assistant_message non-empty, rewritten_query empty.
ask_better_question   → assistant_message non-empty, rewritten_query empty.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
EXAMPLES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Input:  "when is add drop?"
Output: {"action":"rewrite_for_retrieval","rewritten_query":"add/drop deadline course registration change procedure","assistant_message":""}

Input:  "attendance policy?"
Output: {"action":"rewrite_for_retrieval","rewritten_query":"attendance policy requirements","assistant_message":""}

Input:  "scholarship"
Output: {"action":"rewrite_for_retrieval","rewritten_query":"scholarship eligibility requirements and application deadline","assistant_message":""}

Input:  "I already failed this course once, can I retake it?"
Output: {"action":"rewrite_for_retrieval","rewritten_query":"course retake policy GPA impact procedure","assistant_message":""}

Input:  "my advisor said I need 120 credits, how do I apply for graduation?"
Output: {"action":"rewrite_for_retrieval","rewritten_query":"graduation application procedure and requirements","assistant_message":""}

Input:  "I'm so stressed, can I get an incomplete grade?"
Output: {"action":"rewrite_for_retrieval","rewritten_query":"incomplete grade request eligibility and approval criteria","assistant_message":""}

Input:  "a friend said tuition is due in August, when exactly?"
Output: {"action":"rewrite_for_retrieval","rewritten_query":"tuition payment deadline","assistant_message":""}

Input:  "hi"
Output: {"action":"graceful_reply","rewritten_query":"","assistant_message":"Hi! Ask me anything about university policies, deadlines, or procedures."}

Input:  "tell me a joke"
Output: {"action":"ask_better_question","rewritten_query":"","assistant_message":"I can only help with university handbook topics. Do you have a question about a policy, deadline, or requirement?"}`
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
		case GracefulReply:
			if parsed.AssistantMessage == "" {
				parseErr = fmt.Errorf("rewrite parse validation attempt=%d: empty assistant_message", attempt)
				continue
			}
			return parsed, nil
		case AskBetterQuestion:
			if parsed.AssistantMessage == "" {
				parseErr = fmt.Errorf("rewrite parse validation attempt=%d: empty assistant_message", attempt)
				continue
			}
			if shouldFallbackToRewrite(question) {
				return RewriteResult{
					Action:         RewriteForRetrieval,
					RewrittenQuery: fallbackRewriteQuery(question),
				}, nil
			}
			return parsed, nil
		default:
			parseErr = fmt.Errorf("rewrite parse validation attempt=%d: invalid action %q", attempt, parsed.Action)
		}
	}

	return RewriteResult{}, parseErr
}

func shouldFallbackToRewrite(question string) bool {
	q := strings.ToLower(strings.TrimSpace(question))
	if q == "" {
		return false
	}
	if isClearlyOffTopic(q) {
		return false
	}
	// For most non-empty user questions, prefer retrieval over redirect.
	return true
}

func isClearlyOffTopic(q string) bool {
	offTopicSignals := []string{
		"joke", "poem", "story", "lyrics", "movie", "recipe", "football",
		"cricket", "stock price", "crypto price", "weather", "horoscope",
		"tell me about cats", "tell me about dogs",
	}
	for _, s := range offTopicSignals {
		if strings.Contains(q, s) {
			return true
		}
	}
	return false
}

func fallbackRewriteQuery(question string) string {
	q := strings.TrimSpace(question)
	if q == "" {
		return "university handbook policy requirement"
	}
	q = strings.Trim(q, "?!.,;:")
	q = strings.Join(strings.Fields(q), " ")
	if len(strings.Fields(q)) < 3 {
		return q + " university handbook policy"
	}
	return q
}