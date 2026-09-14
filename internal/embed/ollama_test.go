package embed

import (
	"context"
	"io"
	"net/http"
	"strconv"
	"strings"
	"testing"
)

// Characterization tests: these lock in OllamaClient's CURRENT behavior
// (progressive-shrink retry loop, context-length detection) as ground truth
// for the Python port to match. See docs/MIGRATION.md §3/§6.

// roundTripFunc lets a test inject mock HTTP responses without a real
// Ollama instance or a real network listener.
type roundTripFunc func(*http.Request) (*http.Response, error)

func (f roundTripFunc) RoundTrip(req *http.Request) (*http.Response, error) {
	return f(req)
}

func newResponse(status int, body string) *http.Response {
	return &http.Response{
		StatusCode: status,
		Body:       io.NopCloser(strings.NewReader(body)),
		Header:     make(http.Header),
	}
}

func newMockClient(rt roundTripFunc) *OllamaClient {
	return &OllamaClient{
		baseURL: "http://mock-ollama",
		client:  &http.Client{Transport: rt},
	}
}

func wordsN(n int) string {
	words := make([]string, n)
	for i := range words {
		words[i] = "w"
	}
	return strings.Join(words, " ")
}

// --- shrinkText ---

func TestShrinkText_HalvesAboveFloor(t *testing.T) {
	got := shrinkText(wordsN(100))
	want := wordsN(50)
	if got != want {
		t.Fatalf("got %d words, want %d words", len(strings.Fields(got)), len(strings.Fields(want)))
	}
}

func TestShrinkText_ClampsToFloorOf40(t *testing.T) {
	// 41 words halved would be 20, but the 40-word floor clamps it to 40.
	got := shrinkText(wordsN(41))
	want := wordsN(40)
	if got != want {
		t.Fatalf("got %d words, want 40", len(strings.Fields(got)))
	}
}

func TestShrinkText_EmptyWhenAtOrBelow40Words(t *testing.T) {
	for _, n := range []int{0, 1, 40} {
		got := shrinkText(wordsN(n))
		if got != "" {
			t.Fatalf("shrinkText(%d words) = %q, want empty string", n, got)
		}
	}
}

// --- embedOnce retryability ---

func TestEmbedOnce_Status500WithContextLengthBodyIsRetryable(t *testing.T) {
	cases := []string{
		"error: context length exceeded",
		"ERROR: CONTEXT LENGTH EXCEEDED",
		"Context Length exceeded the model's limit",
	}
	for _, body := range cases {
		c := newMockClient(func(req *http.Request) (*http.Response, error) {
			return newResponse(http.StatusInternalServerError, body), nil
		})
		_, retry, err := c.embedOnce(context.Background(), "m", "hello")
		if err == nil {
			t.Fatalf("body %q: expected error", body)
		}
		if !retry {
			t.Fatalf("body %q: expected retry=true", body)
		}
	}
}

func TestEmbedOnce_OtherNon2xxIsNotRetryable(t *testing.T) {
	cases := []struct {
		name   string
		status int
		body   string
	}{
		{"500 without context length text", http.StatusInternalServerError, "internal server error"},
		{"400 mentioning context length", http.StatusBadRequest, "context length exceeded"},
		{"404", http.StatusNotFound, "not found"},
		{"503", http.StatusServiceUnavailable, "context length exceeded"},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			c := newMockClient(func(req *http.Request) (*http.Response, error) {
				return newResponse(tc.status, tc.body), nil
			})
			_, retry, err := c.embedOnce(context.Background(), "m", "hello")
			if err == nil {
				t.Fatalf("expected error")
			}
			if retry {
				t.Fatalf("expected retry=false for status=%d body=%q", tc.status, tc.body)
			}
		})
	}
}

func TestEmbedOnce_SuccessReturnsVectorNoRetry(t *testing.T) {
	c := newMockClient(func(req *http.Request) (*http.Response, error) {
		return newResponse(http.StatusOK, `{"embedding":[0.1,0.2,0.3]}`), nil
	})
	vec, retry, err := c.embedOnce(context.Background(), "m", "hello")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if retry {
		t.Fatalf("expected retry=false on success")
	}
	if len(vec) != 3 {
		t.Fatalf("got vector len %d, want 3", len(vec))
	}
}

// --- Embed: empty/whitespace input rejected before any HTTP call ---

func TestEmbed_RejectsEmptyOrWhitespaceInputWithoutHTTPCall(t *testing.T) {
	for _, in := range []string{"", "   ", "\t\n  "} {
		calls := 0
		c := newMockClient(func(req *http.Request) (*http.Response, error) {
			calls++
			return newResponse(http.StatusOK, `{"embedding":[0.1]}`), nil
		})
		_, err := c.Embed(context.Background(), "m", in)
		if err == nil {
			t.Fatalf("input %q: expected error", in)
		}
		if calls != 0 {
			t.Fatalf("input %q: expected no HTTP calls, got %d", in, calls)
		}
	}
}

// --- Embed: non-retryable error returns immediately ---

func TestEmbed_NonRetryableErrorReturnsImmediately(t *testing.T) {
	calls := 0
	c := newMockClient(func(req *http.Request) (*http.Response, error) {
		calls++
		return newResponse(http.StatusBadRequest, "bad request"), nil
	})
	_, err := c.Embed(context.Background(), "m", "some short prompt")
	if err == nil {
		t.Fatalf("expected error")
	}
	if calls != 1 {
		t.Fatalf("expected exactly 1 HTTP call, got %d", calls)
	}
	if !strings.Contains(err.Error(), "status 400") {
		t.Fatalf("expected embedOnce's status-400 error to surface, got: %v", err)
	}
}

// --- Embed: exhausts 4 attempts on persistent context-length errors ---

func TestEmbed_StopsAfter4AttemptsOnPersistentContextLengthError(t *testing.T) {
	// 1000 words survives 4 successive shrinkText halvings without ever
	// dropping to the empty-string floor condition, so all 4 attempts are
	// genuinely exhausted rather than short-circuited by an empty shrink result.
	calls := 0
	c := newMockClient(func(req *http.Request) (*http.Response, error) {
		calls++
		return newResponse(http.StatusInternalServerError, "error: context length exceeded"), nil
	})

	_, err := c.Embed(context.Background(), "m", wordsN(1000))
	if err == nil {
		t.Fatalf("expected error")
	}
	if calls != 4 {
		t.Fatalf("expected exactly 4 HTTP calls, got %d", calls)
	}
	if !strings.Contains(err.Error(), "failed after retries") {
		t.Fatalf("expected 'failed after retries' error, got: %v", err)
	}
}

// sanity check that the mock transport is wired correctly and status codes
// round-trip as expected (guards against a broken test double masking a
// false pass above).
func TestMockTransport_StatusCodeSanity(t *testing.T) {
	c := newMockClient(func(req *http.Request) (*http.Response, error) {
		return newResponse(http.StatusTeapot, strconv.Itoa(http.StatusTeapot)), nil
	})
	_, retry, err := c.embedOnce(context.Background(), "m", "hello")
	if err == nil || retry {
		t.Fatalf("expected non-retryable error for 418, got retry=%v err=%v", retry, err)
	}
}
