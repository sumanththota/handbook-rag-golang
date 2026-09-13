package chunk

import (
	"reflect"
	"testing"
)

// Characterization tests: these lock in Build's CURRENT behavior (including
// known-mediocre choices like zero overlap) as ground truth for the Python
// port to match. They do not assert Build is "correct" — see docs/report.md
// for the known chunking-quality issues, which are a separate, later change.

func TestBuild_EmptyInput(t *testing.T) {
	got := Build(nil, 5)
	if len(got) != 0 {
		t.Fatalf("expected no chunks, got %d: %+v", len(got), got)
	}
}

func TestBuild_BlankPageTextIsSkipped(t *testing.T) {
	got := Build([]Input{{Page: 1, Text: "   "}}, 5)
	if len(got) != 0 {
		t.Fatalf("expected blank page to produce no chunks, got %+v", got)
	}
}

func TestBuild_SingleChunkUnderLimit(t *testing.T) {
	got := Build([]Input{{Page: 1, Text: "one two three"}}, 5)
	want := []Chunk{{ID: 0, Page: 1, Text: "one two three"}}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("got %+v, want %+v", got, want)
	}
}

func TestBuild_ExactMultipleBoundary(t *testing.T) {
	// 10 words, 5 per chunk -> exactly 2 chunks, no trailing empty chunk.
	got := Build([]Input{{Page: 1, Text: "a b c d e f g h i j"}}, 5)
	want := []Chunk{
		{ID: 0, Page: 1, Text: "a b c d e"},
		{ID: 1, Page: 1, Text: "f g h i j"},
	}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("got %+v, want %+v", got, want)
	}
}

func TestBuild_NonMultipleTrailingChunkIsShort(t *testing.T) {
	// 7 words, 5 per chunk -> [0:5], [5:7] (last chunk shorter, not padded/merged).
	got := Build([]Input{{Page: 1, Text: "a b c d e f g"}}, 5)
	want := []Chunk{
		{ID: 0, Page: 1, Text: "a b c d e"},
		{ID: 1, Page: 1, Text: "f g"},
	}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("got %+v, want %+v", got, want)
	}
}

func TestBuild_ZeroOverlapBetweenChunks(t *testing.T) {
	// Locks in the known characteristic from docs/report.md: no word appears
	// in two chunks. A quality fix must change this test deliberately, not
	// accidentally during the port.
	got := Build([]Input{{Page: 1, Text: "a b c d e f g h i j"}}, 4)
	seen := map[string]int{}
	for _, c := range got {
		for _, w := range splitWords(c.Text) {
			seen[w]++
		}
	}
	for w, n := range seen {
		if n != 1 {
			t.Fatalf("word %q appeared in %d chunks, want exactly 1 (zero overlap)", w, n)
		}
	}
}

func TestBuild_IDsIncrementAcrossPages(t *testing.T) {
	got := Build([]Input{
		{Page: 1, Text: "a b c d e f"},
		{Page: 2, Text: "g h i"},
	}, 5)
	want := []Chunk{
		{ID: 0, Page: 1, Text: "a b c d e"},
		{ID: 1, Page: 1, Text: "f"},
		{ID: 2, Page: 2, Text: "g h i"},
	}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("got %+v, want %+v", got, want)
	}
}

func TestBuild_SkippedBlankPageDoesNotBurnAnID(t *testing.T) {
	// A blank page in the middle must not leave a gap in chunk IDs, since the
	// Python port must reproduce identical IDs for identical input.
	got := Build([]Input{
		{Page: 1, Text: "a b c"},
		{Page: 2, Text: ""},
		{Page: 3, Text: "d e f"},
	}, 5)
	want := []Chunk{
		{ID: 0, Page: 1, Text: "a b c"},
		{ID: 1, Page: 3, Text: "d e f"},
	}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("got %+v, want %+v", got, want)
	}
}

func TestBuild_WhitespaceIsCollapsedByFieldsSplit(t *testing.T) {
	// strings.Fields treats any run of whitespace (spaces, tabs, newlines) as
	// one separator and re-joins with single spaces — this normalizes text,
	// it is not a lossless roundtrip. The Python port must match this exactly
	// (e.g. Python's str.split() with no args has the same semantics).
	got := Build([]Input{{Page: 1, Text: "a\n\tb   c\n\n d"}}, 10)
	want := []Chunk{{ID: 0, Page: 1, Text: "a b c d"}}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("got %+v, want %+v", got, want)
	}
}

func TestBuild_NonPositiveWordsPerChunkDefaultsTo350(t *testing.T) {
	words := make([]byte, 0)
	text := ""
	for i := 0; i < 400; i++ {
		words = append(words, 'x', ' ')
	}
	text = string(words)

	gotZero := Build([]Input{{Page: 1, Text: text}}, 0)
	gotNeg := Build([]Input{{Page: 1, Text: text}}, -5)
	gotExplicit350 := Build([]Input{{Page: 1, Text: text}}, 350)

	if len(gotZero) != 2 || len(gotNeg) != 2 {
		t.Fatalf("wordsPerChunk<=0 should default to 350 (400 words -> 2 chunks), got zero=%d neg=%d", len(gotZero), len(gotNeg))
	}
	if !reflect.DeepEqual(gotZero, gotExplicit350) || !reflect.DeepEqual(gotNeg, gotExplicit350) {
		t.Fatalf("wordsPerChunk<=0 must behave identically to explicit 350")
	}
}

func splitWords(s string) []string {
	var out []string
	word := ""
	for _, r := range s {
		if r == ' ' {
			if word != "" {
				out = append(out, word)
				word = ""
			}
			continue
		}
		word += string(r)
	}
	if word != "" {
		out = append(out, word)
	}
	return out
}
