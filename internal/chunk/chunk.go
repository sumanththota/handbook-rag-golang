package chunk

import "strings"

type Input struct {
	Page int
	Text string
}

type Chunk struct {
	ID   int
	Page int
	Text string
}

func Build(inputs []Input, wordsPerChunk int) []Chunk {
	if wordsPerChunk <= 0 {
		wordsPerChunk = 350
	}

	var (
		out []Chunk
		id  int
	)

	for _, in := range inputs {
		words := strings.Fields(in.Text)
		if len(words) == 0 {
			continue
		}

		for start := 0; start < len(words); start += wordsPerChunk {
			end := start + wordsPerChunk
			if end > len(words) {
				end = len(words)
			}

			text := strings.Join(words[start:end], " ")
			out = append(out, Chunk{
				ID:   id,
				Page: in.Page,
				Text: text,
			})
			id++
		}
	}

	return out
}
