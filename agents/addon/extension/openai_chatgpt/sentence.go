package extension

func isPunctuation(r rune) bool {
	if r == ',' || r == '，' ||
		r == '.' || r == '。' ||
		r == '?' || r == '？' ||
		r == '!' || r == '！' {
		return true
	}
	return false
}

func parseSentence(sentence, content string) (string, string, bool) {
	var remain string
	var foundPunc bool
	var lastChar rune

	for _, r := range content {
		// To avoid too many breaks due to consecutive punctuation
        if isPunctuation(r) && isPunctuation(lastChar) {
            sentence += string(r)
            continue
        }
		if !foundPunc {
			sentence += string(r)
		} else {
			remain += string(r)
		}

		if !foundPunc && isPunctuation(r) && len(sentence) > 3 {
			foundPunc = true
		}

		lastChar = r
	}

	return sentence, remain, foundPunc
}
