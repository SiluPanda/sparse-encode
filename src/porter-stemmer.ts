// Porter Stemmer — classic algorithm (Porter 1980)

function hasCVC(word: string): boolean {
  // ends in consonant-vowel-consonant where final consonant is not w, x, y
  if (word.length < 3) return false
  const last = word[word.length - 1]
  if ('wxy'.includes(last)) return false
  const vowels = new Set(['a', 'e', 'i', 'o', 'u'])
  const c = word.length - 1
  return !vowels.has(word[c]) && vowels.has(word[c - 1]) && !vowels.has(word[c - 2])
}

function containsVowel(word: string): boolean {
  for (const ch of word) {
    if ('aeiou'.includes(ch)) return true
    if (ch === 'y' && word.indexOf(ch) > 0) return true
  }
  return false
}

function measure(word: string): number {
  // count VC sequences
  const vowels = new Set(['a', 'e', 'i', 'o', 'u'])
  let m = 0
  let inVowel = false
  for (let i = 0; i < word.length; i++) {
    const ch = word[i]
    const isVowel = vowels.has(ch) || (ch === 'y' && i > 0)
    if (isVowel) {
      inVowel = true
    } else {
      if (inVowel) {
        m++
        inVowel = false
      }
    }
  }
  return m
}

function endsDoubleConsonant(word: string): boolean {
  if (word.length < 2) return false
  const last = word[word.length - 1]
  const prev = word[word.length - 2]
  return last === prev && !'aeiou'.includes(last)
}

export function stem(word: string): string {
  if (word.length <= 2) return word

  let w = word.toLowerCase()

  // Step 1a
  if (w.endsWith('sses')) {
    w = w.slice(0, -2)
  } else if (w.endsWith('ies')) {
    w = w.slice(0, -2) // ies → i
  } else if (w.endsWith('ss')) {
    // keep
  } else if (w.endsWith('s')) {
    w = w.slice(0, -1)
  }

  // Step 1b
  let step1bTriggered = false
  if (w.endsWith('eed')) {
    const stem1b = w.slice(0, -3)
    if (measure(stem1b) > 0) {
      w = w.slice(0, -1) // eed → ee
    }
  } else if (w.endsWith('ed')) {
    const stem1b = w.slice(0, -2)
    if (containsVowel(stem1b)) {
      w = stem1b
      step1bTriggered = true
    }
  } else if (w.endsWith('ing')) {
    const stem1b = w.slice(0, -3)
    if (containsVowel(stem1b)) {
      w = stem1b
      step1bTriggered = true
    }
  }

  if (step1bTriggered) {
    if (w.endsWith('at') || w.endsWith('bl') || w.endsWith('iz')) {
      w = w + 'e'
    } else if (endsDoubleConsonant(w) && !w.endsWith('l') && !w.endsWith('s') && !w.endsWith('z')) {
      w = w.slice(0, -1)
    } else if (measure(w) === 1 && hasCVC(w)) {
      w = w + 'e'
    }
  }

  // Step 1c
  if (w.endsWith('y') && w.length > 2) {
    const before = w.slice(0, -1)
    if (containsVowel(before)) {
      w = before + 'i'
    }
  }

  // Step 2
  const step2Map: [string, string][] = [
    ['ational', 'ate'],
    ['tional', 'tion'],
    ['enci', 'ence'],
    ['anci', 'ance'],
    ['izer', 'ize'],
    ['abli', 'able'],
    ['alli', 'al'],
    ['entli', 'ent'],
    ['eli', 'e'],
    ['ousli', 'ous'],
    ['ization', 'ize'],
    ['ation', 'ate'],
    ['ator', 'ate'],
    ['alism', 'al'],
    ['iveness', 'ive'],
    ['fulness', 'ful'],
    ['ousness', 'ous'],
    ['aliti', 'al'],
    ['iviti', 'ive'],
    ['biliti', 'ble'],
  ]
  for (const [suffix, replacement] of step2Map) {
    if (w.endsWith(suffix)) {
      const base = w.slice(0, -suffix.length)
      if (measure(base) > 0) {
        w = base + replacement
      }
      break
    }
  }

  // Step 3
  const step3Map: [string, string][] = [
    ['icate', 'ic'],
    ['ative', ''],
    ['alize', 'al'],
    ['iciti', 'ic'],
    ['ical', 'ic'],
    ['ful', ''],
    ['ness', ''],
  ]
  for (const [suffix, replacement] of step3Map) {
    if (w.endsWith(suffix)) {
      const base = w.slice(0, -suffix.length)
      if (measure(base) > 0) {
        w = base + replacement
      }
      break
    }
  }

  // Step 4
  const step4Suffixes = [
    'ement', 'ment', 'ance', 'ence', 'able', 'ible', 'ant', 'ent',
    'ion', 'ism', 'ate', 'iti', 'ous', 'ive', 'ize', 'al', 'er', 'ic',
  ]
  for (const suffix of step4Suffixes) {
    if (w.endsWith(suffix)) {
      const base = w.slice(0, -suffix.length)
      if (suffix === 'ion') {
        if (measure(base) > 1 && (base.endsWith('s') || base.endsWith('t'))) {
          w = base
        }
      } else if (measure(base) > 1) {
        w = base
      }
      break
    }
  }

  // Step 5a
  if (w.endsWith('e')) {
    const base = w.slice(0, -1)
    if (measure(base) > 1) {
      w = base
    } else if (measure(base) === 1 && !hasCVC(base)) {
      w = base
    }
  }

  // Step 5b
  if (w.endsWith('ll') && measure(w.slice(0, -1)) > 1) {
    w = w.slice(0, -1)
  }

  return w
}
