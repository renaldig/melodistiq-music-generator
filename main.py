import random
import pandas as pd
import nltk
from nltk.util import ngrams
from nltk.probability import FreqDist
from nltk.tokenize import word_tokenize
import openai
import pronouncing
import re
from mingus.containers import Note
from mingus.containers import Bar
from mingus.containers import Track
from nltk.corpus import cmudict
import mingus.extra.lilypond as lilypond
from mingus.midi import midi_file_out
import contractions
from fpdf import FPDF
import fasttext
import fasttext.util

# Load the fastText model
fasttext.util.download_model('en', if_exists='ignore')
ft = fasttext.load_model('cc.en.300.bin')

nltk.download('cmudict')
from nltk.corpus import cmudict
d = cmudict.dict()

print('there' in d)

# Data Collection
df = pd.read_csv('EdSheeran.csv')
df = df.dropna()
lyrics = df['Lyric'].str.replace('\n', ' ').str.replace('\r', ' ').tolist()

# Data Preprocessing
words = [word_tokenize(re.sub(r'\W+', ' ', lyric).lower()) for lyric in lyrics]
words = [word for sublist in words for word in sublist]


# N-gram Model
n_values = [2, 5, 7]
generated_lyrics = ""

for n in n_values:
    ngrams_list = list(ngrams(words, n, pad_left=True, pad_right=True))
    freq_dist = FreqDist(ngrams_list)

    # Lyrics Generation
    def generate_lyrics(starting_ngram, freq_dist, num_words):
        generated_words = list(starting_ngram)
        for _ in range(num_words):
            next_word_candidates = [ngram[-1] for ngram in freq_dist.keys() if ngram[:n-1] == tuple(generated_words[-(n-1):])]
            if next_word_candidates:
                next_word = random.choice(next_word_candidates)
                generated_words.append(next_word)
            else:
                break
        return ' '.join(generated_words).replace(' ,', ',').replace(' .', '.').replace(' ;', ';')

    starting_ngram = random.choice(list(freq_dist.keys()))
    generated_lyrics += generate_lyrics(starting_ngram, freq_dist, 200)

# Use GPT-3.5 API
openai.api_key = '<INSERT_API_KEY_HERE>'

response = openai.Completion.create(
  engine="text-davinci-003",
  prompt=generated_lyrics,
  max_tokens=200
)

gpt_lyrics = response.choices[0].text.strip()

def find_similar_word(word):
    # Get the 10 most similar words to the given word
    similar_words = ft.get_nearest_neighbors(word, k=10)
    # Select the most similar word that is present in the CMU dictionary
    for _, similar_word in similar_words:
        if similar_word in d:
            return similar_word
    return None

# Make the Lyrics Rhyme
def format_lyrics(lyrics):
    lines = lyrics.split('.')
    formatted_lyrics = ""
    for i in range(len(lines)):
        line = lines[i]
        words = line.split(' ')
        last_word = words[-1].lower()
        rhymes = pronouncing.rhymes(last_word)
        rhymes = [rhyme for rhyme in rhymes if rhyme in words]
        if rhymes:
            rhyme = random.choice(rhymes)
            formatted_line = re.sub(rf'\b{last_word}\b', rhyme, line)
            formatted_lyrics += formatted_line + '\n'
        else:
            formatted_lyrics += line + '\n'
    return formatted_lyrics

def add_newlines(lyrics):
    # Split the lyrics at punctuation marks
    lines = re.split(r'[,.;]', lyrics)
    # Remove empty lines and strip whitespace
    lines = [line.strip() for line in lines if line.strip()]
    return '\n'.join(lines)

def generate_melody(lyrics):
    # Load the CMU Pronouncing Dictionary
    stress_pattern = []

    tokens = nltk.word_tokenize(lyrics)

    # Fix contractions
    fixed_tokens = [contractions.fix(token) for token in tokens]

    # Function to get the stress pattern of a word
    def get_stress(word):
        word = word.lower()
        phones = pronouncing.phones_for_word(word)
        if phones:
            stress_pattern = [int(s) for s in pronouncing.stresses(phones[0])]
            return [stress_pattern]
        else:
            # handle contractions
            if "'" in word:
                parts = word.split("'")
                stress_pattern = []
                for part in parts:
                    stress_pattern += get_stress(part)
                return stress_pattern
            # handle hyphenated words
            elif '-' in word:
                parts = word.split('-')
                stress_pattern = []
                for part in parts:
                    stress_pattern += get_stress(part)
                return stress_pattern
            else:
                print(f'Word not found in dictionary: {word}')
                # Find a similar word in the dictionary and use its stress pattern
                similar_word = find_similar_word(word)
                if similar_word:
                    return get_stress(similar_word)
                else:
                    return [[0, 1, 2]]  # Use default pattern if no similar word is found

    # Get the stress pattern of the lyrics
    for word in fixed_tokens:
        word = re.sub(r'[^\w\s]', '', word)  # remove punctuation
        stress_pattern += get_stress(word)

    # Flatten the stress_pattern list
    stress_pattern = [item for sublist in stress_pattern for item in sublist]

    print(lyrics)
    print(tokens)
    print(["Here are the stress patterns:"] + stress_pattern)

    # Generate a melody based on the stress pattern
    track = Track()
    b = Bar()
    b.set_meter((4, 4))
    beats_in_current_bar = 0
    for stress in stress_pattern:
        if stress == 0:
            note = Note('C', 4)
        elif stress == 1:
            note = Note('E', 4)
        elif stress == 2:
            note = Note('G', 4)
        b + note
        beats_in_current_bar += 1
        if beats_in_current_bar == 4:
            track.add_bar(b)
            b = Bar()
            b.set_meter((4, 4))
            beats_in_current_bar = 0
    track.add_bar(b)


    return track



# Generate and print formatted lyrics
formatted_lyrics = format_lyrics(gpt_lyrics)
formatted_lyrics_with_newlines = add_newlines(formatted_lyrics)
print("Here are the lyrics:" + formatted_lyrics_with_newlines)

# Generate melody
melody = generate_melody(formatted_lyrics_with_newlines)

# Export melody to a LilyPond file
lilypond_string = lilypond.from_Track(melody)
with open('melody.ly', 'w') as f:
    f.write(lilypond_string)

print(melody)

# Create PDF document
pdf = FPDF()
pdf.add_page()
pdf.set_font('Arial', 'B', 16)
title = formatted_lyrics_with_newlines.split('\n')[0]
pdf.cell(0, 10, title, 0, 1, 'C')
pdf.set_font('Arial', '', 12)
pdf.multi_cell(0, 10, formatted_lyrics_with_newlines)
pdf.output('lyrics.pdf')

# Export melody to a MIDI file
midi_file_out.write_Track("melody.mid", melody)
