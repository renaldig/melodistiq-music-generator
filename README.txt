# GPT-3.5-powered Song Lyrics Generator

This project generates new song lyrics using 2-gram, 5-gram, and 7-gram models, then feeds the result into the GPT-3.5 API, and post-processes the generated lyrics to make them rhyme and have a specific rhythm.

## Installation

1. Install the required packages by running:

2. Sign up for an OpenAI account and get your API key.

3. Replace `'your-api-key'` in the script with your actual API key.

## Usage

1. Place your dataset in a CSV file named `CharliePuth.csv` with a column `Lyrics` that contains the song lyrics.

2. Run the script: python lyrics_generator.py

3. The script will generate lyrics, feed them into the GPT-3.5 API, and then make the generated lyrics rhyme and have a specific rhythm.

## License

This project is licensed under the MIT License - see the `LICENSE` file for details.
