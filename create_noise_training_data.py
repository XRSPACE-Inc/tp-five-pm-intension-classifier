import argparse
import sys
# from noise_machine_chinese import process_text as process_chinese
from noise_machine_deutsch import process_text as process_deutsch
from noise_machine_english import process_text as process_english
from noise_machine_french import process_text as process_french

def main():
    """
    Main function to process text based on user input and generate language-specific output.
    """
    # Set up argument parser for system parameters
    parser = argparse.ArgumentParser(description="Generate noisy text based on language selection.")
    parser.add_argument("--language", type=str, required=True, choices=['chinese', 'german', 'english', 'french'],
                        help="The language of the text to process (chinese, german, english, french).")
    parser.add_argument("--input", type=str, required=True, help="Input text file path.")
    parser.add_argument("--output", type=str, default="test/output_test.txt", help="Output text file path.")
    parser.add_argument("--typo_probability", type=float, default=0.1, help="Probability of introducing typos (default is 0.1).")
    parser.add_argument("--max_lines", type=int, default=100, help="Maximum number of lines to process (default is 100).")

    # Parse the command line arguments
    args = parser.parse_args()

    # Choose the appropriate language-specific processing function
    if args.language == 'german':
        process_deutsch(args.input, args.output, args.typo_probability, args.max_lines)
    elif args.language == 'english':
        process_english(args.input, args.output, args.typo_probability, args.max_lines)
    elif args.language == 'french':
        process_french(args.input, args.output, args.typo_probability, args.max_lines)

if __name__ == "__main__":
    main()