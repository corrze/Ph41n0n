import re
import sys

# Regex pattern to match most emoji ranges
emoji_pattern = re.compile(
    "["
    "\U0001F600-\U0001F64F"  # emoticons
    "\U0001F300-\U0001F5FF"  # symbols & pictographs
    "\U0001F680-\U0001F6FF"  # transport & map symbols
    "\U0001F1E0-\U0001F1FF"  # flags
    "\U00002700-\U000027BF"  # dingbats
    "\U0001F900-\U0001F9FF"  # supplemental symbols & pictographs
    "\U00002600-\U000026FF"  # misc symbols
    "\U00002B00-\U00002BFF"  # arrows
    "\U0001FA70-\U0001FAFF"  # symbols
    "\U000025A0-\U000025FF"  # geometric shapes
    "]+",
    flags=re.UNICODE
)

def remove_emojis_from_file(input_file, output_file=None):
    # Read file
    with open(input_file, "r", encoding="utf-8") as f:
        text = f.read()

    # Remove emojis
    cleaned = emoji_pattern.sub("", text)

    # Write back (in place if no output_file specified)
    if output_file is None:
        output_file = input_file

    with open(output_file, "w", encoding="utf-8") as f:
        f.write(cleaned)

    print(f"✅ Emojis removed. Saved to: {output_file}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python remove_emojis.py <python_file> [<output_file>]")
        sys.exit(1)

    input_path = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 else None

    remove_emojis_from_file(input_path, output_path)
