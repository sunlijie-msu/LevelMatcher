import sys
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--line", help="80-character line to validate")
    parser.add_argument("--file", help="File to validate")
    args = parser.parse_args()

    if args.line:
        line = args.line
        print("1234567890" * 8)
        print(line)
        if len(line.rstrip('\r\n')) != 80:
            print(f"ERROR: Line length is {len(line.rstrip())}, expected 80.")
            sys.exit(1)
        print("Line length OK (80 characters).")
        
        # Check col 8
        if line[7] not in ['L', 'G', 'I']:
            print(f"Warning: Col 8 is {line[7]}, usually L or G.")
            
        sys.exit(0)

if __name__ == "__main__":
    main()
