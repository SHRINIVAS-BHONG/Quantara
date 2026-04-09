import sys
import io
from quantara.agent import run_quantara_agent

# Force UTF-8 encoding for Windows terminals to support emojis (🤖, ✅)
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stdin = io.TextIOWrapper(sys.stdin.buffer, encoding='utf-8')

if __name__ == "__main__":
    query = input("Enter query: ")
    result = run_quantara_agent(query)

    print("\n=== RESULT ===\n")
    print(result)
