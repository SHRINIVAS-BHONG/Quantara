from quantara.agent import run_quantara_agent

if __name__ == "__main__":
    query = input("Enter query: ")
    result = run_quantara_agent(query)

    print("\n=== RESULT ===\n")
    print(result)