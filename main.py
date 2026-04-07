from hermes_app.agent import run_agent


if __name__ == "__main__":
    query = input("Enter query: ")
    result = run_agent(query)

    print("\n=== RESULT ===\n")
    print(result["recommendation"])

    print("\nTop Traders:\n")
    for t in result["top_traders"]:
        print(t)