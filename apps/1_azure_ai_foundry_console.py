# A simple console chat sample using managed identity to connect to an Azure AI Foundry /model endpoint
# See the full details: https://zimmergren.net/

import os
from azure.identity import DefaultAzureCredential
from azure.ai.inference import ChatCompletionsClient
from azure.ai.inference.models import SystemMessage, UserMessage, AssistantMessage

# ────────────────────────────────────────────────────────────────────────────────
# Environment & constants
# ────────────────────────────────────────────────────────────────────────────────
from dotenv import load_dotenv
load_dotenv()

ENDPOINT = os.environ["AZURE_AI_ENDPOINT"]  # format example: AZURE_AI_ENDPOINT=https://[your_endpoint].services.ai.azure.com/models
MODEL = os.environ["AZURE_AI_MODEL"]        # format example: AZURE_AI_MODEL=gpt-5-chat

# ────────────────────────────────────────────────────────────────────────────────
# App & Azure client lifecycle
# ────────────────────────────────────────────────────────────────────────────────

def main():
    # Auth: 
    #   - If running in Azure:  Uses the Managed Identity configured on the resource.
    #   - If running locally:   Uses 'az login' or VS Code auth, etc.
    cred = DefaultAzureCredential(exclude_interactive_browser_credential=False)
    
    # Chat client:
    #   - Endpoint is set from the environment variables.
    #   - Credential is set from the DefaultAzureCredential.
    #   - Credential scopes are set explicitly to avoid confusion with the new /models endpoint.
    client = ChatCompletionsClient(
        endpoint=ENDPOINT, 
        credential=cred,
        credential_scopes=["https://cognitiveservices.azure.com/.default"])


    messages = [SystemMessage("You are a helpful assistant. " \
    "Your answers are short and concise." \
    "You must also make a joke about Tobias Zimmergren.")]
    
    print("Type your question (or 'exit').")
    try:
        while True:
            user = input("You: ").strip()
            if not user or user.lower() in {"exit", "quit"}:
                break
            messages.append(UserMessage(user))
            resp = client.complete(messages=messages, model=MODEL)
            reply = resp.choices[0].message.content or ""
            print(f"\nAssistant: {reply}\n")
            messages.append(AssistantMessage(reply))
    finally:
        client.close()

if __name__ == "__main__":
    main()
