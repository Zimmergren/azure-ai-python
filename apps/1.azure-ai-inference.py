# Sample by Tobias Zimmergren
# See: 

import os
from azure.identity import DefaultAzureCredential
from azure.ai.inference import ChatCompletionsClient
from azure.ai.inference.models import SystemMessage, UserMessage, AssistantMessage

# Load environment variables from .env file locally; 
# doesn't load anything in Azure, where it instead fetches App Settings.
from dotenv import load_dotenv
load_dotenv()

ENDPOINT = os.environ["AZURE_AI_ENDPOINT"]
MODEL = os.environ["AZURE_AI_MODEL"]

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
