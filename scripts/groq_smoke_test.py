"""Smoke test script for Groq API integration and tool calling."""

import json
import os
import sys
from dotenv import load_dotenv
from groq import Groq

MODEL_NAME = "openai/gpt-oss-120b"


def get_test_status() -> dict:
    """Harmless test tool to verify local function calling."""
    return {"status": "tool_calling_works"}


def main():
    # Ensure stdout/stderr handles UTF-8 on Windows consoles
    if sys.stdout.encoding != "utf-8":
        try:
            sys.stdout.reconfigure(encoding="utf-8")
            sys.stderr.reconfigure(encoding="utf-8")
        except Exception:
            pass

    # 1. Load environment variables from .env
    load_dotenv()
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key or not api_key.strip():
        print("ERROR: GROQ_API_KEY is not set or empty in .env file.", file=sys.stderr)
        print("Please configure GROQ_API_KEY in .env before running this test.", file=sys.stderr)
        sys.exit(1)

    client = Groq(api_key=api_key.strip())

    print("=" * 60)
    print(f"GROQ SMOKE TEST - Model: {MODEL_NAME}")
    print("=" * 60)

    # 2. Simple Chat Completion Test
    print("\n--- Test 1: Simple Chat Completion ---")
    prompt = "In one sentence, explain what a customer support RAG agent does."
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
        )
    except Exception as e:
        print(f"Chat completion failed: {e}", file=sys.stderr)
        sys.exit(1)

    choice = response.choices[0]
    print(f"Model Name: {MODEL_NAME}")
    print(f"Finish Reason: {choice.finish_reason}")
    print(f"Response Text: {choice.message.content.strip()}")

    # 3. Local Tool Calling Test
    print("\n--- Test 2: Local Tool Calling ---")
    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_test_status",
                "description": "Check the system test status.",
                "parameters": {
                    "type": "object",
                    "properties": {},
                    "required": [],
                },
            },
        }
    ]

    messages = [
        {"role": "user", "content": "Use the test tool to check the system status."}
    ]

    try:
        tool_response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            tools=tools,
            tool_choice="auto",
            temperature=0.0,
        )
    except Exception as e:
        print(f"Tool call completion failed: {e}", file=sys.stderr)
        sys.exit(1)

    tool_choice_msg = tool_response.choices[0].message
    if tool_choice_msg.tool_calls:
        for tool_call in tool_choice_msg.tool_calls:
            tool_name = tool_call.function.name
            raw_args = tool_call.function.arguments or "{}"
            try:
                tool_args = json.loads(raw_args)
            except Exception:
                tool_args = raw_args

            if tool_name == "get_test_status":
                tool_result = get_test_status()
            else:
                tool_result = {"error": f"Unknown tool: {tool_name}"}

            print(f"TOOL CALLED: {tool_name}")
            print(f"TOOL ARGUMENTS: {json.dumps(tool_args)}")
            print(f"TOOL RESULT: {json.dumps(tool_result)}")

            # Append assistant's tool call message and tool response message
            messages.append(tool_choice_msg)
            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "name": tool_name,
                "content": json.dumps(tool_result),
            })

            # Fetch final response from model after tool execution
            final_response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=messages,
                temperature=0.0,
            )
            final_text = final_response.choices[0].message.content or ""
            print(f"FINAL RESPONSE: {final_text.strip()}")
    else:
        print("TOOL NOT CALLED")
        if tool_choice_msg.content:
            print(f"Response: {tool_choice_msg.content.strip()}")

    print("\n" + "=" * 60)
    print("GROQ SMOKE TEST COMPLETED SUCCESSFULLY")
    print("=" * 60)


if __name__ == "__main__":
    main()
