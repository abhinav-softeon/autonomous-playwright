import argparse
import os
import sys
from collections import OrderedDict

import boto3
from botocore.exceptions import BotoCoreError, ClientError
from dotenv import load_dotenv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Send a short prompt to AWS Bedrock and print only the model text."
    )
    parser.add_argument(
        "--model-id",
        default=os.getenv("BEDROCK_MODEL_ID", ""),
        help="Bedrock model or inference profile ID. Falls back to BEDROCK_MODEL_ID from .env.",
    )
    parser.add_argument(
        "--all-models",
        action="store_true",
        help="Test all BEDROCK_MODEL* entries from .env and print each response/error.",
    )
    parser.add_argument(
        "--region",
        default=os.getenv("AWS_DEFAULT_REGION") or os.getenv("AWS_REGION") or "us-east-1",
        help="AWS region for Bedrock runtime.",
    )
    parser.add_argument(
        "--prompt",
        default="hi",
        help="Prompt text to send.",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=128,
        help="Max tokens for the model response.",
    )
    return parser.parse_args()


def invoke_prompt(model_id: str, region: str, prompt: str, max_tokens: int) -> str:
    client = boto3.client("bedrock-runtime", region_name=region)

    response = client.converse(
        modelId=model_id,
        messages=[
            {
                "role": "user",
                "content": [{"text": prompt}],
            }
        ],
        inferenceConfig={"maxTokens": max_tokens},
    )

    parts = response.get("output", {}).get("message", {}).get("content", [])
    texts = [p.get("text", "") for p in parts if isinstance(p, dict) and p.get("text")]
    return "\n".join(texts).strip()


def get_models_from_env() -> OrderedDict[str, str]:
    models: OrderedDict[str, str] = OrderedDict()
    for key in sorted(os.environ.keys()):
        if key.startswith("BEDROCK_MODEL"):
            value = (os.environ.get(key) or "").strip()
            if value:
                models[key] = value
    return models


def main() -> int:
    load_dotenv(".env")
    args = parse_args()
    explicit_model_arg = "--model-id" in sys.argv

    models: OrderedDict[str, str] = OrderedDict()
    if args.all_models or not explicit_model_arg:
        models = get_models_from_env()
    elif args.model_id:
        models["BEDROCK_MODEL_ID"] = args.model_id

    if not models:
        print("ERROR: No model IDs found.")
        print("Set BEDROCK_MODEL_ID/BEDROCK_MODEL_* in .env or pass --model-id.")
        return 2

    any_error = False
    for env_key, model_id in models.items():
        print(f"MODEL_KEY: {env_key}")
        print(f"MODEL_ID: {model_id}")
        try:
            text = invoke_prompt(
                model_id=model_id,
                region=args.region,
                prompt=args.prompt,
                max_tokens=args.max_tokens,
            )
            print(f"RESPONSE: {text if text else '[No text returned]'}")
        except (ClientError, BotoCoreError) as err:
            any_error = True
            print(f"ERROR [{type(err).__name__}]: {err}")
        except Exception as err:
            any_error = True
            print(f"ERROR [{type(err).__name__}]: {err}")
        print()

    return 1 if any_error else 0


if __name__ == "__main__":
    sys.exit(main())
