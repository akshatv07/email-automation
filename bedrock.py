import boto3
import json
from urllib.error import HTTPError
import time
import os
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()


def generate_llm_response_with_backoff(prompt, max_tokens, retries=5):
    session = boto3.Session(
        aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
        region_name=os.getenv("AWS_REGION", "ap-south-1")
    )
    bedrock_runtime = session.client('bedrock-runtime')

    # Specify the model ID from environment variable, fallback to default
    model_id = os.getenv('BEDROCK_MODEL_ID', 'mistral.mistral-7b-instruct-v0:2')
    delay = 5  # Initial delay time in seconds
    for attempt in range(retries):
        try:
            body = json.dumps({
                "prompt": prompt,
                "max_tokens": 2000,
                "temperature": 0.4,
                "top_p": 0.8,
                "top_k": 10
            })
            response = bedrock_runtime.invoke_model(
                body=body,
                modelId=model_id,
                accept='application/json',
                contentType='application/json'
            )

            response_body = json.loads(response.get('body').read())

            # Extract and print the generated text
            return response_body.get('outputs')[0].get('text').strip()
            # Try generating the response
            # answer = llm_client.text_generation(
            #     prompt,
            #     max_new_tokens=max_tokens,
            #     temperature=0.4,
            #     top_k=10,
            #     top_p=0.8,
            #     repetition_penalty=1.1
            # ).strip()
            # return answer
        except HTTPError as e:
            if e.response.status_code == 429:  # Too many requests
                print(f"Too many hits. Retrying in {delay} seconds...")
                time.sleep(delay)  # Wait for 'delay' seconds
                delay *= 2  # Exponential backoff: double the wait time
            else:
                raise e  # Raise other errors immediately
    raise RuntimeError("Failed after multiple retries due to rate limiting.")


if __name__ == "__main__":
    reply = generate_llm_response_with_backoff("who is narendra modi tell me in 2 sentence?", 2000)
    print(reply)