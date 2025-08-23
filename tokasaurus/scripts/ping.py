import time

import pydra
from openai import OpenAI


class ScriptConfig(pydra.Config):
    prompt: str

    model: str = ""
    port: int = 10210
    host: str = "0.0.0.0"
    chat: bool = False
    max_tokens: int = 100
    n: int = 1
    stream: bool = False
    temperature: float = 0.0
    hide: bool = False
    retries: int = 0


def ping(config: ScriptConfig):
    client = OpenAI(
        base_url=f"http://{config.host}:{config.port}/v1",
        api_key="fake-key",
        max_retries=config.retries,
    )

    print("Making request...")
    start = time.time()
    
    responses = []
    
    if config.chat:
        out = client.chat.completions.create(
            model=config.model,
            messages=[{"role": "user", "content": config.prompt}],
            max_tokens=config.max_tokens,
            n=config.n,
            temperature=config.temperature,
            stream=config.stream,
        )
        if config.stream:
            collected_content = []
            for chunk in out:
                if chunk.choices and len(chunk.choices) > 0 and chunk.choices[0].delta.content is not None:
                    content = chunk.choices[0].delta.content
                    print(content, end="", flush=True)
                    collected_content.append(content)
            print()  # newline after streaming
            responses = ["".join(collected_content)]
        else:
            responses = [choice.message.content for choice in out.choices]
    else:
        out = client.completions.create(
            model=config.model,
            prompt=config.prompt,
            max_tokens=config.max_tokens,
            n=config.n,
            temperature=config.temperature,
            stream=config.stream,
        )
        if config.stream:
            collected_content = []
            for chunk in out:
                if chunk.choices and len(chunk.choices) > 0 and chunk.choices[0].text is not None:
                    content = chunk.choices[0].text
                    print(content, end="", flush=True)
                    collected_content.append(content)
            print()  # newline after streaming
            responses = ["".join(collected_content)]
        else:
            responses = [choice.text for choice in out.choices]

    end = time.time()
    print(f"Time taken: {end - start} seconds")

    if not config.hide:
        print("Responses:")
        print("-" * 100)
        for i, response in enumerate(responses):
            print(f"Response {i}: {response}")
            print("-" * 100)


def main():
    pydra.run(ping)


if __name__ == "__main__":
    main()
