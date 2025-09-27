from openai_parallel_toolkit import OpenAIModel, ParallelToolkit


model = OpenAIModel("gpt-4o", temperature=0.1)
tool = ParallelToolkit(
    config_path="./new/config.json",
    input_path="./new/enwiki/input.jsonl",
    output_path="./new/enwiki/output.jsonl",
    openai_model=model,
    threads=320,
    max_retries=10,
)
tool.run()
tool.merge("./new/enwiki/merged.json")
