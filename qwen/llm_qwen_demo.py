from LLM import Qwen2_5_LLM
llm = Qwen2_5_LLM(mode_name_or_path = "/home/aigc3/llm/qwen25instruct/model/qwen/Qwen2___5-7B-Instruct")

print(llm("你是谁"))