llama_cog_tokenizer = ""
llama_r1_tokenizer = ""
llama_noreason_tokenizer = ""

qwen_cog_tokenizer = ""
qwen_noreason_tokenizer = ""
qwen_r1_tokenizer = ""

tokenizer_name_of_files = {
    # llama files
    "llama_distilled_r1": llama_r1_tokenizer,
    "llama_direct": llama_noreason_tokenizer,
    "llama_cogflow": llama_cog_tokenizer,
    # qwen files
    "qwen_cogflow": qwen_cog_tokenizer,
    "qwen_distilled_r1": qwen_r1_tokenizer,
    "qwen_direct": qwen_noreason_tokenizer,

    # api files
    "glm-4.5_results_api": "THUDM/glm-4-9b-chat",
    "o3_results_api": "gpt-4o", 
    "o3-mini_results_api": "gpt-4o",
    "gpt-4o-cot_results_api": "gpt-4o",
    "gpt-4o_results_api": "gpt-4o",
    # "gemini-2.5-flash_results_api": "gemini-2.5-flash",
    "gemini-2.5-flash_results_api": "gpt-4o",
    # "gemini-2.5-pro_results_api": "gemini-2.5-pro",
    "gemini-2.5-pro_results_api": "gpt-4o",
}