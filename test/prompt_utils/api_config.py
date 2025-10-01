from openai import OpenAI


# api clients, use your own api key. We recommend you modify here. 
custom_client = OpenAI(
    base_url="", 
    api_key=""
)


try:
	from zhipuai import ZhipuAI
	zhipuai_client = ZhipuAI(
		api_key=""
	)
except:
	pass
siliconflow_client = OpenAI(
    api_key="", 
    base_url="https://api.siliconflow.cn/v1"
)
bingxing_client = OpenAI(
    base_url="https://llmapi.paratera.com/v1", 
    api_key=""
)
deepseek_client = OpenAI(
    base_url="https://api.deepseek.com", 
    api_key=""
)

model_name_map_silicon = {
	"ds-r1": "Pro/deepseek-ai/DeepSeek-R1",
	"ds-v3": "Pro/deepseek-ai/DeepSeek-V3",
	"qwen3_32B": "Qwen/Qwen3-32B",
	"qwen2.5_72B": "Qwen/Qwen2.5-72B-Instruct",
	"default": "Pro/deepseek-ai/DeepSeek-R1",
}
model_name_map_bingxing = {
	"ds-r1": "DeepSeek-R1",
	"ds-v3": "DeepSeek-V3-P001",
	"default": "Pro/deepseek-ai/DeepSeek-R1",
}
model_name_map_deepseek = {
	"ds-r1": "deepseek-reasoner",
	"ds-v3": "deepseek-chat",
	"default": "deepseek-reasoner",
}
model_name_map_custom = {
	"o1": "o1-preview", 
	"o3-mini": "o3-mini-2025-01-31",
	"o3": "o3-2025-04-16",
	"gpt-4": "gpt-4",
	"gpt-4o": "gpt-4o",
	"gpt": "gpt-4o",
	"gemini-2.5-pro": "gemini-2.5-pro",
	"gemini-2.5-flash": "gemini-2.5-flash",
	"ds-r1": "deepseek-reasoner",
	"ds-v3": "deepseek-chat",
	"default": "deepseek-reasoner",
}
model_name_map_glm = {
	'glm': 'glm-4-plus',
	'glm-4.5': 'glm-4.5',
	"default": "glm-4-plus",
}