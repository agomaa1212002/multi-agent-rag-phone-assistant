# Choose ONE:
# from multi_agent_rule_planner import MultiAgentAssistant
from multi_agent_llm_planner import MultiAgentAssistant

bot = MultiAgentAssistant()

tests = [
    "give me info about Honor 400 Pro",
    "compare Honor 400 vs Honor 400 Pro",
    "best battery android phone",
    "android phone with 12gb ram and 256gb storage and battery 6000mah",
    "best phone under $500"  # should refuse
]

for q in tests:
    print("\n" + "="*80)
    print("USER:", q)
    print(bot.chat(q, debug=True))
