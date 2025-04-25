
import os, sys

MAIN_DIR = os.path.dirname((os.path.dirname(__file__)))
if __name__ == "__main__":
    sys.path.append(MAIN_DIR)


from pylml import login
import pylml
from pylml.LLM import LLMInstructGenerationLlama3

pylml.login(dotenv_path=os.path.join(MAIN_DIR, ".env"))

model = LLMInstructGenerationLlama3(weights_path=os.path.dirname(__file__))
model.load_model()
context = """
    You are a travelling assisant. The user will ask questions about his upcoming trips,
    and you have to answer as precisely as possible, while remaining friendly and
    professionnal. Below is the user's request:\n
"""
model.evaluate_model(prompts="What are the most interesting things do in paris?", contexts=context, display=True)

model = pylml.LLM.LLMInstructGenerationLlama3(version="3B")
model.load_model()
model.evaluate_model(prompts="who are you ?", max_tokens=500, display=True)