
import os, sys

MAIN_DIR = os.path.dirname((os.path.dirname(__file__)))
if __name__ == "__main__":
    sys.path.append(MAIN_DIR)


from pylml import login
import pylml
from pylml.LLM import LLMInstructGenerationLlama3

pylml.login(dotenv_path=os.path.join(MAIN_DIR, ".env"))

model = LLMInstructGenerationLlama3(weights_path=os.path.dirname(__file__))
model.load_model(display=True)

while True:
    input_txt = input("chat: ")
    model.evaluate_model(prompts=input_txt, display=True)

# model = pylml.LLM.LLMInstructGenerationLlama3(version="3B")
# model.load_model()
# model.evaluate_model(prompts="who are you ?", max_tokens=500, display=True)

