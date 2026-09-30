"""Synthetic feature example; NOT image recognition."""
from acm import CognitiveMemory

memory = CognitiveMemory()
memory.learn("ZENTRA", [0.9, 0.1, 0.0], concepts=["car brand", "electric vehicle"], source="synthetic-demo")
print("Known:", memory.recognize([0.88, 0.12, 0.0]))
print("Unfamiliar:", memory.recognize([0.0, 0.0, 1.0]))
