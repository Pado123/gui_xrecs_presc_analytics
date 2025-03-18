from typing import Dict, Any
import time
import google.generativeai as genai
import os

class RateLimiter:
    """Simple rate limiter for API calls"""
    def __init__(self, calls: int = 10, per_seconds: int = 62):
        self.calls = calls
        self.per_seconds = per_seconds
        self.timestamps = []
    
    def wait_if_needed(self):
        """Wait if we've exceeded our rate limit"""
        now = time.time()
        self.timestamps = [t for t in self.timestamps if now - t < self.per_seconds]
        
        if len(self.timestamps) >= self.calls:
            sleep_time = self.timestamps[0] + self.per_seconds - now
            if sleep_time > 0:
                time.sleep(sleep_time)
            self.timestamps = self.timestamps[1:]
        
        self.timestamps.append(now)

class GenAI:
    """
    A simple wrapper for Google's Gemini API
    
    Example usage:
    ```python
    # Initialize the model
    model = GenAI(system_prompt="You are a helpful assistant.")
    
    # Simple query
    response = model.generate("What is Python?")
    print(response)
    
    # Chat conversation
    messages = [
        "Hello, how can you help me with Python?",
        "I want to learn about lists"
    ]
    response = model.generate_chat(messages)
    print(response)
    ```
    """
    def __init__(self, system_prompt: str = None, model_name: str = "gemini-2.0-flash-exp"):
        """
        Initialize Gemini model
        
        Args:
            system_prompt: Initial system prompt to set model behavior
            model_name: Gemini model to use (default: gemini-pro)
        """
        # Load API key from AI Studio
        api_key = # os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY not found in environment variables")
        
        self.model_name = model_name
        self.system_prompt = system_prompt
        self.rate_limiter = RateLimiter()
        
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(
            model_name,
            system_instruction=system_prompt
        )
    
    def generate(self, prompt: str) -> str:
        """
        Generate a response for a single prompt
        
        Args:
            prompt: Input text prompt
            
        Returns:
            Model's response as string
        """
        self.rate_limiter.wait_if_needed()
        response = self.model.generate_content(prompt)
        return response.text
    
    def generate_chat(self, messages: list[str]) -> str:
        """
        Generate a response for a chat conversation
        
        Args:
            messages: List of conversation messages
            
        Returns:
            Model's response as string
        """
        self.rate_limiter.wait_if_needed()
        chat = self.model.start_chat()
        for message in messages:
            response = chat.send_message(message)
        return response.text
    
    def get_config(self) -> Dict[str, Any]:
        """Return current model configuration"""
        return {
            "model_name": self.model_name,
            "system_prompt": self.system_prompt
        }

def main():
    """Example usage"""
    # Initialize with a system prompt
    model = GenAI(system_prompt="You are a Python expert. Provide concise answers.",
                  model_name="gemini-2.0-flash-exp") # Or gemini-2.0-flash-thinking-exp-01-21 for reasoning
    
    # Simple query example
    response = model.generate("Explain list comprehension in Python")
    print("Single query response:")
    print(response)
    print("\n" + "="*50 + "\n")
    
    # Chat conversation example
    messages = [
        "What are the main data types in Python?",
        "Give me an example of using dictionaries"
    ]
    response = model.generate_chat(messages)
    print("Chat conversation response:")
    print(response)

if __name__ == "__main__":
    main()