import asyncio
from helper import MODEL_NAME, AUTO_ANSWERS, ROLE_ASSISTANT
from typing import List
from huggingface_hub import InferenceClient
import os




class RAGModel:
    """ Class for model related functions, such as loading the model, API/model interaction and such."""

    def __init__(self, api_key: str, model_name: str = MODEL_NAME):
        self.api_key = api_key
        self.model_name = model_name
        self.client = InferenceClient(provider="nebius", token=api_key)

        # Backup (maybe)
        #try:
        #    self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        #except Exception as e:
        #    print(f"Lokaler Tokenizer nicht verfügbar: {e}")
        #    self.tokenizer = None

    def get_auto_answer(self, reason: str): 
        return f"Apologies: {reason}"

    def generate_response(self, 
                               query: str, 
                               context: List[str],
                               temperature: float = 0.7,
                               max_tokens: int = 512,
                               top_p: float = 0.9,
                               top_k: int = None) -> str:
        """Generate an answer via Hugging Face API with configurable parameters
        
            Args: 
                query: User input string 
                context: Context retrieved from context DB
                temperature: Sampling temperature (0.0 to 2.0)
                max_tokens: Maximum tokens to generate
                top_p: Nucleus sampling parameter
                top_k: Top-k sampling parameter (not used in HF API)
        """
        
        if len(context) >= 10: 
            context_text = "\n".join(context[:10])  # limit to 10 most relevant chunks
        else: 
            context_text = "\n".join(context)

        UNIVERSITY_ASSISTANT_SYSTEM_PROMPT = """
        You are a university assistant that helps ONLY with university-related topics using the available database.

        ## APPROPRIATE RESPONSES
        You can help with:
        - "What courses is [student] taking?"
        - "Who teaches [course]?" 
        - "Which students are in [professor]'s class?"
        - "What is [professor]'s email?"
        - "What department is [faculty] in?"

        ## STRICT BOUNDARIES
        Never discuss:
        - Grades, academic performance, or GPA
        - Financial information, tuition, or payments
        - Sensitive student data beyond basic directory info
        - Any non-university topics (medical, legal, financial advice)

        ## RESPONSE STYLE
        - Be helpful and professional
        - Redirect inappropriate requests: "I can only help with university academic topics"
        - For sensitive data: "I don't have access to that information. Please contact [relevant office]"
        - Only share information appropriate for academic purposes

        ## KNOWLEDGE USAGE 
        Use the provided Context below to answer user requests.
        """
        
        prompt = f"{UNIVERSITY_ASSISTANT_SYSTEM_PROMPT} \n\nContext: {context_text}\nQuestion: {query}\nAnswer: Based on the given context,"

        try:   
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                {
                    "role": ROLE_ASSISTANT,
                    "content": [
                        {
                            "type": "text",
                            "text": prompt
                        },
                    ]
                }
            ],
                max_tokens=max_tokens,      # Use configurable parameter
                temperature=temperature,     # Use configurable parameter
                top_p=top_p,                # Use configurable parameter
            )
                    
            if response: 
                return response.choices[0].message.content
            else:
                print(f"NO RESPONSE")
                return self.get_auto_answer(AUTO_ANSWERS.COULD_NOT_GENERATE.value)
                
        except asyncio.TimeoutError:
            print("API Timeout")
            return self.get_auto_answer(AUTO_ANSWERS.REQUEST_TIMED_OUT.value)
        except Exception as e:
            print(f"API Fehler: {e}")
            return self.get_auto_answer(AUTO_ANSWERS.UNEXPECTED_ERROR.value)

