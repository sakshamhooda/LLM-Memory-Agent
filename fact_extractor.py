import json
import logging
from typing import List, Dict, Any
from openai import OpenAI
from config import Config

logger = logging.getLogger(__name__)

class FactExtractor:
    """Extracts atomic facts from user messages using OpenAI."""
    
    def __init__(self, client: OpenAI):
        self.client = client
        self.model = Config.GPT_FACT_EXTRACTION_MODEL
    
    def extract_facts(self, text: str) -> List[str]:
        """
        Extract atomic facts from the given text.
        
        Args:
            text: The input text to extract facts from
            
        Returns:
            List of atomic facts as strings
        """
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": """Extract atomic facts from the user's message. 
                        Return only a JSON array of strings, each representing one fact.
                        
                        Examples:
                        - Input: "I use Shram and Magnet as productivity tools"
                        - Output: ["User uses Shram as productivity tool", "User uses Magnet as productivity tool"]
                        
                        - Input: "I don't use Magnet anymore"
                        - Output: ["User no longer uses Magnet"]
                        
                        - Input: "My name is John and I live in New York"
                        - Output: ["User's name is John", "User lives in New York"]
                        
                        Keep facts atomic and specific. Don't include unnecessary words."""
                    },
                    {"role": "user", "content": f"Extract facts from: '{text}'"}
                ],
                temperature=0.1,
                max_tokens=500
            )
            
            content = response.choices[0].message.content.strip()
            
            # Try to parse as JSON
            try:
                facts = json.loads(content)
                if isinstance(facts, list):
                    return [str(fact) for fact in facts if fact]
                else:
                    logger.warning(f"Expected list but got {type(facts)}: {facts}")
                    return [text]
            except json.JSONDecodeError:
                logger.warning(f"Failed to parse JSON response: {content}")
                return [text]
                
        except Exception as e:
            logger.error(f"Error extracting facts: {e}")
            return [text]  # Fallback to original text
    
    def extract_deletion_facts(self, text: str) -> List[str]:
        """
        Extract facts that should be deleted from the given text.
        
        Args:
            text: The input text to extract deletion facts from
            
        Returns:
            List of facts to be deleted
        """
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": """Extract facts that should be deleted from the user's message.
                        Return only a JSON array of strings, each representing one fact to delete.
                        
                        Examples:
                        - Input: "I don't use Magnet anymore"
                        - Output: ["User uses Magnet"]
                        
                        - Input: "I stopped using Shram"
                        - Output: ["User uses Shram"]
                        
                        - Input: "I no longer live in New York"
                        - Output: ["User lives in New York"]
                        
                        Extract the positive version of what the user is saying they no longer do/have."""
                    },
                    {"role": "user", "content": f"Extract deletion facts from: '{text}'"}
                ],
                temperature=0.1,
                max_tokens=500
            )
            
            content = response.choices[0].message.content.strip()
            
            try:
                facts = json.loads(content)
                if isinstance(facts, list):
                    return [str(fact) for fact in facts if fact]
                else:
                    logger.warning(f"Expected list but got {type(facts)}: {facts}")
                    return [text]
            except json.JSONDecodeError:
                logger.warning(f"Failed to parse JSON response: {content}")
                return [text]
                
        except Exception as e:
            logger.error(f"Error extracting deletion facts: {e}")
            return [text]
