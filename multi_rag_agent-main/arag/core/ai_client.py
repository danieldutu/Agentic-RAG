"""
Google AI API client for the ARAG system.

This module handles interactions with Google's Generative AI models (Gemini) for all agent functions.
"""

import logging
import time
import google.generativeai as genai
from typing import Dict, Any, List, Optional, Union
import json

from arag.config import GOOGLE_API_KEY

logger = logging.getLogger(__name__)

# Configure the Google AI client
genai.configure(api_key=GOOGLE_API_KEY)

class AIClient:
    """
    Client for interacting with Google's Generative AI models.
    
    This client provides methods to generate text using Google's Gemini models
    for various agent functions in the ARAG system.
    """
    
    def __init__(self):
        """Initialize the Google AI client."""
        if not GOOGLE_API_KEY:
            logger.error("GOOGLE_API_KEY not set or empty")
            raise ValueError("GOOGLE_API_KEY must be set")
        
        logger.info("Google AI client initialized")
        self.models = {}  # Cache for loaded models
    
    def _get_model(self, model_name: str):
        """
        Get a model instance, creating it if it doesn't exist.
        
        Args:
            model_name: Name of the model to get
            
        Returns:
            Model instance
        """
        if model_name not in self.models:
            self.models[model_name] = genai.GenerativeModel(model_name)
        return self.models[model_name]
    
    def generate(
        self,
        prompt: str,
        model_name: str = "gemini-1.5-pro",
        max_retries: int = 3,
        retry_delay: float = 1.0,
        temperature: float = 0.2,
        top_p: float = 0.95,
        top_k: int = 40,
        max_output_tokens: int = 8192,
        structured_output: Optional[Dict[str, Any]] = None
    ) -> Union[str, Dict[str, Any]]:
        """
        Generate text using a Google Generative AI model.
        
        Args:
            prompt: Prompt to send to the model
            model_name: Name of the model to use
            max_retries: Maximum number of retry attempts
            retry_delay: Delay in seconds between retries
            temperature: Temperature for generation
            top_p: Top-p for nucleus sampling
            top_k: Top-k for sampling
            max_output_tokens: Maximum number of tokens to generate
            structured_output: Schema for structured output, if desired
            
        Returns:
            Generated text or structured output as a dictionary
        """
        model = self._get_model(model_name)
        
        generation_config = {
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
            "max_output_tokens": max_output_tokens,
        }
        
        # DISABLED: Not using response_schema at all due to API compatibility issues
        # if structured_output:
        #     generation_config["response_schema"] = structured_output
        
        for attempt in range(1, max_retries + 1):
            try:
                logger.debug(f"Generating with {model_name}, prompt: {prompt[:100]}...")
                
                # Log the generation request
                logger.debug(f"Generation config: {generation_config}")
                
                # Generate the response
                response = model.generate_content(
                    prompt,
                    generation_config=generation_config
                )
                
                # Handle the response
                if structured_output:
                    # Just return the text even if structured output was requested
                    # This avoids the 400 error with response_schema
                    logger.warning("Structured output was requested but disabled due to API compatibility issues")
                    return response.text
                else:
                    # Return raw text for non-structured output
                    return response.text
                
            except Exception as e:
                logger.error(f"Error generating with {model_name}: {str(e)}")
                
                if attempt < max_retries:
                    wait_time = retry_delay * attempt
                    logger.info(f"Retrying in {wait_time}s (attempt {attempt}/{max_retries})")
                    time.sleep(wait_time)
                else:
                    logger.error(f"Generation failed after {max_retries} attempts")
                    raise
        
        # This should not be reached due to the exception in the loop
        raise RuntimeError("All generation attempts failed")
    
    def generate_query_rewrites(
        self,
        original_query: str,
        model_name: str = "gemini-1.5-pro",
        max_variations: int = 3
    ) -> List[str]:
        """
        Generate rewritten queries for the Query Rewriter Agent.
        
        Args:
            original_query: Original user query
            model_name: Name of the model to use
            max_variations: Maximum number of query variations to generate
            
        Returns:
            List of rewritten query strings
        """
        prompt = f"""
        You are a Query Rewriter Agent in a technical documentation retrieval system.
        Your task is to transform the user question into {max_variations} optimized search queries 
        to maximize retrieval effectiveness.
        
        Analyze technical terminology, identify key concepts, and generate variations of the query
        to ensure comprehensive coverage.
        
        Original query: "{original_query}"
        
        Output exactly {max_variations} different search queries, each focusing on a different aspect
        of the original query. Return only the rewritten queries, one per line.
        """
        
        try:
            result = self.generate(prompt, model_name=model_name)
            
            # Parse the result into individual queries
            queries = [q.strip() for q in result.strip().split('\n') if q.strip()]
            
            # Limit to the requested number
            queries = queries[:max_variations]
            
            if not queries:
                # Fallback to the original query if parsing fails
                logger.warning("Query rewriting failed, falling back to original query")
                return [original_query]
            
            logger.info(f"Generated {len(queries)} query variations")
            return queries
            
        except Exception as e:
            logger.error(f"Error in query rewriting: {str(e)}")
            # Fallback to the original query
            return [original_query]
    
    def extract_knowledge(
        self,
        document_chunks: List[Dict[str, Any]],
        user_query: str,
        model_name: str = "gemini-1.5-pro"
    ) -> List[Dict[str, Any]]:
        """
        Extract structured knowledge from document chunks.
        
        Args:
            document_chunks: List of document chunks to extract knowledge from
            user_query: Original user query for context
            model_name: Name of the model to use
            
        Returns:
            List of extracted knowledge items
        """
        # Process document chunks in batches to prevent context window overflow
        knowledge_items = []
        
        for i, chunk in enumerate(document_chunks):
            chunk_content = chunk.get("content", "")
            chunk_metadata = chunk.get("metadata", {})
            
            prompt = f"""
            You are a Knowledge Extraction Agent in a technical documentation retrieval system.
            Your task is to extract structured knowledge from document chunks that are relevant to the user query.
            
            Follow these knowledge extraction principles:
            1. Identify key facts, specifications, procedures, and technical details
            2. Focus on information that is most relevant to the user query
            3. For visual references, describe the content and purpose of the visual
            4. Highlight safety information when present
            5. Extract atomic, focused knowledge items - each item should capture one specific fact or detail
            
            User query: "{user_query}"
            
            Document chunk: 
            {chunk_content}
            
            Extract relevant knowledge from this document chunk in this format exactly (JSON format):
            {{"knowledge_items": [
              {{"content": "extracted knowledge point 1", 
               "relevance": relevance_score_from_0_to_10,
               "source_type": "text|image|table|code"}}
            ]}}
            
            IMPORTANT: Your response MUST be valid JSON and follow the exact format provided above.
            Do not include any explanations, markdown formatting, or additional text outside of the JSON.
            Make sure to properly escape quotes within the JSON values.
            """
            
            try:
                result = self.generate(
                    prompt,
                    model_name=model_name
                )
                
                # Try to extract useful content even if JSON parsing fails
                try:
                    # First try to parse as is
                    try:
                        parsed_result = json.loads(result)
                    except json.JSONDecodeError:
                        # If that fails, try to find and extract JSON within the text
                        json_start = result.find('{')
                        if json_start >= 0:
                            # Find the matching closing brace
                            json_content = result[json_start:]
                            parsed_result = json.loads(json_content)
                        else:
                            raise json.JSONDecodeError("No JSON object found", result, 0)
                            
                    items = parsed_result.get("knowledge_items", [])
                    
                    # Add document metadata to each knowledge item
                    for item in items:
                        item["source"] = chunk_metadata.get("source", "unknown")
                        item["section"] = chunk_metadata.get("section", "unknown")
                        item["page"] = chunk_metadata.get("page", 0)
                        
                    knowledge_items.extend(items)
                    logger.info(f"Extracted {len(items)} knowledge items from document {chunk_metadata.get('source', 'unknown')}")
                    
                except json.JSONDecodeError as e:
                    logger.warning(f"Failed to parse knowledge extraction result as JSON for document {chunk_metadata.get('source', 'unknown')}: {str(e)}")
                    
                    # Extract useful content from non-JSON response
                    # Look for paragraphs or list items that might contain knowledge
                    content_lines = result.split('\n')
                    extracted_paragraphs = []
                    
                    # Extract actual knowledge content from the response
                    for line in content_lines:
                        # Skip empty lines, headers, and JSON formatting lines
                        line = line.strip()
                        if (len(line) > 15 and 
                            not line.startswith('```') and 
                            not line.startswith('{') and 
                            not line.startswith('}') and
                            not line.startswith('"knowledge_items') and
                            '"content":' not in line and
                            '"relevance":' not in line and
                            '"source_type":' not in line):
                            # This might be actual content
                            extracted_paragraphs.append(line)
                    
                    # Create a fallback knowledge item with the extracted content
                    if extracted_paragraphs:
                        fallback_content = " ".join(extracted_paragraphs)
                        # Truncate if too long
                        if len(fallback_content) > 1000:
                            fallback_content = fallback_content[:997] + "..."
                            
                        fallback_item = {
                            "content": fallback_content,
                            "relevance": 5,  # Medium relevance as we're uncertain
                            "source_type": "text",
                            "source": chunk_metadata.get("source", "unknown"),
                            "section": chunk_metadata.get("section", "unknown"),
                            "page": chunk_metadata.get("page", 0),
                            "is_fallback": True
                        }
                        knowledge_items.append(fallback_item)
                        logger.info(f"Created fallback knowledge item from content extraction for document {chunk_metadata.get('source', 'unknown')}")
                    else:
                        # If we couldn't extract meaningful content, create a simple fallback
                        # Take the first 200 characters from the chunk as a simple fallback
                        content_preview = chunk_content[:min(200, len(chunk_content))]
                        if len(content_preview) < len(chunk_content):
                            content_preview += "..."
                            
                        fallback_item = {
                            "content": content_preview,
                            "relevance": 3,  # Lower relevance for this basic fallback
                            "source_type": "text",
                            "source": chunk_metadata.get("source", "unknown"),
                            "section": chunk_metadata.get("section", "unknown"),
                            "page": chunk_metadata.get("page", 0),
                            "is_fallback": True
                        }
                        knowledge_items.append(fallback_item)
                        logger.info(f"Created minimal fallback knowledge item for document {chunk_metadata.get('source', 'unknown')}")
                    
            except Exception as e:
                logger.error(f"Error in knowledge extraction: {str(e)}")
                
        return knowledge_items
    
    def identify_missing_info(
        self,
        knowledge_items: List[Dict[str, Any]],
        user_query: str,
        model_name: str = "gemini-1.5-pro"
    ) -> List[Dict[str, Any]]:
        """
        Identify missing information in the extracted knowledge.
        
        Args:
            knowledge_items: List of extracted knowledge items
            user_query: Original user query for context
            model_name: Name of the model to use
            
        Returns:
            List of missing information references
        """
        if not knowledge_items:
            logger.warning("No knowledge items provided for missing info identification")
            return []
            
        # Format knowledge items for the prompt
        knowledge_text = ""
        for i, item in enumerate(knowledge_items):
            knowledge_text += f"Item {i+1}: {item.get('content', '')} (Source: {item.get('source', 'unknown')})\n"
        
        prompt = f"""
        You are a Missing Information Agent in a technical documentation retrieval system.
        Your task is to identify missing information in the extracted knowledge that would be helpful
        to provide a comprehensive answer to the user's query.
        
        Follow these missing information identification principles:
        1. Focus on gaps directly related to the user's query
        2. Identify information that is referenced but not provided
        3. Look for partial explanations that need elaboration
        4. Consider prerequisites for understanding the content
        5. Identify technical details that are missing
        
        User query: "{user_query}"
        
        Extracted knowledge:
        {knowledge_text}
        
        Identify missing information references that would be helpful to answer the query.
        Return in this format exactly (JSON format):
        {{"missing_references": [
          {{"reference": "specific missing information", 
           "importance": importance_score_from_1_to_10,
           "context": "context in which this was referenced"}}
        ]}}
        
        IMPORTANT: Your response MUST be valid JSON and follow the exact format provided above.
        Do not include any explanations, markdown formatting, or additional text outside of the JSON.
        """
        
        try:
            result = self.generate(
                prompt,
                model_name=model_name
            )
            
            # Try to extract JSON from the response
            try:
                # First try to parse as is
                try:
                    parsed_result = json.loads(result)
                except json.JSONDecodeError:
                    # If that fails, try to find and extract JSON within the text
                    json_start = result.find('{')
                    if json_start >= 0:
                        # Find the matching closing brace
                        json_content = result[json_start:]
                        parsed_result = json.loads(json_content)
                    else:
                        raise json.JSONDecodeError("No JSON object found", result, 0)
                
                if isinstance(parsed_result, dict) and "missing_references" in parsed_result:
                    logger.info(f"Identified {len(parsed_result['missing_references'])} missing information references")
                    return parsed_result["missing_references"]
                else:
                    logger.warning("Unexpected format from missing info identification")
                    return []
            except json.JSONDecodeError:
                logger.warning("Failed to parse missing info identification result as JSON")
                return []
                
        except Exception as e:
            logger.error(f"Error in missing info identification: {str(e)}")
            return []

    def identify_knowledge_gaps(
        self,
        knowledge_items: List[Dict[str, Any]],
        user_query: str,
        model_name: str = "gemini-1.5-pro"
    ) -> Dict[str, Any]:
        """
        Identify knowledge gaps in the extracted knowledge.
        
        Args:
            knowledge_items: List of extracted knowledge items
            user_query: Original user query for context
            model_name: Name of the model to use
            
        Returns:
            Dictionary with knowledge gaps, completeness score, and reflection
        """
        if not knowledge_items:
            logger.warning("No knowledge items provided for knowledge gaps identification")
            return {
                "knowledge_gaps": [],
                "completeness_score": 0,
                "reflection": "No knowledge items to analyze."
            }
            
        # Format knowledge items for the prompt
        knowledge_text = ""
        for i, item in enumerate(knowledge_items):
            knowledge_text += f"Item {i+1}: {item.get('content', '')} (Source: {item.get('source', 'unknown')})\n"
        
        prompt = f"""
        You are a Knowledge Gap Analyzer in a technical documentation retrieval system.
        Your task is to identify knowledge gaps in the extracted knowledge that would prevent
        providing a comprehensive answer to the user's query.
        
        Follow these knowledge gap analysis principles:
        1. Evaluate the extracted knowledge against the user's query
        2. Identify key aspects of the query that are not covered
        3. Consider implicit information needs
        4. Look for technical details that are missing
        5. Consider different dimensions of the query
        6. Assess the completeness of the extracted knowledge
        
        User query: "{user_query}"
        
        Extracted knowledge:
        {knowledge_text}
        
        Analyze the knowledge gaps and provide a completeness score.
        Return in this format exactly (JSON format):
        {{"knowledge_gaps": [
          {{"gap": "specific knowledge gap", 
           "importance": importance_score_from_1_to_10}}
        ],
        "completeness_score": score_from_0_to_100,
        "reflection": "brief reflection on the extracted knowledge"
        }}
        
        IMPORTANT: Your response MUST be valid JSON and follow the exact format provided above.
        Do not include any explanations, markdown formatting, or additional text outside of the JSON.
        """
        
        try:
            result = self.generate(
                prompt,
                model_name=model_name
            )
            
            # Try to extract JSON from the response
            try:
                # First try to parse as is
                try:
                    parsed_result = json.loads(result)
                except json.JSONDecodeError:
                    # If that fails, try to find and extract JSON within the text
                    json_start = result.find('{')
                    if json_start >= 0:
                        # Find the matching closing brace
                        json_content = result[json_start:]
                        parsed_result = json.loads(json_content)
                    else:
                        raise json.JSONDecodeError("No JSON object found", result, 0)
                        
                if isinstance(parsed_result, dict) and "knowledge_gaps" in parsed_result:
                    logger.info(f"Identified {len(parsed_result['knowledge_gaps'])} knowledge gaps with completeness score {parsed_result.get('completeness_score', 0)}")
                    return parsed_result
                else:
                    logger.warning("Unexpected format from knowledge gaps identification")
                    return {
                        "knowledge_gaps": [],
                        "completeness_score": 0,
                        "reflection": "Failed to analyze knowledge gaps."
                    }
            except json.JSONDecodeError:
                logger.warning("Failed to parse knowledge gaps identification result as JSON")
                return {
                    "knowledge_gaps": [],
                    "completeness_score": 0,
                    "reflection": "Failed to parse knowledge gaps analysis."
                }
                
        except Exception as e:
            logger.error(f"Error in knowledge gaps identification: {str(e)}")
            return {
                "knowledge_gaps": [],
                "completeness_score": 0,
                "reflection": f"Error during analysis: {str(e)}"
            }
    
    def make_decision(
        self,
        knowledge_items: List[Dict[str, Any]],
        knowledge_gaps: Dict[str, Any],
        iteration_count: int,
        max_iterations: int = 3,
        min_completeness: float = 70.0,
        model_name: str = "gemini-1.5-pro"
    ) -> Dict[str, Any]:
        """
        Make a decision on whether to continue searching or generate an answer.
        
        Args:
            knowledge_items: List of extracted knowledge items
            knowledge_gaps: Dictionary with knowledge gaps and reflection
            iteration_count: Current iteration count
            max_iterations: Maximum number of iterations allowed
            min_completeness: Minimum completeness score to generate an answer
            model_name: Name of the model to use
            
        Returns:
            Dictionary with action (continue or complete), confidence, and reasoning
        """
        completeness_score = knowledge_gaps.get("completeness_score", 0)
        knowledge_count = len(knowledge_items)
        gaps_count = len(knowledge_gaps.get("knowledge_gaps", []))
        
        # Format knowledge summary
        knowledge_summary = f"Knowledge items: {knowledge_count}\n"
        knowledge_summary += f"Knowledge gaps: {gaps_count}\n"
        knowledge_summary += f"Completeness score: {completeness_score}%\n"
        knowledge_summary += f"Current iteration: {iteration_count}/{max_iterations}\n"
        
        prompt = f"""
        You are a Decision Agent in a technical documentation retrieval system.
        Your task is to decide whether to continue searching for more information or
        to generate an answer with the current knowledge.
        
        User query context:
        {knowledge_summary}
        
        Knowledge gaps reflection:
        {knowledge_gaps.get("reflection", "No reflection available.")}
        
        Make a decision based on:
        1. The current completeness of the knowledge
        2. The number of iterations performed
        3. The importance of identified knowledge gaps
        4. The quality of existing knowledge
        
        Return in this format exactly (JSON format):
        {{"action": "continue|complete", 
         "confidence": confidence_score_from_0_to_100,
         "reasoning": "reasoning for the decision"}}
        
        IMPORTANT: Your response MUST be valid JSON and follow the exact format provided above.
        Do not include any explanations, markdown formatting, or additional text outside of the JSON.
        """
        
        try:
            result = self.generate(
                prompt,
                model_name=model_name
            )
            
            # Try to extract JSON from the response
            try:
                # First try to parse as is
                try:
                    parsed_result = json.loads(result)
                except json.JSONDecodeError:
                    # If that fails, try to find and extract JSON within the text
                    json_start = result.find('{')
                    if json_start >= 0:
                        # Find the matching closing brace
                        json_content = result[json_start:]
                        parsed_result = json.loads(json_content)
                    else:
                        raise json.JSONDecodeError("No JSON object found", result, 0)
                
                if isinstance(parsed_result, dict) and "action" in parsed_result:
                    logger.info(f"Decision: {parsed_result['action']} with confidence {parsed_result.get('confidence', 0)}")
                    return parsed_result
                else:
                    logger.warning("Unexpected format from decision making")
                    # Default decision based on iteration count
                    forced_decision = "complete" if iteration_count >= max_iterations else "continue"
                    return {
                        "action": forced_decision,
                        "confidence": 50,
                        "reasoning": f"Forced decision due to processing error. Iteration {iteration_count}/{max_iterations}."
                    }
            except json.JSONDecodeError:
                logger.warning("Failed to parse decision making result as JSON")
                # Default decision based on iteration count
                forced_decision = "complete" if iteration_count >= max_iterations else "continue"
                return {
                    "action": forced_decision,
                    "confidence": 50,
                    "reasoning": f"Forced decision due to parsing error. Iteration {iteration_count}/{max_iterations}."
                }
                
        except Exception as e:
            logger.error(f"Error in decision making: {str(e)}")
            # Default decision based on iteration count
            forced_decision = "complete" if iteration_count >= max_iterations else "continue"
            return {
                "action": forced_decision,
                "confidence": 50,
                "reasoning": f"Forced decision due to error: {str(e)}. Iteration {iteration_count}/{max_iterations}."
            }

    def generate_answer(
        self,
        knowledge_items: List[Dict[str, Any]],
        user_query: str,
        model_name: str = "gemini-1.5-pro"
    ) -> Dict[str, Any]:
        """
        Generate an answer based on the extracted knowledge.
        
        Args:
            knowledge_items: List of extracted knowledge items
            user_query: Original user query
            model_name: Name of the model to use
            
        Returns:
            Dictionary with answer text, citations, and sections
        """
        if not knowledge_items:
            logger.warning("No knowledge items provided for answer generation")
            return {
                "text": "I don't have enough information to answer your question. Please try a different query or provide more context.",
                "citations": [],
                "sections": []
            }
            
        # Format knowledge items for the prompt with citation indices
        knowledge_text = ""
        for i, item in enumerate(knowledge_items):
            knowledge_text += f"[{i+1}] {item.get('content', '')}\nSource: {item.get('source', 'unknown')}\n\n"
        
        prompt = f"""
        You are an Answer Generation Agent in a technical documentation retrieval system.
        Your task is to generate a comprehensive, accurate answer based on the provided knowledge items.
        
        Follow these answer generation principles:
        1. Only use information provided in the knowledge items
        2. Do not make up information or rely on your general knowledge
        3. Organize the answer logically with appropriate sections if needed
        4. Cite sources using the knowledge item numbers [1], [2], etc.
        5. Be comprehensive but concise
        6. Maintain technical accuracy
        7. Use appropriate formatting for readability
        8. If the knowledge is insufficient, acknowledge this limitation
        
        User query: "{user_query}"
        
        Knowledge items:
        {knowledge_text}
        
        Generate a comprehensive answer with appropriate citations.
        Return in this format exactly (JSON format):
        {{"answer": "comprehensive answer text with [1], [2] citations embedded",
         "citations": [
           {{"text": "cited text", 
            "source": "source reference",
            "citation_index": item_number}}
         ],
         "sections": [
           {{"title": "section title", 
            "content": "section content"}}
         ]
        }}
        
        IMPORTANT: Your response MUST be valid JSON and follow the exact format provided above.
        Do not include any explanations, markdown formatting, or additional text outside of the JSON.
        """
        
        try:
            result = self.generate(
                prompt,
                model_name=model_name,
                temperature=0.3  # Slightly higher temperature for better answer generation
            )
            
            # Try to extract JSON from the response
            try:
                # First try to parse as is
                try:
                    parsed_result = json.loads(result)
                except json.JSONDecodeError:
                    # If that fails, try to find and extract JSON within the text
                    json_start = result.find('{')
                    if json_start >= 0:
                        # Find the matching closing brace
                        json_content = result[json_start:]
                        parsed_result = json.loads(json_content)
                    else:
                        raise json.JSONDecodeError("No JSON object found", result, 0)
                
                if isinstance(parsed_result, dict) and "answer" in parsed_result:
                    logger.info(f"Generated answer with {len(parsed_result.get('citations', []))} citations")
                    return {
                        "text": parsed_result["answer"],
                        "citations": parsed_result.get("citations", []),
                        "sections": parsed_result.get("sections", [])
                    }
                else:
                    logger.warning("Unexpected format from answer generation")
                    # Extract content from the non-JSON response
                    answer_text = self._extract_answer_from_text(result)
                    if answer_text:
                        return {
                            "text": answer_text,
                            "citations": [],
                            "sections": []
                        }
                    # If we have knowledge items but parsing failed, try to return a simple answer
                    elif knowledge_items:
                        return self._create_fallback_answer(knowledge_items, user_query)
                    else:
                        return {
                            "text": "I'm sorry, but I couldn't generate a proper answer due to a processing error.",
                            "citations": [],
                            "sections": []
                        }
            except json.JSONDecodeError:
                logger.warning("Failed to parse answer generation result as JSON")
                # Extract content from the non-JSON response
                answer_text = self._extract_answer_from_text(result)
                if answer_text:
                    return {
                        "text": answer_text,
                        "citations": [],
                        "sections": []
                    }
                # If we have knowledge items but parsing failed, try to return a simple answer
                elif knowledge_items:
                    return self._create_fallback_answer(knowledge_items, user_query)
                else:
                    return {
                        "text": "I'm sorry, but I couldn't generate a proper answer due to a parsing error.",
                        "citations": [],
                        "sections": []
                    }
                
        except Exception as e:
            logger.error(f"Error in answer generation: {str(e)}")
            if knowledge_items:
                return self._create_fallback_answer(knowledge_items, user_query)
            else:
                return {
                    "text": f"I'm sorry, but I couldn't generate an answer due to an error: {str(e)}",
                    "citations": [],
                    "sections": []
                }

    def _create_fallback_answer(self, knowledge_items: List[Dict[str, Any]], user_query: str) -> Dict[str, Any]:
        """
        Create a fallback answer directly from knowledge items when answer generation fails.
        
        Args:
            knowledge_items: List of extracted knowledge items
            user_query: Original user query
            
        Returns:
            Dictionary with answer text, citations, and sections
        """
        # Sort by relevance to get most relevant items first
        sorted_items = sorted(knowledge_items, key=lambda x: x.get("relevance", 0), reverse=True)
        
        # Get top 5 most relevant items (if available)
        top_items = sorted_items[:min(5, len(sorted_items))]
        
        # Construct a simple answer from these items
        answer_paragraphs = []
        for i, item in enumerate(top_items):
            # Add content from each top item
            content = item.get("content", "").strip()
            if content:
                answer_paragraphs.append(f"{content} (Source: {item.get('source', 'unknown')})")
        
        if answer_paragraphs:
            # Combine paragraphs with proper spacing
            answer_text = "\n\n".join(answer_paragraphs)
            
            # Add introduction
            intro = f"Based on the available information about {user_query}, I found the following relevant details:\n\n"
            
            # Add disclaimer
            disclaimer = "\n\nThis information has been directly extracted from the available documents. The answer may be incomplete as it was generated without full processing."
            
            final_answer = intro + answer_text + disclaimer
            
            # Create citations
            citations = []
            for i, item in enumerate(top_items):
                if item.get("content"):
                    citations.append({
                        "text": item.get("content", "")[:100] + "...",  # Truncate long citations
                        "source": item.get("source", "unknown"),
                        "citation_index": i + 1
                    })
            
            return {
                "text": final_answer,
                "citations": citations,
                "sections": []
            }
        else:
            # Fallback to generic message
            return {
                "text": "Based on the available information, I can provide a partial answer, but there may be gaps in the response due to processing limitations.",
                "citations": [],
                "sections": []
            }
    
    def _extract_answer_from_text(self, text: str) -> str:
        """
        Extract a usable answer from non-JSON text response.
        
        Args:
            text: The raw text response from the model
            
        Returns:
            Extracted answer text or empty string if extraction fails
        """
        # Remove any code blocks, JSON formatting
        lines = text.split('\n')
        content_lines = []
        in_code_block = False
        
        for line in lines:
            if line.strip().startswith('```'):
                in_code_block = not in_code_block
                continue
                
            if not in_code_block and not line.strip().startswith('{') and not line.strip().startswith('}'):
                # Skip JSON formatting lines and other noise
                if ('"answer":' not in line and 
                    '"citations":' not in line and 
                    '"sections":' not in line and
                    'IMPORTANT:' not in line):
                    content_line = line.strip()
                    if content_line:
                        content_lines.append(content_line)
        
        # If we found content, join it together
        if content_lines:
            # Look for patterns that might indicate the start of an actual answer
            answer_start_indices = []
            for i, line in enumerate(content_lines):
                # Common patterns that might indicate the start of an answer
                if any(pattern in line.lower() for pattern in 
                       ["based on", "according to", "the answer", "in summary", 
                        "to answer your", "regarding your", "in response to"]):
                    answer_start_indices.append(i)
            
            # If we found potential answer start points, use the earliest one
            if answer_start_indices:
                start_idx = min(answer_start_indices)
                answer_text = " ".join(content_lines[start_idx:])
            else:
                # Otherwise use all the content
                answer_text = " ".join(content_lines)
                
            # Truncate if too long
            if len(answer_text) > 2000:
                answer_text = answer_text[:1997] + "..."
                
            return answer_text
        
        return ""
    
    def evaluate_answer(
        self,
        answer: Dict[str, Any],
        knowledge_items: List[Dict[str, Any]],
        user_query: str,
        model_name: str = "gemini-1.5-pro"
    ) -> Dict[str, Any]:
        """
        Evaluate the quality of the generated answer.
        
        Args:
            answer: Generated answer to evaluate
            knowledge_items: Knowledge items used to generate the answer
            user_query: Original user query
            model_name: Name of the model to use
            
        Returns:
            Dictionary with evaluation scores and feedback
        """
        # We're not using structured output due to API compatibility issues
        # structured_output = {...}
        
        prompt = f"""
        You are an Evaluator Agent in a technical documentation retrieval system.
        Your task is to assess the quality of the generated answer against predefined criteria.
        
        User query: "{user_query}"
        
        Answer to evaluate: 
        {answer.get('text', '')}
        
        Evaluate the answer on these criteria, rating each from 1-10:
        1. Accuracy: Does the answer contain only factual information from the knowledge?
        2. Completeness: Does the answer address all aspects of the query?
        3. Relevance: Is the information directly relevant to the query?
        4. Clarity: Is the answer well-structured and easy to understand?
        5. Citations: Are sources properly cited for factual statements?
        
        Also provide specific feedback on areas of improvement and list key strengths.
        
        Return in this format exactly (JSON format):
        {{
          "scores": {{
            "accuracy": accuracy_score_from_1_to_10,
            "completeness": completeness_score_from_1_to_10,
            "relevance": relevance_score_from_1_to_10,
            "clarity": clarity_score_from_1_to_10,
            "citations": citations_score_from_1_to_10,
            "overall": overall_score_from_1_to_10
          }},
          "feedback": [
            {{"aspect": "aspect name", "issue": "issue description", "suggestion": "suggestion for improvement"}}
          ],
          "strengths": [
            "strength description"
          ]
        }}
        """
        
        try:
            result = self.generate(
                prompt,
                model_name=model_name
            )
            
            # Try to parse the result as JSON
            try:
                parsed_result = json.loads(result)
                if isinstance(parsed_result, dict) and "scores" in parsed_result:
                    logger.info(f"Evaluation complete with overall score: {parsed_result['scores'].get('overall', 0)}")
                    return parsed_result
                else:
                    logger.warning("Unexpected format from answer evaluation")
                    return {
                        "scores": {
                            "accuracy": 5,
                            "completeness": 5,
                            "relevance": 5,
                            "clarity": 5,
                            "citations": 5,
                            "overall": 5
                        },
                        "feedback": [
                            {
                                "aspect": "processing",
                                "issue": "Evaluation failed",
                                "suggestion": "Please review the answer manually"
                            }
                        ],
                        "strengths": []
                    }
            except json.JSONDecodeError:
                logger.warning("Failed to parse answer evaluation result as JSON")
                return {
                    "scores": {
                        "accuracy": 5,
                        "completeness": 5,
                        "relevance": 5,
                        "clarity": 5,
                        "citations": 5,
                        "overall": 5
                    },
                    "feedback": [
                        {
                            "aspect": "processing",
                            "issue": "Evaluation parsing failed",
                            "suggestion": "Please review the answer manually"
                        }
                    ],
                    "strengths": []
                }
                
        except Exception as e:
            logger.error(f"Error in answer evaluation: {str(e)}")
            return {
                "scores": {
                    "accuracy": 5,
                    "completeness": 5,
                    "relevance": 5,
                    "clarity": 5,
                    "citations": 5,
                    "overall": 5
                },
                "feedback": [
                    {
                        "aspect": "processing",
                        "issue": f"Evaluation error: {str(e)}",
                        "suggestion": "Please review the answer manually"
                    }
                ],
                "strengths": []
            }
    
    def improve_answer(
        self,
        original_answer: Dict[str, Any],
        evaluation: Dict[str, Any],
        knowledge_items: List[Dict[str, Any]],
        user_query: str,
        model_name: str = "gemini-1.5-pro"
    ) -> Dict[str, Any]:
        """
        Improve the answer based on evaluation feedback.
        
        Args:
            original_answer: Original answer to improve
            evaluation: Evaluation with scores and feedback
            knowledge_items: Knowledge items to use for the improvement
            user_query: Original user query
            model_name: Name of the model to use
            
        Returns:
            Dictionary with the improved answer and metadata
        """
        # Extract feedback and scores
        feedback = evaluation.get("feedback", [])
        feedback_text = "\n".join([
            f"- {item.get('aspect', '')}: {item.get('issue', '')}. Suggestion: {item.get('suggestion', '')}"
            for item in feedback
        ])
        
        scores = evaluation.get("scores", {})
        scores_text = "\n".join([
            f"- {key}: {value}/10"
            for key, value in scores.items()
        ])
        
        # We're not using structured output due to API compatibility issues
        # structured_output = {...}
        
        prompt = f"""
        You are an Improver Agent in a technical documentation retrieval system.
        Your task is to enhance the answer based on evaluation feedback while maintaining
        strict grounding in the provided knowledge.
        
        User query: "{user_query}"
        
        Original answer:
        {original_answer.get('text', '')}
        
        Evaluation scores:
        {scores_text}
        
        Feedback:
        {feedback_text}
        
        Improve the answer by:
        1. Addressing all feedback points
        2. Maintaining 100% accuracy (no invented information)
        3. Improving structure and readability
        4. Ensuring proper citations
        5. Enhancing completeness
        
        Do not remove accurate information from the original answer, only enhance and refine it.
        
        Return in this format exactly (JSON format):
        {{
          "improved_answer": "improved comprehensive answer text",
          "citations": [
            {{"text": "cited text", "source": "source reference"}}
          ],
          "sections": [
            {{"title": "section title", "content": "section content"}}
          ],
          "improvements_made": [
            "improvement description"
          ]
        }}
        """
        
        try:
            result = self.generate(
                prompt,
                model_name=model_name,
                temperature=0.3  # Slightly higher temperature for better improvement
            )
            
            # Try to parse the result as JSON
            try:
                parsed_result = json.loads(result)
                if isinstance(parsed_result, dict) and "improved_answer" in parsed_result:
                    logger.info(f"Improved answer with {len(parsed_result.get('improvements_made', []))} improvements")
                    return {
                        "text": parsed_result["improved_answer"],
                        "citations": parsed_result.get("citations", original_answer.get("citations", [])),
                        "sections": parsed_result.get("sections", original_answer.get("sections", [])),
                        "improvements": parsed_result.get("improvements_made", [])
                    }
                else:
                    logger.warning("Unexpected format from answer improvement")
                    return original_answer  # Return the original answer if improvement fails
            except json.JSONDecodeError:
                logger.warning("Failed to parse answer improvement result as JSON")
                return original_answer  # Return the original answer if improvement fails
                
        except Exception as e:
            logger.error(f"Error in answer improvement: {str(e)}")
            return original_answer  # Return the original answer if improvement fails
    
    def generate_process_narrative(
        self,
        stage: str,
        context: Dict[str, Any],
        model_name: str = "gemini-1.5-flash"  # Use a faster model for this
    ) -> str:
        """
        Generate a narrative about the system's internal processes.
        
        Args:
            stage: Current stage in the pipeline
            context: Context information for the narrative
            model_name: Name of the model to use
            
        Returns:
            Process narrative as a string
        """
        # Construct a context-specific prompt based on the stage
        if stage == "query_rewriting":
            narrative_prompt = f"""
            You are a Process Agent in a technical documentation retrieval system.
            Your task is to provide a first-person narrative about the system's internal processes.
            
            Current stage: Query Rewriting
            
            Original query: "{context.get('original_query', '')}"
            Rewritten queries: {context.get('rewritten_queries', [])}
            
            Provide a brief, action-centered narrative about what the system is doing at this stage
            and why. Use concise, first-person language that makes the process transparent to users.
            Maximum 3 sentences.
            """
        elif stage == "document_retrieval":
            narrative_prompt = f"""
            You are a Process Agent in a technical documentation retrieval system.
            Your task is to provide a first-person narrative about the system's internal processes.
            
            Current stage: Document Retrieval
            
            Queries used: {context.get('queries', [])}
            Documents retrieved: {context.get('document_count', 0)}
            
            Provide a brief, action-centered narrative about what the system is doing at this stage
            and why. Use concise, first-person language that makes the process transparent to users.
            Maximum 3 sentences.
            """
        elif stage == "knowledge_extraction":
            narrative_prompt = f"""
            You are a Process Agent in a technical documentation retrieval system.
            Your task is to provide a first-person narrative about the system's internal processes.
            
            Current stage: Knowledge Extraction
            
            Documents processed: {context.get('document_count', 0)}
            Knowledge items extracted: {context.get('knowledge_count', 0)}
            
            Provide a brief, action-centered narrative about what the system is doing at this stage
            and why. Use concise, first-person language that makes the process transparent to users.
            Maximum 3 sentences.
            """
        elif stage == "missing_info":
            narrative_prompt = f"""
            You are a Process Agent in a technical documentation retrieval system.
            Your task is to provide a first-person narrative about the system's internal processes.
            
            Current stage: Missing Information Identification
            
            References found: {context.get('reference_count', 0)}
            
            Provide a brief, action-centered narrative about what the system is doing at this stage
            and why. Use concise, first-person language that makes the process transparent to users.
            Maximum 3 sentences.
            """
        elif stage == "knowledge_gaps":
            narrative_prompt = f"""
            You are a Process Agent in a technical documentation retrieval system.
            Your task is to provide a first-person narrative about the system's internal processes.
            
            Current stage: Knowledge Gap Analysis
            
            Gaps identified: {context.get('gap_count', 0)}
            Completeness score: {context.get('completeness_score', 0)}%
            
            Provide a brief, action-centered narrative about what the system is doing at this stage
            and why. Use concise, first-person language that makes the process transparent to users.
            Maximum 3 sentences.
            """
        elif stage == "decision":
            narrative_prompt = f"""
            You are a Process Agent in a technical documentation retrieval system.
            Your task is to provide a first-person narrative about the system's internal processes.
            
            Current stage: Decision Making
            
            Decision: {context.get('decision', 'unknown')}
            Confidence: {context.get('confidence', 0)}%
            Iteration: {context.get('iteration', 1)} of {context.get('max_iterations', 3)}
            
            Provide a brief, action-centered narrative about what the system is doing at this stage
            and why. Use concise, first-person language that makes the process transparent to users.
            Maximum 3 sentences.
            """
        elif stage == "answer_generation":
            narrative_prompt = f"""
            You are a Process Agent in a technical documentation retrieval system.
            Your task is to provide a first-person narrative about the system's internal processes.
            
            Current stage: Answer Generation
            
            Knowledge items used: {context.get('knowledge_count', 0)}
            
            Provide a brief, action-centered narrative about what the system is doing at this stage
            and why. Use concise, first-person language that makes the process transparent to users.
            Maximum 3 sentences.
            """
        elif stage == "evaluation":
            narrative_prompt = f"""
            You are a Process Agent in a technical documentation retrieval system.
            Your task is to provide a first-person narrative about the system's internal processes.
            
            Current stage: Answer Evaluation
            
            Overall score: {context.get('overall_score', 0)}/10
            Feedback items: {context.get('feedback_count', 0)}
            
            Provide a brief, action-centered narrative about what the system is doing at this stage
            and why. Use concise, first-person language that makes the process transparent to users.
            Maximum 3 sentences.
            """
        elif stage == "improvement":
            narrative_prompt = f"""
            You are a Process Agent in a technical documentation retrieval system.
            Your task is to provide a first-person narrative about the system's internal processes.
            
            Current stage: Answer Improvement
            
            Improvements made: {context.get('improvement_count', 0)}
            
            Provide a brief, action-centered narrative about what the system is doing at this stage
            and why. Use concise, first-person language that makes the process transparent to users.
            Maximum 3 sentences.
            """
        else:
            narrative_prompt = f"""
            You are a Process Agent in a technical documentation retrieval system.
            Your task is to provide a first-person narrative about the system's internal processes.
            
            Current stage: {stage}
            
            Provide a brief, action-centered narrative about what the system is doing at this stage
            and why. Use concise, first-person language that makes the process transparent to users.
            Maximum 3 sentences.
            """
        
        try:
            result = self.generate(
                narrative_prompt,
                model_name=model_name,
                temperature=0.4,  # Higher temperature for more natural narratives
                max_output_tokens=1024  # Keep narratives concise
            )
            
            logger.info(f"Generated process narrative for stage: {stage}")
            return result.strip()
                
        except Exception as e:
            logger.error(f"Error generating process narrative for {stage}: {str(e)}")
            return f"I'm now at the {stage} stage of processing your query."