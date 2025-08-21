"""
Memory management for the ARAG system.

This module handles the persistent memory that maintains context and accumulated knowledge
across retrieval iterations and throughout the user session.
"""

from typing import Dict, List, Any, Optional
import json
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

class Memory:
    """
    Memory management for maintaining context and accumulated knowledge.
    
    Attributes:
        session_id: Unique identifier for the current session
        user_query: Original user query
        optimized_queries: List of optimized search queries
        retrieved_documents: List of documents retrieved from the vector database
        extracted_knowledge: Accumulated knowledge extracted from documents
        missing_info_references: References to external information identified
        knowledge_gaps: Identified gaps in the current knowledge
        decision_history: History of decisions made by the decision agent
        answer_history: History of generated answers
        evaluation_history: History of answer evaluations
        improvement_history: History of answer improvements
        process_narratives: Narratives about the system's internal processes
    """
    
    def __init__(self, session_id: str, user_query: str):
        """
        Initialize the memory for a new session.
        
        Args:
            session_id: Unique identifier for the session
            user_query: Original user query
        """
        self.session_id = session_id
        self.created_at = datetime.now().isoformat()
        self.user_query = user_query
        self.optimized_queries = []
        self.retrieved_documents = []
        self.extracted_knowledge = []
        self.missing_info_references = []
        self.knowledge_gaps = []
        self.decision_history = []
        self.answer_history = []
        self.evaluation_history = []
        self.improvement_history = []
        self.process_narratives = []
        self.iteration_count = 0
        
        logger.info(f"Memory initialized for session {session_id}")
    
    def add_optimized_queries(self, queries: List[str]) -> None:
        """
        Add optimized queries to memory.
        
        Args:
            queries: List of optimized search queries
        """
        self.optimized_queries.extend(queries)
        logger.debug(f"Added {len(queries)} optimized queries to memory")
    
    def add_retrieved_documents(self, documents: List[Dict[str, Any]]) -> None:
        """
        Add retrieved documents to memory.
        
        Args:
            documents: List of document chunks with metadata
        """
        self.retrieved_documents.extend(documents)
        logger.debug(f"Added {len(documents)} documents to memory")
    
    def add_extracted_knowledge(self, knowledge_items: List[Dict[str, Any]]) -> None:
        """
        Add extracted knowledge items to memory.
        
        Args:
            knowledge_items: List of structured knowledge items
        """
        # Check for duplicates to avoid redundancy
        existing_sources = {item.get('source'): True for item in self.extracted_knowledge}
        new_items = [item for item in knowledge_items if item.get('source') not in existing_sources]
        
        self.extracted_knowledge.extend(new_items)
        logger.debug(f"Added {len(new_items)} knowledge items to memory")
    
    def add_missing_info_references(self, references: List[Dict[str, Any]]) -> None:
        """
        Add missing information references to memory.
        
        Args:
            references: List of references to external information
        """
        self.missing_info_references.extend(references)
        logger.debug(f"Added {len(references)} missing info references to memory")
    
    def add_knowledge_gaps(self, gaps: List[Dict[str, Any]]) -> None:
        """
        Add identified knowledge gaps to memory.
        
        Args:
            gaps: List of identified knowledge gaps
        """
        self.knowledge_gaps = gaps  # Replace current gaps with the latest assessment
        logger.debug(f"Updated knowledge gaps in memory with {len(gaps)} items")
    
    def add_decision(self, decision: Dict[str, Any]) -> None:
        """
        Add a decision to memory.
        
        Args:
            decision: Decision made by the decision agent
        """
        self.decision_history.append(decision)
        logger.debug(f"Added decision to memory: {decision.get('action', 'unknown')}")
    
    def add_answer(self, answer: Dict[str, Any]) -> None:
        """
        Add a generated answer to memory.
        
        Args:
            answer: Generated answer
        """
        self.answer_history.append(answer)
        logger.debug(f"Added answer to memory (length: {len(answer.get('text', ''))})")
    
    def add_evaluation(self, evaluation: Dict[str, Any]) -> None:
        """
        Add an answer evaluation to memory.
        
        Args:
            evaluation: Evaluation of the generated answer
        """
        self.evaluation_history.append(evaluation)
        logger.debug(f"Added evaluation to memory with score: {evaluation.get('score', 'unknown')}")
    
    def add_improvement(self, improvement: Dict[str, Any]) -> None:
        """
        Add an answer improvement to memory.
        
        Args:
            improvement: Improved answer
        """
        self.improvement_history.append(improvement)
        logger.debug(f"Added improved answer to memory (length: {len(improvement.get('text', ''))})")
    
    def add_process_narrative(self, narrative: str) -> None:
        """
        Add a process narrative to memory.
        
        Args:
            narrative: Narrative about the system's internal process
        """
        self.process_narratives.append({
            "timestamp": datetime.now().isoformat(),
            "text": narrative
        })
        logger.debug(f"Added process narrative to memory (length: {len(narrative)})")
    
    def increment_iteration(self) -> None:
        """Increment the iteration count."""
        self.iteration_count += 1
        logger.debug(f"Incremented iteration count to {self.iteration_count}")
    
    def get_current_state(self) -> Dict[str, Any]:
        """
        Get the current state of all memory components.
        
        Returns:
            Dictionary with the current state of memory
        """
        return {
            "session_id": self.session_id,
            "created_at": self.created_at,
            "user_query": self.user_query,
            "iteration_count": self.iteration_count,
            "optimized_queries": self.optimized_queries,
            "retrieved_documents_count": len(self.retrieved_documents),
            "extracted_knowledge_count": len(self.extracted_knowledge),
            "missing_info_references_count": len(self.missing_info_references),
            "knowledge_gaps_count": len(self.knowledge_gaps),
            "decision_history_count": len(self.decision_history),
            "answer_history_count": len(self.answer_history),
            "evaluation_history_count": len(self.evaluation_history),
            "improvement_history_count": len(self.improvement_history),
            "process_narratives_count": len(self.process_narratives)
        }
    
    def get_latest_answer(self) -> Optional[Dict[str, Any]]:
        """
        Get the latest answer from memory.
        
        Returns:
            The latest answer, or None if no answers exist
        """
        if self.improvement_history:
            return self.improvement_history[-1]
        elif self.answer_history:
            return self.answer_history[-1]
        return None
    
    def get_all_extracted_knowledge(self) -> List[Dict[str, Any]]:
        """
        Get all extracted knowledge items from memory.
        
        Returns:
            List of all extracted knowledge items
        """
        return self.extracted_knowledge
    
    def save_to_file(self, filepath: str) -> None:
        """
        Save the memory state to a file.
        
        Args:
            filepath: Path to the file where the memory will be saved
        """
        try:
            with open(filepath, 'w') as f:
                json.dump(self.__dict__, f, indent=2)
            logger.info(f"Memory saved to {filepath}")
        except Exception as e:
            logger.error(f"Failed to save memory to {filepath}: {str(e)}")
    
    @classmethod
    def load_from_file(cls, filepath: str) -> 'Memory':
        """
        Load memory from a file.
        
        Args:
            filepath: Path to the file from which to load memory
            
        Returns:
            Memory instance with the loaded state
        """
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
            
            # Create a new instance with the session_id and user_query
            memory = cls(data.get('session_id', ''), data.get('user_query', ''))
            
            # Update attributes with loaded data
            for key, value in data.items():
                if hasattr(memory, key):
                    setattr(memory, key, value)
            
            logger.info(f"Memory loaded from {filepath}")
            return memory
        except Exception as e:
            logger.error(f"Failed to load memory from {filepath}: {str(e)}")
            raise