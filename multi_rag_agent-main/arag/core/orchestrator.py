"""
Orchestrator for the ARAG system.

This module coordinates the various agent functions and manages the RAG pipeline flow.
"""

import logging
import uuid
from typing import Dict, List, Any, Optional, Tuple
import time

from arag.core.memory import Memory
from arag.core.vector_db import VectorDBClient
from arag.core.ai_client import AIClient
from arag.config import MAX_RETRIEVAL_ITERATIONS

from arag.agents.query_rewriter import QueryRewriterAgent
from arag.agents.knowledge_extractor import KnowledgeExtractorAgent
from arag.agents.missing_info import MissingInfoAgent
from arag.agents.knowledge_gaps import KnowledgeGapsAgent
from arag.agents.decision import DecisionAgent
from arag.agents.answer import AnswerAgent
from arag.agents.evaluator import EvaluatorAgent
from arag.agents.improver import ImproverAgent
from arag.agents.process import ProcessAgent

logger = logging.getLogger(__name__)

class Orchestrator:
    """
    Orchestrator for coordinating the ARAG pipeline.
    
    This class manages the flow of information between the different agent functions
    and controls the overall RAG process.
    """
    
    def __init__(
        self,
        vector_db: Optional[VectorDBClient] = None,
        ai_client: Optional[AIClient] = None
    ):
        """
        Initialize the orchestrator.
        
        Args:
            vector_db: Vector database client, or None to create a new one
            ai_client: AI client, or None to create a new one
        """
        self.vector_db = vector_db or VectorDBClient()
        self.ai_client = ai_client or AIClient()
        
        # Initialize agents
        self.query_rewriter = QueryRewriterAgent(self.ai_client)
        self.knowledge_extractor = KnowledgeExtractorAgent(self.ai_client)
        self.missing_info = MissingInfoAgent(self.ai_client)
        self.knowledge_gaps = KnowledgeGapsAgent(self.ai_client)
        self.decision = DecisionAgent(self.ai_client)
        self.answer = AnswerAgent(self.ai_client)
        self.evaluator = EvaluatorAgent(self.ai_client)
        self.improver = ImproverAgent(self.ai_client)
        self.process = ProcessAgent(self.ai_client)
        
        logger.info("Orchestrator initialized with all agents")
    
    def process_query(
        self,
        user_query: str,
        session_id: Optional[str] = None,
        max_iterations: int = MAX_RETRIEVAL_ITERATIONS
    ) -> Tuple[Dict[str, Any], Memory]:
        """
        Process a user query through the ARAG pipeline.
        
        Args:
            user_query: Natural language question from the user
            session_id: Optional session ID, or None to generate a new one
            max_iterations: Maximum number of retrieval iterations
            
        Returns:
            Tuple of (final answer, memory) where final answer is a dictionary 
            with the answer text and metadata, and memory contains the full session context
        """
        # Initialize session
        session_id = session_id or str(uuid.uuid4())
        memory = Memory(session_id, user_query)
        
        # Process narrative for initialization
        init_narrative = self.process.generate_narrative(
            "initialization",
            {"user_query": user_query}
        )
        memory.add_process_narrative(init_narrative)
        
        # Main processing loop
        for iteration in range(1, max_iterations + 1):
            logger.info(f"Starting iteration {iteration}/{max_iterations}")
            memory.increment_iteration()
            
            # Step 1: Query Rewriting
            rewritten_queries = self._run_query_rewriting(user_query, memory)
            
            # Step 2: Document Retrieval
            documents = self._run_document_retrieval(rewritten_queries, memory)
            
            # Step 3: Knowledge Extraction
            knowledge_items = self._run_knowledge_extraction(documents, user_query, memory)
            
            # Step 4: Missing Information Identification
            missing_info = self._run_missing_info_identification(
                memory.get_all_extracted_knowledge(), user_query, memory
            )
            
            # Step 5: Knowledge Gap Analysis
            knowledge_gaps = self._run_knowledge_gap_analysis(
                memory.get_all_extracted_knowledge(), user_query, memory
            )
            
            # Step 6: Decision Point
            decision = self._run_decision_making(
                memory.get_all_extracted_knowledge(),
                knowledge_gaps,
                iteration,
                max_iterations,
                memory
            )
            
            # Check if we have enough information to proceed to answer generation
            if decision.get("action") == "complete":
                logger.info(f"Decision to complete after iteration {iteration}")
                break
            
            # If this is the last iteration, force completion
            if iteration == max_iterations:
                logger.info(f"Forced completion after reaching max iterations ({max_iterations})")
                break
        
        # Step 7: Answer Generation
        answer = self._run_answer_generation(
            memory.get_all_extracted_knowledge(), user_query, memory
        )
        
        # Step 8: Answer Evaluation
        evaluation = self._run_answer_evaluation(
            answer, memory.get_all_extracted_knowledge(), user_query, memory
        )
        
        # Step 9: Answer Improvement
        improved_answer = self._run_answer_improvement(
            answer, evaluation, memory.get_all_extracted_knowledge(), user_query, memory
        )
        
        # Return the final answer and memory
        return improved_answer, memory
    
    def _run_query_rewriting(self, user_query: str, memory: Memory) -> List[str]:
        """
        Run the Query Rewriter Agent.
        
        Args:
            user_query: Original user query
            memory: Memory instance for the session
            
        Returns:
            List of rewritten queries
        """
        start_time = time.time()
        logger.info("Running Query Rewriter Agent")
        
        # Generate rewritten queries
        rewritten_queries = self.query_rewriter.rewrite_query(user_query)
        
        # Add to memory
        memory.add_optimized_queries(rewritten_queries)
        
        # Generate process narrative
        narrative = self.process.generate_narrative(
            "query_rewriting",
            {
                "original_query": user_query,
                "rewritten_queries": rewritten_queries
            }
        )
        memory.add_process_narrative(narrative)
        
        logger.info(f"Query rewriting completed in {time.time() - start_time:.2f}s")
        return rewritten_queries
    
    def _run_document_retrieval(self, queries: List[str], memory: Memory) -> List[Dict[str, Any]]:
        """
        Run document retrieval from the vector database.
        
        Args:
            queries: List of search queries
            memory: Memory instance for the session
            
        Returns:
            List of retrieved documents
        """
        start_time = time.time()
        logger.info("Running Document Retrieval")
        
        # Retrieve documents from the vector database
        documents = self.vector_db.batch_query(queries)
        
        # Add to memory
        memory.add_retrieved_documents(documents)
        
        # Generate process narrative
        narrative = self.process.generate_narrative(
            "document_retrieval",
            {
                "queries": queries,
                "document_count": len(documents)
            }
        )
        memory.add_process_narrative(narrative)
        
        logger.info(f"Document retrieval completed in {time.time() - start_time:.2f}s")
        return documents
    
    def _run_knowledge_extraction(
        self, 
        documents: List[Dict[str, Any]], 
        user_query: str, 
        memory: Memory
    ) -> List[Dict[str, Any]]:
        """
        Run the Knowledge Agent.
        
        Args:
            documents: List of document chunks
            user_query: Original user query
            memory: Memory instance for the session
            
        Returns:
            List of extracted knowledge items
        """
        start_time = time.time()
        logger.info("Running Knowledge Agent")
        
        # Extract knowledge from documents
        knowledge_items = self.knowledge_extractor.extract_knowledge(documents, user_query)
        
        # Add to memory
        memory.add_extracted_knowledge(knowledge_items)
        
        # Generate process narrative
        narrative = self.process.generate_narrative(
            "knowledge_extraction",
            {
                "document_count": len(documents),
                "knowledge_count": len(knowledge_items)
            }
        )
        memory.add_process_narrative(narrative)
        
        logger.info(f"Knowledge extraction completed in {time.time() - start_time:.2f}s")
        return knowledge_items
    
    def _run_missing_info_identification(
        self, 
        knowledge_items: List[Dict[str, Any]], 
        user_query: str, 
        memory: Memory
    ) -> List[Dict[str, Any]]:
        """
        Run the Missing Info Agent.
        
        Args:
            knowledge_items: List of knowledge items
            user_query: Original user query
            memory: Memory instance for the session
            
        Returns:
            List of missing information references
        """
        start_time = time.time()
        logger.info("Running Missing Info Agent")
        
        # Identify missing information
        missing_info = self.missing_info.identify_missing_info(knowledge_items, user_query)
        
        # Add to memory
        memory.add_missing_info_references(missing_info)
        
        # Generate process narrative
        narrative = self.process.generate_narrative(
            "missing_info",
            {
                "reference_count": len(missing_info)
            }
        )
        memory.add_process_narrative(narrative)
        
        logger.info(f"Missing info identification completed in {time.time() - start_time:.2f}s")
        return missing_info
    
    def _run_knowledge_gap_analysis(
        self, 
        knowledge_items: List[Dict[str, Any]], 
        user_query: str, 
        memory: Memory
    ) -> Dict[str, Any]:
        """
        Run the Knowledge Gaps Agent.
        
        Args:
            knowledge_items: List of knowledge items
            user_query: Original user query
            memory: Memory instance for the session
            
        Returns:
            Dictionary with knowledge gaps and reflection
        """
        start_time = time.time()
        logger.info("Running Knowledge Gaps Agent")
        
        # Identify knowledge gaps
        knowledge_gaps = self.knowledge_gaps.identify_knowledge_gaps(knowledge_items, user_query)
        
        # Add to memory
        memory.add_knowledge_gaps(knowledge_gaps.get("knowledge_gaps", []))
        
        # Generate process narrative
        narrative = self.process.generate_narrative(
            "knowledge_gaps",
            {
                "gap_count": len(knowledge_gaps.get("knowledge_gaps", [])),
                "completeness_score": knowledge_gaps.get("completeness_score", 0)
            }
        )
        memory.add_process_narrative(narrative)
        
        logger.info(f"Knowledge gap analysis completed in {time.time() - start_time:.2f}s")
        return knowledge_gaps
    
    def _run_decision_making(
        self, 
        knowledge_items: List[Dict[str, Any]],
        knowledge_gaps: Dict[str, Any],
        iteration: int,
        max_iterations: int,
        memory: Memory
    ) -> Dict[str, Any]:
        """
        Run the Decision Agent.
        
        Args:
            knowledge_items: List of knowledge items
            knowledge_gaps: Knowledge gaps analysis
            iteration: Current iteration count
            max_iterations: Maximum number of iterations
            memory: Memory instance for the session
            
        Returns:
            Decision dictionary with action and reasoning
        """
        start_time = time.time()
        logger.info("Running Decision Agent")
        
        # Make decision
        decision = self.decision.make_decision(
            knowledge_items, knowledge_gaps, iteration, max_iterations
        )
        
        # Add to memory
        memory.add_decision(decision)
        
        # Generate process narrative
        narrative = self.process.generate_narrative(
            "decision",
            {
                "decision": decision.get("action", "unknown"),
                "confidence": decision.get("confidence", 0),
                "iteration": iteration,
                "max_iterations": max_iterations
            }
        )
        memory.add_process_narrative(narrative)
        
        logger.info(f"Decision making completed in {time.time() - start_time:.2f}s")
        return decision
    
    def _run_answer_generation(
        self, 
        knowledge_items: List[Dict[str, Any]], 
        user_query: str, 
        memory: Memory
    ) -> Dict[str, Any]:
        """
        Run the Answer Agent.
        
        Args:
            knowledge_items: List of knowledge items
            user_query: Original user query
            memory: Memory instance for the session
            
        Returns:
            Answer dictionary with text and metadata
        """
        start_time = time.time()
        logger.info("Running Answer Agent")
        
        # Generate answer
        answer = self.answer.generate_answer(knowledge_items, user_query)
        
        # Add to memory
        memory.add_answer(answer)
        
        # Generate process narrative
        narrative = self.process.generate_narrative(
            "answer_generation",
            {
                "knowledge_count": len(knowledge_items)
            }
        )
        memory.add_process_narrative(narrative)
        
        logger.info(f"Answer generation completed in {time.time() - start_time:.2f}s")
        return answer
    
    def _run_answer_evaluation(
        self, 
        answer: Dict[str, Any],
        knowledge_items: List[Dict[str, Any]], 
        user_query: str, 
        memory: Memory
    ) -> Dict[str, Any]:
        """
        Run the Evaluator Agent.
        
        Args:
            answer: Generated answer
            knowledge_items: List of knowledge items
            user_query: Original user query
            memory: Memory instance for the session
            
        Returns:
            Evaluation dictionary with scores and feedback
        """
        start_time = time.time()
        logger.info("Running Evaluator Agent")
        
        # Evaluate answer
        evaluation = self.evaluator.evaluate_answer(answer, knowledge_items, user_query)
        
        # Add to memory
        memory.add_evaluation(evaluation)
        
        # Generate process narrative
        narrative = self.process.generate_narrative(
            "evaluation",
            {
                "overall_score": evaluation.get("scores", {}).get("overall", 0),
                "feedback_count": len(evaluation.get("feedback", []))
            }
        )
        memory.add_process_narrative(narrative)
        
        logger.info(f"Answer evaluation completed in {time.time() - start_time:.2f}s")
        return evaluation
    
    def _run_answer_improvement(
        self, 
        answer: Dict[str, Any],
        evaluation: Dict[str, Any],
        knowledge_items: List[Dict[str, Any]], 
        user_query: str, 
        memory: Memory
    ) -> Dict[str, Any]:
        """
        Run the Improver Agent.
        
        Args:
            answer: Original answer
            evaluation: Evaluation of the original answer
            knowledge_items: List of knowledge items
            user_query: Original user query
            memory: Memory instance for the session
            
        Returns:
            Improved answer dictionary with text and metadata
        """
        start_time = time.time()
        logger.info("Running Improver Agent")
        
        # Improve answer
        improved_answer = self.improver.improve_answer(
            answer, evaluation, knowledge_items, user_query
        )
        
        # Add to memory
        memory.add_improvement(improved_answer)
        
        # Generate process narrative
        narrative = self.process.generate_narrative(
            "improvement",
            {
                "improvement_count": len(improved_answer.get("improvements", []))
            }
        )
        memory.add_process_narrative(narrative)
        
        logger.info(f"Answer improvement completed in {time.time() - start_time:.2f}s")
        return improved_answer