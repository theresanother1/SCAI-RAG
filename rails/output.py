from sentence_transformers import SentenceTransformer
from typing import Dict
from helper import EMAIL_PATTERN, get_similarity_model, AUTO_ANSWERS, get_toxicity_model, get_hallucination_model, check_toxicity
from transformers import pipeline, AutoTokenizer
from sklearn.metrics.pairwise import cosine_similarity
from typing import Dict, List
from dataclasses import dataclass
from enum import Enum
import re
from guards.svnr import is_valid_svnr

@dataclass
class GuardrailResult:
    """Guardrail Result Dataclass"""
    passed: bool
    confidence: float
    issues: List[str]
    metrics: Dict[str, float]

class GuardrailType(Enum):
    RELEVANCE = "relevance"
    HALLUCINATION = "hallucination"
    CORRECTNESS = "correctness"
    TOXICITY = "toxicity"

class OutputGuardrails:
    """Guardrails for LLM output validation"""

    def redact_svnrs(self, text: str) -> str:
        """
        Finds and redacts valid Austrian social security numbers (SVNRs) in the text.
        """
        # Find all 10-digit numbers that could be SVNRs
        potential_svnrs = re.findall(r'\b\d{10}\b', text)
        
        # Validate each potential SVNR and redact if valid
        for svnr in potential_svnrs:
            if is_valid_svnr(svnr):
                text = text.replace(svnr, "[REDACTED SVNR]")
        
        return text
    
    def __init__(self):

        self.hallucination_patterns = [
            re.compile(r'\b(definitely|certainly|100%|absolutely)\b', re.IGNORECASE),         # Overstated certainty
            re.compile(r'\b(all|never|always|no one|everyone)\b', re.IGNORECASE),             # Absolute statements
            re.compile(r'\b(studies show|experts say|research proves)\b', re.IGNORECASE),     # Unsupported authority claims
        ]

        self.fact_patterns = [
            re.compile(r'\b\d{4}\b'),  # years
            re.compile(r'\b\d+%\b'),   # %
            re.compile(r'\b\d+\s*(Euro|Dollar|meter|kilometer|km|kg|tons)\b', re.IGNORECASE),  
        ]
        
        try:
            self.sentence_model = get_similarity_model()
        except Exception as e:
            print(f"Could not retrieve sentence model for similarity checks: {e}.")
            
        try:
            self.hallucination_model = get_hallucination_model()
        except Exception as e:
            print(f"Could not retrieve hallucination check pipeline: {e}.")

        try:
            self.toxicity_classifier = get_toxicity_model()
        except Exception as e:
            print(f"Could not toxicity check pipeline: {e}.")
       


    def check_query_relevance(self, 
                             query: str, 
                             response: str, 
                             context: List[str]) -> GuardrailResult:
        """Checks how relevant the answer is to the query. 
        
            Args: 
                query (str): The input query 
                response (str): The generated response by the model
                context List(str): The retrieved context from DB
                
        """
        
        try:
            # query relevance
            query_embedding = self.sentence_model.encode([query])
            response_embedding = self.sentence_model.encode([response])
            
            relevance_score = cosine_similarity(query_embedding, response_embedding)[0][0]
            
            # context relevance
            context_text = " ".join(context)
            context_embedding = self.sentence_model.encode([context_text])
            context_relevance = cosine_similarity(response_embedding, context_embedding)[0][0]
            
            # combined relevance (could change weighting if necessary)
            combined_relevance = (relevance_score * 0.5) + (context_relevance * 0.5)
            
            issues = []
            if relevance_score < 0.3:
                issues.append(AUTO_ANSWERS.ANSWER_NOT_RELEVANT_TO_QUERY.value)
            if context_relevance < 0.4:
                issues.append(AUTO_ANSWERS.ANSWER_NOT_RELEVANT_TO_CONTEXT.value)
            
            passed = combined_relevance >= 0.5 and len(issues) == 0
            
            return GuardrailResult(
                passed=passed,
                confidence=combined_relevance,
                issues=issues,
                metrics={
                    "query_relevance": relevance_score,
                    "context_relevance": context_relevance,
                    "combined_relevance": combined_relevance
                }
            )
            
        except Exception as e:
            print(f"{AUTO_ANSWERS.RELEVANCE_CHECK_FAILED.value}: {e}")
            return GuardrailResult(False, 0.0, [AUTO_ANSWERS.RELEVANCE_CHECK_FAILED.value], {})

    def _check_contradictions(self, response: str, context: str) -> float:
        """
        Check for contradictions between response and context
            Args: 
                response (str): The generated model response
                context (str): DB context combined to string
        """
        
        response_sentences = response.split('.')
        context_sentences = context.split('.')
        
        contradiction_score = 0.0
        
        for resp_sent in response_sentences:
            resp_sent = resp_sent.strip().lower()
            if not resp_sent:
                continue
                
            # Simple negated check (could be extended )
            not_str = ' not '
            negated_resp = resp_sent.replace(not_str, ' ').replace(' no ', ' ')
            
            for ctx_sent in context_sentences:
                ctx_sent = ctx_sent.strip().lower()
                if not ctx_sent:
                    continue
                
                # check if response contradicts context and other way round
                if len(set(negated_resp.split()) & set(ctx_sent.split())) > 2:
                    if (not_str in resp_sent) != (not_str in ctx_sent):
                        contradiction_score += 0.3
                        break
        
        return min(contradiction_score, 1.0)

    def check_hallucination(self, 
                           response: str, 
                           context: List[str]) -> GuardrailResult:
        """Filter out hallucinations. 
        Args:
            response (str): Generated model response 
            context List(str): Retrieved context from DB 
        """
        
        try:
            context_text = " ".join(context)
            issues = []
            
            # Check if model facts are consistent with context
            if self.hallucination_model:
                
                prompt = f"<pad> Determine if the hypothesis is true given the premise?\n\nPremise: {context}\n\nHypothesis: {response}"

                classifier = pipeline(
                            "text-classification",
                            model='vectara/hallucination_evaluation_model',
                            tokenizer=AutoTokenizer.from_pretrained('google/flan-t5-base'),
                            trust_remote_code=True
                        )
                full_scores = classifier(prompt, top_k=None) # List[List[Dict[str, float]]]
                hallucinated_entry = next((item for item in full_scores if item['label'] == 'hallucinated'), None)
                hallucination_score = hallucinated_entry['score']
                
            else:
                # Do semantic similarity check if hallucination model pipeline is unavail
                context_embedding = self.sentence_model.encode([context_text])
                response_embedding = self.sentence_model.encode([response])
                similarity = cosine_similarity(context_embedding, response_embedding)[0][0]
                hallucination_score = 1.0 - similarity  
            
            # check for known hallucination patterns                  
            pattern_score = 0
            for pattern in self.hallucination_patterns:
                if pattern.search(response):
                    pattern_score += 0.2
                    issues.append(f"Matched potential hallucination: {pattern.pattern}.")
            
            # Check if contradictions to context exitst
            contradiction_score = self._check_contradictions(response, context_text)
            
            # Gesamtscore
            total_hallucination_score = min(
                hallucination_score*0.4 + pattern_score*0.3 + contradiction_score*0.3, 1.0
            )
            
            if total_hallucination_score > 0.7:
                issues.append("High hallucination probability.")
            
            passed = total_hallucination_score < 0.5
            
            return GuardrailResult(
                passed=passed,
                confidence=1.0 - total_hallucination_score,
                issues=issues,
                metrics={
                    "hallucination_score": total_hallucination_score,
                    "pattern_score": pattern_score,
                    "contradiction_score": contradiction_score
                }
            )

        except Exception as e:
            print(f"{AUTO_ANSWERS.HALLUCINATION_CHECK_FAILED.value}: {e}")
            return GuardrailResult(False, 0.0, [AUTO_ANSWERS.HALLUCINATION_CHECK_FAILED.value], {})
            

    def _extract_facts(self, text: str) -> List[str]:
        """
        Extract checkable facts from text based on sentence structure
            Args: 
                text (str): text to factcheck
        """
        
        sentences = text.split('.')
        facts = []
    
        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) > 10:  # ignore short sentences
                for pattern in self.fact_patterns:
                    if pattern.search(sentence):
                        facts.append(sentence)
                        break
        
        return facts
    
    def _is_fact_supported(self, fact: str, context: str) -> bool:
        """Check is fact is supported by context: 

            Args: 
                fact (str): extracted fact
                context (str): context joined to str

        """
        fact_embedding = self.sentence_model.encode([fact])
        
        # split context into sentences, find best match
        context_sentences = [s.strip() for s in context.split('.') if len(s.strip()) > 10]
        
        if not context_sentences:
            return False
        
        context_embeddings = self.sentence_model.encode(context_sentences)
        similarities = cosine_similarity(fact_embedding, context_embeddings)[0]
        
        return max(similarities) > 0.7  


    def _check_consistency(self, response: str, context: str) -> float:
        """Check consistency between response and context

            Args: 
                response (str): generated model response 
                context (str): context combined to string
        
        """
        response_embedding = self.sentence_model.encode([response])
        context_embedding = self.sentence_model.encode([context])
        
        consistency = cosine_similarity(response_embedding, context_embedding)[0][0]
        return max(0.0, consistency)
    
    def _check_completeness(self, response: str, context: str) -> float:
        """Check if answer is complete based on answer length and context coverage: 
        
            Args: 
                resposne (str): generated model output
                context (str): context combined to str
        """

        response_words = set(response.lower().split())
        context_words = set(context.lower().split())
        
        if len(context_words) == 0:
            return 1.0
        
        optimal_length =  20 # might want to adapt optimal word length, just a guess

        coverage = len(response_words & context_words) / min(len(context_words), 100)
        length_factor = min(len(response.split()) / optimal_length, 1.0)  
        
        return (coverage * 0.7) + (length_factor * 0.3)
    

    def check_correctness(self, 
                         response: str, 
                         context: List[str]) -> GuardrailResult:
        """Check correctness based on given context. 
           Args:
            response (str): Generated model response 
            context List(str): Retrieved context from DB 
        """
        
        try:
            context_text = " ".join(context)
            issues = []
            
            # context supports facts
            response_facts = self._extract_facts(response)
            supported_facts = 0
            total_facts = len(response_facts)
            
            if total_facts > 0:
                for fact in response_facts:
                    if self._is_fact_supported(fact, context_text):
                        supported_facts += 1
                    else:
                        issues.append(f"Unsupported fact: {fact}")
                
                support_ratio = supported_facts / total_facts
            else:
                support_ratio = 1.0 # no checkable facts 
            
     
            consistency_score = self._check_consistency(response, context_text)
            completeness_score = self._check_completeness(response, context_text)
            correctness_score = (
                support_ratio * 0.5 + 
                consistency_score * 0.3 + 
                completeness_score * 0.2
            )
            
            if support_ratio < 0.7:
                issues.append(AUTO_ANSWERS.NOT_SUPPORTED_BY_CONTEXT.value)
            if consistency_score < 0.4:
                issues.append(AUTO_ANSWERS.INCONSISTENT_WITH_CONTEXT.value)
            
            passed = correctness_score >= 0.5 and len(issues) <= 1
            
            return GuardrailResult(
                passed=passed,
                confidence=correctness_score,
                issues=issues,
                metrics={
                    "support_ratio": support_ratio,
                    "consistency_score": consistency_score,
                    "completeness_score": completeness_score,
                    "correctness_score": correctness_score
                }
            )
            
        except Exception as e:
            print(f"{AUTO_ANSWERS.CORRECTNESS_CHECK_FAILED.value} {e}")
            return GuardrailResult(False, 0.0, [AUTO_ANSWERS.CORRECTNESS_CHECK_FAILED.value], {})
    
    
    def check_toxicity(self, response: str): 
        """ Toxicity check with toxicity check pipeline
            Args: 
                response (str): response generated by model
            Returns: 

        """
        print("CHECKING OUTPUT LANGUAGE: ")
        toxicity_passed, toxicity_score, text = check_toxicity(response)

        return GuardrailResult(
            passed=toxicity_passed,
            confidence=1 - toxicity_score,
            issues=[text] if text else [],
            metrics={"toxicity_passed": toxicity_passed}
        )
        

    def check(self, 
        query: str, 
        response: str, 
        context: List[str]) -> Dict[GuardrailType, GuardrailResult]:
        """Execute Guard rail checks.
        
            Args: 
                query (str): User query
                response (str): Response generated by model
                context (str): given context 
        """
        results = {}

        if response.find("I can only help with university academic topics") != -1 or response.find("I don't have access to that information. Please contact ") != -1: 
            tasks = [
                self.check_hallucination(response, context), 
                self.check_toxicity(response)
            ] 
            hallucination_result, toxicity_result = tasks
            results[GuardrailType.HALLUCINATION] = hallucination_result
            results[GuardrailType.TOXICITY] = toxicity_result

        else: 
            tasks = [
                self.check_query_relevance(query, response, context),
                self.check_hallucination(response, context),
                self.check_correctness(response, context),
                self.check_toxicity(response)
            ] 
            relevance_result, hallucination_result, correctness_result, toxicity_result = tasks
            results[GuardrailType.RELEVANCE] = relevance_result
            results[GuardrailType.HALLUCINATION] = hallucination_result
            results[GuardrailType.CORRECTNESS] = correctness_result
            results[GuardrailType.TOXICITY] = toxicity_result

        print("Result for query: ", query)
        print("WITH CONTEXT: ", context)
        print("Model output: ", response)
        for guardrail_type, result in results.items():
            print(f"\n Guardrail: {guardrail_type.value.upper()}")
            print(f"   Passed:     {result.passed}")
            print(f"   Confidence: {result.confidence:.3f}")
            
            if result.issues:
                print(f"   Issues:     {', '.join(result.issues)}")
            else:
                print(f"   Issues:     None")
            
            print("   Metrics:")
            for metric, value in result.metrics.items():
                print(f"     - {metric}: {value:.4f}")
            
        return results
    

    def format_guardrail_issues(self, gr_result: dict, response: str) -> str:
        """Formats the issues text to readable output
        Parametes: 
            - gr_result (dict): The dictionary containing the different results from the output guardrail check
        Returns: 
            Formatted string. 
        """
        count = 0
        for key, gr in gr_result.items(): 
            if not gr.passed and key in [GuardrailType.HALLUCINATION, GuardrailType.TOXICITY]: 
                count += 1
            
        lines = ["ISSUES DETECTED WITH OUTPUT:\n"]

        for guardrail_type, result in gr_result.items():
            issues = result.issues
            title = f"- {guardrail_type.value.capitalize()} issues:"
            print(issues)
            if issues:
                lines.append(title)
                for issue in issues:
                    lines.append(f"  • {issue}")
                if guardrail_type == GuardrailType.TOXICITY and not result.passed: 
                    lines.append(f"  • {AUTO_ANSWERS.REPHRASE_SENTENCE.value}")
            print(lines)
        if count != 0: 
            return "\n".join(lines)
        
        lines.append(f"\nThe respective reponse is: \n\n{response}")
        return "\n".join(lines)
