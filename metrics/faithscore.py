"""
FaithScore metric for fine-grained hallucination evaluation in VLMs.

Implements the three-stage FaithScore pipeline:
    1. Sub-sentence identification (descriptive vs analytical)
    2. Atomic fact extraction from descriptive sub-sentences
    3. Vision-language verification of each atomic fact against the image

Uses LLaMA 2 7B (4-bit quantized) for stages 1-2 and a VLM for stage 3.
Wraps components from the ``faithscore`` pip package where possible.

Reference:
    Jing et al., "FaithScore: Fine-grained Evaluations of Hallucinations
    in Large Vision-Language Models", Proceedings of EMNLP 2024, pp. 5042-5063.
    https://github.com/bcdnlp/FAITHSCORE
"""

import logging
import os
import re
from typing import Any, Dict, List, Optional, Tuple

import nltk
import torch
from tqdm import tqdm

from .base_metric import BaseMetric

logger = logging.getLogger(__name__)

# Ensure NLTK data available
try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt_tab', quiet=True)
try:
    nltk.data.find('taggers/averaged_perceptron_tagger_eng')
except LookupError:
    nltk.download('averaged_perceptron_tagger_eng', quiet=True)


def _load_prompt(filename: str) -> str:
    """Load a prompt template from the faithscore package's prompts directory.

    Args:
        filename: Prompt template filename (e.g. ``prompt_llama.txt``).

    Returns:
        The prompt text.
    """
    import faithscore
    prompts_dir = os.path.join(os.path.dirname(faithscore.__file__), "prompts")
    with open(os.path.join(prompts_dir, filename), "r") as f:
        return f.read()


class FaithScoreMetric(BaseMetric):
    """FaithScore: fine-grained hallucination evaluation for vision-language models.

    This metric decomposes model-generated captions into atomic facts and
    verifies each fact against the source image.  It is **reference-free** --
    images serve as the ground truth rather than object lists.

    The implementation reuses prompt templates and the LLaMA preprocessing
    utilities from the ``faithscore`` pip package (Jing et al., EMNLP 2024).

    Args:
        llama_path: Path to a HuggingFace-format LLaMA model used for
            sub-sentence classification and atomic fact extraction (stages 1-2).
        verification_model: A loaded model instance that exposes a
            ``generate(image, prompt, ...)`` method compatible with
            :class:`models.BaseModel`.  Used for stage-3 fact verification.
            If ``None``, stage 3 is skipped and only atomic-fact counts are
            reported.
    """

    def __init__(
        self,
        llama_path: Optional[str] = None,
        verification_model: Optional[Any] = None,
        caption_model_name: Optional[str] = None,
    ):
        super().__init__("faithscore")

        self.llama_path = llama_path
        # FIXME: Ideally the verification model (stage 3) should be a different
        # VLM from the one used to generate the captions being evaluated.
        self.verification_model = verification_model
        self.caption_model_name = caption_model_name

        # Lazy-loaded LLaMA pipeline
        self._llama_pipeline = None
        self._llama_tokenizer = None

        # Accumulators
        self.total_captions: int = 0
        self.caption_details: List[Dict[str, Any]] = []
        self.faithscore_scores: List[float] = []
        self.sentence_scores: List[float] = []

    # ------------------------------------------------------------------
    # LLaMA loading (stages 1-2)
    # ------------------------------------------------------------------

    def _ensure_llama(self) -> None:
        """Lazy-load the LLaMA pipeline for stages 1-2."""
        if self._llama_pipeline is not None:
            return
        if self.llama_path is None:
            raise ValueError(
                "llama_path is required for FaithScore stages 1-2. "
                "Provide a path to a HuggingFace-format LLaMA model."
            )
        from faithscore.llama_pre import load_llama
        logger.info("Loading LLaMA model from %s ...", self.llama_path)
        self._llama_pipeline, self._llama_tokenizer = load_llama(self.llama_path)
        logger.info("LLaMA model loaded successfully")

    # ------------------------------------------------------------------
    # Stage 1: Sub-sentence identification
    # ------------------------------------------------------------------

    def _stage1(self, answers: List[str]) -> List[str]:
        """Classify sub-sentences as descriptive [D] or analytical [A].

        Re-implements ``faithscore.llama_pre.stage1_llama`` with
        ``max_new_tokens=10`` instead of the package's ``max_length=1000``.
        The package generates up to 1000 tokens per call but only reads the
        first 15 characters to look for "analytical" — i.e. ~50x wasted
        compute. We need only a handful of tokens to make the D/A call.

        Args:
            answers: List of generated captions.

        Returns:
            List of labeled strings with [D]/[A] tags.
        """
        self._ensure_llama()
        import faithscore
        prompts_dir = os.path.join(os.path.dirname(faithscore.__file__), "prompts")
        with open(os.path.join(prompts_dir, "prompt_llama.txt"), "r") as f:
            prompt_label_des_ana = f.read() + "\n\n"

        labeled: List[str] = []
        for answer in tqdm(answers, desc="FaithScore Stage 1"):
            answer = answer.replace("\n", " ")
            result = ""
            for subs in re.split(r"[,.]", answer):
                if len(subs.replace(" ", "")) == 0:
                    continue
                pts = prompt_label_des_ana + subs.replace("\n", " ") + "\n" + " This sub-sentence is "
                with torch.no_grad():
                    sequences = self._llama_pipeline(
                        pts,
                        do_sample=True,
                        top_k=10,
                        num_return_sequences=1,
                        eos_token_id=self._llama_tokenizer.eos_token_id,
                        max_new_tokens=10,
                    )
                    response = sequences[0]["generated_text"][len(pts):len(pts) + 15]
                if "analytical" in response.lower():
                    result += subs + " [A] "
                else:
                    result += subs + " [D] "
            labeled.append(result)
        return labeled

    # ------------------------------------------------------------------
    # Stage 2: Atomic fact extraction
    # ------------------------------------------------------------------

    def _extract_descriptive_text(self, labeled: List[str]) -> List[str]:
        """Extract descriptive portions from [D]/[A]-labeled text.

        Args:
            labeled: Stage-1 output with [D]/[A] tags.

        Returns:
            Concatenated descriptive text per caption.
        """
        all_texts: List[str] = []
        for ss in labeled:
            desc = ""
            pos_des = [m.start() for m in re.finditer(r"\[D\]", ss)]
            pos_ana = [m.start() for m in re.finditer(r"\[A\]", ss)]
            pos_seg = sorted(pos_des + pos_ana)
            for i, pos in enumerate(pos_seg):
                if pos in pos_des:
                    if i == 0:
                        desc += ss[:pos - 1]
                    else:
                        desc += ss[pos_seg[i - 1] + 3:pos - 1]
            all_texts.append(desc.replace("\n", " "))
        return all_texts

    def _extract_labeled_subs(self, labeled: List[str]) -> List[List[str]]:
        """Extract list of descriptive sub-sentences (for sentence-level scoring).

        Args:
            labeled: Stage-1 output with [D]/[A] tags.

        Returns:
            List of lists of descriptive sub-sentence strings.
        """
        all_texts: List[List[str]] = []
        for ss in labeled:
            desc: List[str] = []
            pos_des = [m.start() for m in re.finditer(r"\[D\]", ss)]
            pos_ana = [m.start() for m in re.finditer(r"\[A\]", ss)]
            pos_seg = sorted(pos_des + pos_ana)
            for i, pos in enumerate(pos_seg):
                if pos in pos_des:
                    if i == 0:
                        desc.append(ss[:pos - 1])
                    else:
                        desc.append(ss[pos_seg[i - 1] + 3:pos - 1])
            all_texts.append(desc)
        return all_texts

    def _stage2(self, labeled: List[str]) -> Tuple[
        List[List[str]],
        List[List[str]],
        List[List[str]],
        List[List[str]],
        List[List[str]],
        List[List[str]],
    ]:
        """Extract atomic facts from descriptive sub-sentences.

        Uses the LLaMA pipeline to decompose descriptive text into categories:
        Entities, Relations, Colors, Counting, Other attributes.

        Args:
            labeled: Stage-1 output with [D]/[A] tags.

        Returns:
            Tuple of (all_facts, Entities, Relations, Colors, Counting, Others)
            where each is a list (per caption) of lists of fact strings.
        """
        self._ensure_llama()
        desc_texts = self._extract_descriptive_text(labeled)

        prompt_template = _load_prompt("prompt_de_atomic.txt")
        nons = "Entities:\nRelations:\nColors:\nCounting:\nOther attributes:"

        Entities: List[List[str]] = []
        Relations: List[List[str]] = []
        Colors: List[List[str]] = []
        Counting: List[List[str]] = []
        Others: List[List[str]] = []

        results: List[str] = []
        for ans in tqdm(desc_texts, desc="FaithScore Stage 2"):
            ans = ans.replace("\n", "")
            if ans.strip() == "":
                results.append(nons)
                continue

            pts = prompt_template + "\nAnswer: " + ans
            with torch.no_grad():
                sequences = self._llama_pipeline(
                    pts,
                    do_sample=True,
                    top_k=10,
                    num_return_sequences=1,
                    eos_token_id=self._llama_tokenizer.eos_token_id,
                    max_new_tokens=512,
                )
                response = sequences[0]['generated_text'][len(pts):]
                # Take first paragraph
                response = response.split("\n\n")[0] if "\n\n" in response else response

            if "Entities" in response:
                results.append(response)
            else:
                results.append(nons)

        for facts in results:
            ent, rel, col, cnt, oth = [], [], [], [], []
            for line in facts.split("\n"):
                line = line.strip()
                if line.startswith("Entities:"):
                    val = line.replace("Entities: ", "").replace("Entities:", "")
                    ent = [s for s in val.split(". ") if s.strip()] if val.strip() else []
                elif line.startswith("Relations:"):
                    val = line.replace("Relations: ", "").replace("Relations:", "")
                    rel = [s for s in val.split(". ") if s.strip()] if val.strip() else []
                elif line.startswith("Colors:"):
                    val = line.replace("Colors: ", "").replace("Colors:", "")
                    col = [s for s in val.split(". ") if s.strip()] if val.strip() else []
                elif line.startswith("Counting:"):
                    val = line.replace("Counting: ", "").replace("Counting:", "")
                    cnt = [s for s in val.split(". ") if s.strip()] if val.strip() else []
                elif line.startswith("Other attributes:"):
                    val = line.replace("Other attributes: ", "").replace("Other attributes:", "")
                    oth = [s for s in val.split(". ") if s.strip()] if val.strip() else []
            Entities.append(ent)
            Relations.append(rel)
            Colors.append(col)
            Counting.append(cnt)
            Others.append(oth)

        all_facts = [
            Entities[i] + Relations[i] + Colors[i] + Counting[i] + Others[i]
            for i in range(len(Entities))
        ]
        return all_facts, Entities, Relations, Colors, Counting, Others

    # ------------------------------------------------------------------
    # Stage 3: Vision-language verification
    # ------------------------------------------------------------------

    def _stage3(
        self,
        atomic_facts: List[List[str]],
        image_paths: List[str],
    ) -> Tuple[float, List[List[int]]]:
        """Verify atomic facts against images using the verification model.

        Args:
            atomic_facts: Per-caption lists of atomic fact strings.
            image_paths: Corresponding image file paths.

        Returns:
            Tuple of (overall_score, fact_scores) where fact_scores is a
            list of per-caption lists of 0/1 verification results.
        """
        if self.verification_model is None:
            raise ValueError(
                "verification_model is required for FaithScore stage 3. "
                "Pass a loaded model with a generate() method."
            )

        from PIL import Image

        fact_scores: List[List[int]] = []
        for idx, elements in enumerate(tqdm(atomic_facts, desc="FaithScore Stage 3")):
            scores: List[int] = []
            image = Image.open(image_paths[idx])
            for element in elements:
                prompt = (
                    f"Statement: {element} "
                    "Is this statement is right according to the image? "
                    "Please answer yes or no."
                )
                output = self.verification_model.generate(
                    image=image,
                    prompt=prompt,
                    max_new_tokens=16,
                    do_sample=False,
                )
                scores.append(1 if "yes" in output.lower() else 0)
            fact_scores.append(scores)

        instance_scores = [
            sum(s) / len(s) if len(s) > 0 else 0.0
            for s in fact_scores
        ]
        overall = sum(instance_scores) / len(instance_scores) if instance_scores else 0.0
        return overall, fact_scores

    # ------------------------------------------------------------------
    # Sentence-level scoring (follows original FaithScore algorithm)
    # ------------------------------------------------------------------

    def _sentence_faithscore(
        self,
        Entities: List[List[str]],
        Relations: List[List[str]],
        Colors: List[List[str]],
        Counting: List[List[str]],
        Others: List[List[str]],
        labeled_subs: List[List[str]],
        fact_scores: List[List[int]],
    ) -> float:
        """Compute sentence-level FaithScore.

        Cross-references entity recognition with per-fact verification scores
        to produce sentence-level faithfulness assessments.

        Args:
            Entities: Per-caption entity fact lists.
            Relations: Per-caption relation fact lists.
            Colors: Per-caption color fact lists.
            Counting: Per-caption counting fact lists.
            Others: Per-caption other attribute fact lists.
            labeled_subs: Per-caption descriptive sub-sentence lists.
            fact_scores: Per-caption per-fact 0/1 verification results.

        Returns:
            Mean sentence-level faithfulness score.
        """
        # Recognize key entity words via POS tagging
        entities_recog: List[List[str]] = []
        for ents in Entities:
            entities: List[str] = []
            for ent in ents:
                sentences = nltk.sent_tokenize(ent)
                if not sentences:
                    entities.append("")
                    continue
                tags = nltk.pos_tag(nltk.word_tokenize(sentences[0]))
                ent_words = [
                    tag[0] for tag in tags
                    if tag[1] in ['NN', 'NNS', 'JJ', 'NNP', 'VBG', 'JJR', 'NNPS', 'RB', 'DT']
                ]
                entities.append(ent_words[-1] if ent_words else "")
            entities_recog.append(entities)

        # Split fact_scores by category
        entity_scores: List[List[int]] = []
        relation_scores: List[List[int]] = []
        color_scores: List[List[int]] = []
        count_scores: List[List[int]] = []
        other_scores: List[List[int]] = []

        for i in range(len(fact_scores)):
            ne = len(Entities[i])
            nr = len(Relations[i])
            nc = len(Colors[i])
            nk = len(Counting[i])
            entity_scores.append(fact_scores[i][:ne])
            relation_scores.append(fact_scores[i][ne:ne + nr])
            color_scores.append(fact_scores[i][ne + nr:ne + nr + nc])
            count_scores.append(fact_scores[i][ne + nr + nc:ne + nr + nc + nk])
            other_scores.append(fact_scores[i][ne + nr + nc + nk:])

        sentence_scores: List[List[bool]] = []
        for id1 in range(len(labeled_subs)):
            sent_score: List[bool] = []
            for sub_sen in labeled_subs[id1]:
                flag = True
                for id3, ee in enumerate(entities_recog[id1]):
                    if ee and ee in sub_sen:
                        if id3 < len(entity_scores[id1]) and entity_scores[id1][id3] != 1:
                            flag = False
                        for id4, rel in enumerate(relation_scores[id1]):
                            if id4 < len(Relations[id1]) and ee in Relations[id1][id4] and rel != 1:
                                flag = False
                        for id4, rel in enumerate(color_scores[id1]):
                            if id4 < len(Colors[id1]) and ee in Colors[id1][id4] and rel != 1:
                                flag = False
                        for id4, rel in enumerate(count_scores[id1]):
                            if id4 < len(Counting[id1]) and ee in Counting[id1][id4] and rel != 1:
                                flag = False
                        for id4, rel in enumerate(other_scores[id1]):
                            if id4 < len(Others[id1]) and ee in Others[id1][id4] and rel != 1:
                                flag = False
                sent_score.append(flag)
            sentence_scores.append(sent_score)

        score_per_sent = [
            sum(ss) / len(ss) if len(ss) > 0 else 1.0
            for ss in sentence_scores
        ]
        return sum(score_per_sent) / len(score_per_sent) if score_per_sent else 0.0

    # ------------------------------------------------------------------
    # Public API (BaseMetric interface)
    # ------------------------------------------------------------------

    def evaluate(
        self,
        model_outputs: List[str],
        image_paths: List[str],
        **kwargs: Any,
    ) -> Dict[str, float]:
        """Evaluate captions against source images using FaithScore.

        Args:
            model_outputs: List of generated captions.
            image_paths: List of corresponding image file paths.  FaithScore
                is reference-free so images serve as the ground truth.
            **kwargs: Additional arguments (unused).

        Returns:
            Dictionary with ``faithscore``, ``sentence_faithscore``,
            ``num_captions``, and ``total_atomic_facts``.
        """
        if len(model_outputs) != len(image_paths):
            raise ValueError(
                f"Length mismatch: {len(model_outputs)} captions vs "
                f"{len(image_paths)} image paths"
            )

        verifier_name = getattr(self.verification_model, 'model_name', 'unknown')
        llm_name = self.llama_path or "unknown"

        logger.info("FaithScore pipeline model summary:")
        logger.info("  Stage 1-2 (sub-sentence ID + atomic fact extraction): %s", llm_name)
        logger.info("  Stage 3   (fact verification VLM):                    %s", verifier_name)
        logger.info("  Caption source model:                                  %s", self.caption_model_name)

        # Stage 1
        logger.info("FaithScore Stage 1: Sub-sentence identification (%s) ...", llm_name)
        labeled = self._stage1(model_outputs)

        # Stage 2
        logger.info("FaithScore Stage 2: Atomic fact extraction (%s) ...", llm_name)
        atomic_facts, Entities, Relations, Colors, Counting, Others = self._stage2(labeled)

        total_facts = sum(len(f) for f in atomic_facts)
        logger.info("Extracted %d total atomic facts from %d captions", total_facts, len(model_outputs))

        # Stage 3
        logger.info("FaithScore Stage 3: Vision-language verification (%s) ...", verifier_name)
        overall_score, fact_scores = self._stage3(atomic_facts, image_paths)

        # Sentence-level scoring
        labeled_subs = self._extract_labeled_subs(labeled)
        sentence_score = self._sentence_faithscore(
            Entities, Relations, Colors, Counting, Others,
            labeled_subs, fact_scores,
        )

        verified_facts = sum(sum(fs) for fs in fact_scores)

        # Store per-caption details
        for i in range(len(model_outputs)):
            instance_score = (
                sum(fact_scores[i]) / len(fact_scores[i])
                if len(fact_scores[i]) > 0 else 0.0
            )
            self.caption_details.append({
                'caption': model_outputs[i],
                'image_path': image_paths[i],
                'atomic_facts': atomic_facts[i],
                'fact_scores': fact_scores[i],
                'num_facts': len(atomic_facts[i]),
                'num_verified': sum(fact_scores[i]),
                'faithscore': instance_score,
            })
            self.faithscore_scores.append(instance_score)

        self.total_captions += len(model_outputs)
        self.sentence_scores.append(sentence_score)

        self.results = {
            'faithscore': overall_score,
            'sentence_faithscore': sentence_score,
            'num_captions': self.total_captions,
            'total_atomic_facts': total_facts,
            'verified_atomic_facts': verified_facts,
        }

        return self.results

    def format_results(self) -> Dict[str, Any]:
        """Format evaluation results for reporting.

        Returns:
            Dictionary with ``faithscore_results``, ``caption_details``,
            and ``summary``.
        """
        return {
            'faithscore_results': self.results,
            'caption_details': self.caption_details,
            'summary': {
                'faithscore': self.results.get('faithscore', 0.0),
                'sentence_faithscore': self.results.get('sentence_faithscore', 0.0),
                'total_captions': self.total_captions,
                'total_atomic_facts': self.results.get('total_atomic_facts', 0),
                'verified_atomic_facts': self.results.get('verified_atomic_facts', 0),
            },
        }

    def reset(self) -> None:
        """Reset metric state for a new evaluation."""
        super().reset()
        self.total_captions = 0
        self.caption_details = []
        self.faithscore_scores = []
        self.sentence_scores = []
