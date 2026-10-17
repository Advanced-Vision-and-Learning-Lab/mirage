import functools
import json
import logging
import re
from typing import Dict, Iterable, List, Any, Set, Tuple, Optional
from collections import Counter

import nltk
import inflect

from .base_metric import (CAPTION_GEN, MIN_WORDS, SAMPLES, BaseMetric, MissingVerdicts, caption_images,
                          dataset_key, fbeta, ground_truth, verdict_key)

logger = logging.getLogger(__name__)

# The tokenizer data must be installed; it is never downloaded here (LookupError when missing).
nltk.data.find('tokenizers/punkt_tab')

# Inflect engine for singularization
_inflect_engine = inflect.engine()


def _singularize(word: str) -> str:
    """Singularize a word using inflect. Returns original if already singular."""
    singular = _inflect_engine.singular_noun(word)
    if singular is False:
        return word
    return singular


# COCO synonyms from the gold-standard CHAIR implementation
# (https://github.com/Maxlinn/CHAIR-metric-standalone)
# Each line: canonical COCO category -> list of known synonyms
COCO_SYNONYMS_TXT = '''
person, girl, boy, man, woman, kid, child, chef, baker, people, adult, rider, children, baby, worker, passenger, sister, biker, policeman, cop, officer, lady, cowboy, bride, groom, male, female, guy, traveler, mother, father, gentleman, pitcher, player, skier, snowboarder, skater, skateboarder, person, woman, guy, foreigner, child, gentleman, caller, offender, coworker, trespasser, patient, politician, soldier, grandchild, serviceman, walker, drinker, doctor, bicyclist, thief, buyer, teenager, student, camper, driver, solider, hunter, shopper, villager
bicycle, bike, bicycle, bike, unicycle, minibike, trike
car, automobile, van, minivan, sedan, suv, hatchback, cab, jeep, coupe, taxicab, limo, taxi
motorcycle, scooter, motor bike, motor cycle, motorbike, scooter, moped
airplane, jetliner, plane, air plane, monoplane, aircraft, jet, jetliner, airbus, biplane, seaplane
bus, minibus, trolley
train, locomotive, tramway, caboose
truck, pickup, lorry, hauler, firetruck
boat, ship, liner, sailboat, motorboat, dinghy, powerboat, speedboat, canoe, skiff, yacht, kayak, catamaran, pontoon, houseboat, vessel, rowboat, trawler, ferryboat, watercraft, tugboat, schooner, barge, ferry, sailboard, paddleboat, lifeboat, freighter, steamboat, riverboat, battleship, steamship
traffic light, street light, traffic signal, stop light, streetlight, stoplight
fire hydrant, hydrant
stop sign
parking meter
bench, pew
bird, ostrich, owl, seagull, goose, duck, parakeet, falcon, robin, pelican, waterfowl, heron, hummingbird, mallard, finch, pigeon, sparrow, seabird, osprey, blackbird, fowl, shorebird, woodpecker, egret, chickadee, quail, bluebird, kingfisher, buzzard, willet, gull, swan, bluejay, flamingo, cormorant, parrot, loon, gosling, waterbird, pheasant, rooster, sandpiper, crow, raven, turkey, oriole, cowbird, warbler, magpie, peacock, cockatiel, lorikeet, puffin, vulture, condor, macaw, peafowl, cockatoo, songbird
cat, kitten, feline, tabby
dog, puppy, beagle, pup, chihuahua, schnauzer, dachshund, rottweiler, canine, pitbull, collie, pug, terrier, poodle, labrador, doggie, doberman, mutt, doggy, spaniel, bulldog, sheepdog, weimaraner, corgi, cocker, greyhound, retriever, brindle, hound, whippet, husky
horse, colt, pony, racehorse, stallion, equine, mare, foal, palomino, mustang, clydesdale, bronc, bronco
sheep, lamb, ram, lamb, goat, ewe
cow, cattle, oxen, ox, calf, cattle, holstein, heifer, buffalo, bull, zebu, bison
elephant
bear, panda
zebra
giraffe
backpack, knapsack
umbrella
handbag, wallet, purse, briefcase
tie, bow, bow tie
suitcase, suit case, luggage
frisbee
skis, ski
snowboard
sports ball, ball
kite
baseball bat
baseball glove
skateboard
surfboard, longboard, skimboard, shortboard, wakeboard
tennis racket, racket
bottle
wine glass
cup
fork
knife, pocketknife, knive
spoon
bowl, container
banana
apple
sandwich, burger, sub, cheeseburger, hamburger
orange
broccoli
carrot
hot dog
pizza
donut, doughnut, bagel
cake, cheesecake, cupcake, shortcake, coffeecake, pancake
chair, seat, stool
couch, sofa, recliner, futon, loveseat, settee, chesterfield
potted plant, houseplant
bed
dining table, table, desk
toilet, urinal, commode, toilet, lavatory, potty
tv, monitor, televison, television
laptop, computer, notebook, netbook, lenovo, macbook, laptop computer
mouse
remote
keyboard
cell phone, mobile phone, phone, cellphone, telephone, phon, smartphone, iPhone
microwave
oven, stovetop, stove, stove top oven
toaster
sink
refrigerator, fridge, fridge, freezer
book
clock
vase
scissors
teddy bear, teddybear
hair drier, hairdryer
toothbrush
'''


def _build_coco_synonym_structures() -> Tuple[List[str], Dict[str, str], Dict[str, str]]:
    """Parse COCO_SYNONYMS_TXT into lookup structures.

    Returns:
        Tuple of (mscoco_objects, inverse_synonym_dict, double_word_dict)
        - mscoco_objects: flat list of all known synonym words
        - inverse_synonym_dict: maps any synonym -> canonical (first) word
        - double_word_dict: maps compound phrases -> replacement word
    """
    synonyms = [
        [w.strip() for w in line.strip().split(',') if w.strip()]
        for line in COCO_SYNONYMS_TXT.strip().splitlines()
        if line.strip()
    ]

    mscoco_objects: List[str] = []
    inverse_synonym_dict: Dict[str, str] = {}
    for synonym_group in synonyms:
        mscoco_objects.extend(synonym_group)
        canonical = synonym_group[0]
        for s in synonym_group:
            inverse_synonym_dict[s] = canonical

    # Double words: compound COCO phrases that must be detected before splitting
    coco_double_words = [
        'motor bike', 'motor cycle', 'air plane', 'traffic light',
        'street light', 'traffic signal', 'stop light', 'fire hydrant',
        'stop sign', 'parking meter', 'suit case', 'sports ball',
        'baseball bat', 'baseball glove', 'tennis racket', 'wine glass',
        'hot dog', 'cell phone', 'mobile phone', 'teddy bear',
        'hair drier', 'potted plant', 'bow tie', 'laptop computer',
        'stove top oven', 'home plate', 'train track',
    ]

    double_word_dict: Dict[str, str] = {}
    for dw in coco_double_words:
        double_word_dict[dw] = dw

    # Animal qualifiers: "baby bird" -> "bird" (avoids false match on "person")
    animal_words = [
        'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
        'elephant', 'bear', 'zebra', 'giraffe', 'animal', 'cub',
    ]
    vehicle_words = ['jet', 'train']

    for animal in animal_words:
        double_word_dict[f'baby {animal}'] = animal
        double_word_dict[f'adult {animal}'] = animal
    for vehicle in vehicle_words:
        double_word_dict[f'passenger {vehicle}'] = vehicle

    # Specific overrides
    double_word_dict['bow tie'] = 'tie'
    double_word_dict['toilet seat'] = 'toilet'
    double_word_dict['wine glas'] = 'wine glass'

    return mscoco_objects, inverse_synonym_dict, double_word_dict


# Pre-built module-level lookup tables
_COCO_OBJECTS, _COCO_INVERSE_SYNONYMS, _COCO_DOUBLE_WORDS = _build_coco_synonym_structures()

# Words after which "orange" is still the fruit ("an orange on the table",
# "orange and banana"); anything else alphabetic makes it a color ("orange cat").
_ORANGE_FRUIT_FOLLOWERS = {
    'and', 'or', 'in', 'on', 'with', 'of', 'is', 'are', 'was', 'were', 'that',
    'which', 'next', 'to', 'at', 'by', 'the', 'a', 'an', 'for', 'from', 'as',
    'has', 'have', 'had', 'into', 'onto', 'sitting', 'lying', 'placed', 'near',
    'beside', 'under', 'over', 'inside', 'juice', 'fruit', 'slice', 'slices',
    'peel', 'wedge', 'wedges', 'half', 'halves',
}


# Visual Genome names that are not objects (NLTK stopwords are added); without them
# CHAIR counts every caption containing 'the' or 'red' as a mention.
VG_NON_OBJECTS = {
    "red", "white", "black", "blue", "green", "yellow", "brown", "gray",
    "grey", "pink", "purple", "orange", "gold", "silver", "tan", "beige",
    "large", "small", "wooden", "metal", "plastic", "old", "new", "tall",
    "short", "long", "wide", "narrow", "thick", "thin", "open", "closed",
    "cozy", "sunny", "shiny", "rough", "smooth", "clean", "dirty",
    "background", "foreground", "left", "right", "top", "bottom",
    "middle", "center", "side", "corner", "edge",
    "parked", "standing", "sitting", "walking", "lying", "running",
}


def vg_vocabulary(all_objects):
    """CHAIR vocabulary for Visual Genome: object names minus stopwords and VG_NON_OBJECTS."""
    from nltk.corpus import stopwords

    bad = set(stopwords.words("english")) | VG_NON_OBJECTS
    return [name for name in all_objects if name.lower().strip() not in bad and len(name.strip()) >= 2]


def _resolve(item: str, model: str, m: Dict, verdicts: Dict, ref: Iterable[str]) -> str:
    """Mention verdict: annotated, a reference-caption object, not a noun, or a plural/phrase/
    synonym/hypernym of an annotation resolve without the judge."""
    if not m["hall"] or m["word"] in ref:
        return "present"
    if not m["noun"]:
        return "not_object"
    if m["rel"]:
        return "present"
    return verdicts.get((item, f"{model}|{m['word']}"), "missing")


class ChairMetric(BaseMetric):
    """
    Caption Hallucination Assessment with Image Relevance (CHAIR) metric.

    CHAIR evaluates a model's caption generation by measuring hallucination -
    the inclusion of objects that are not actually present in the image.
    It provides both sentence-level (CHAIRs) and instance-level (CHAIRi) metrics,
    as well as Recall of ground-truth objects.

    The metric calculates:
    - CHAIRs: Percentage of sentences with at least one hallucinated object
    - CHAIRi: Percentage of mentioned object instances that are hallucinated
    - Recall: Percentage of ground-truth objects that are mentioned in captions
    """

    def __init__(
        self,
        object_synsets: Optional[Dict[str, List[str]]] = None,
        use_coco_synonyms: bool = True,
        color_guard: bool = False,
    ):
        """
        Initialize the CHAIR metric.

        Args:
            object_synsets: Dictionary mapping canonical object names to lists of
                synonyms. If provided, these are used instead of built-in COCO
                synonyms.  Example:
                {"dog": ["dog", "puppy", "canine"], "car": ["car", "vehicle"]}
            use_coco_synonyms: When True (default) and object_synsets is None,
                use the built-in COCO synonym dictionary from the original CHAIR
                paper. Set to False for exact-match mode with no synonyms.
            color_guard: Skip the token ``orange`` when it is used as a color
                (followed by an ordinary word: "orange cat", "orange walls")
                rather than the COCO fruit class. Off by default, which is the
                canonical CHAIR behaviour and what the published (judged) scores use.
                In the September 2026 COCO runs ~70% of "orange" hallucinations
                per Qwen cell were the color.
        """
        super().__init__("chair")
        self.color_guard = color_guard

        # ----- synonym / object structures -----
        if object_synsets is not None:
            # User-supplied synsets (backward compatible path)
            inverse: Dict[str, str] = {}
            for canonical, syns in object_synsets.items():
                canonical_lower = canonical.lower()
                for s in syns:
                    inverse[s.lower()] = canonical_lower
            self.inverse_synonym_dict = inverse
            self._mscoco_objects_set: Set[str] = set(inverse.keys())
            self.double_word_dict: Dict[str, str] = {}
        elif use_coco_synonyms:
            self.inverse_synonym_dict = _COCO_INVERSE_SYNONYMS
            self._mscoco_objects_set = set(_COCO_OBJECTS)
            self.double_word_dict = _COCO_DOUBLE_WORDS
        else:
            self.inverse_synonym_dict = {}
            self._mscoco_objects_set = set()
            self.double_word_dict = {}

        # ----- accumulators -----
        self.total_sentences = 0
        self.hallucinated_sentences = 0
        self.total_coco_words = 0
        self.hallucinated_word_count = 0
        self.total_gt_objects = 0
        self.recalled_gt_objects = 0

        # Detailed tracking
        self.caption_details: List[Dict[str, Any]] = []
        self.object_hallucination_counts: Counter = Counter()

    @classmethod
    def for_dataset(cls, ds: str, loader) -> "ChairMetric":
        """CHAIR as scored for ``ds``: COCO synonyms, or exact names from the Visual Genome vocabulary."""
        if ds == "vg":
            vocab = vg_vocabulary(loader.get_object_names())
            chair = cls(object_synsets={n: [n] for n in vocab}, use_coco_synonyms=False)
        else:
            chair = cls(use_coco_synonyms=True)
        return chair

    def run(self, loader, model, n: int = SAMPLES, gen=CAPTION_GEN) -> List[Dict]:
        """Caption the first ``n`` annotated images, evaluate them (plain CHAIR, see
        :meth:`format_results`), and return their tagged mentions (see :meth:`images`)."""
        samples, captions = caption_images(loader, model, n, gen)
        self.evaluate(captions, [ground_truth(s) for s in samples])
        return self.images(dataset_key(loader), samples, captions, verdict_key(model))

    def images(self, ds: str, samples: List[Dict], captions: List[str], key: str) -> List[Dict]:
        """The images :meth:`score` takes: ``{"item", "gt", "models": {key: {"caption", "mentions"}}}``
        per sample, where ``key`` names the model in mention verdicts (its registry key)."""
        return [{"item": f"{ds}_{s['image_id']}", "gt": ground_truth(s),
                 "models": {key: {"caption": c, "mentions": self.tag_mentions(c, ground_truth(s))}}}
                for s, c in zip(samples, captions)]

    def reference_objects(self, captions_json, images: List[Dict]) -> Dict[str, set]:
        """COCO objects named in each item's reference captions; the CHAIR paper counts them as present."""
        by_image: Dict[str, List[str]] = {}
        for a in json.load(open(captions_json))["annotations"]:
            by_image.setdefault(f"coco_{a['image_id']}", []).append(a["caption"])
        return {img["item"]: {w for cap in by_image.get(img["item"], []) for w in self.caption_to_words(cap)[1]}
                for img in images}

    @functools.lru_cache(maxsize=1024)
    def normalize_object(self, obj: str) -> str:
        """
        Normalize an object name to its canonical form.

        Args:
            obj: Object name to normalize

        Returns:
            Canonical object name, or the singularized/lowered original if no
            mapping exists.
        """
        obj_lower = obj.lower().strip()
        if obj_lower in self.inverse_synonym_dict:
            return self.inverse_synonym_dict[obj_lower]
        # Try singularized form
        singular = _singularize(obj_lower)
        if singular in self.inverse_synonym_dict:
            return self.inverse_synonym_dict[singular]
        return obj_lower

    def caption_to_words(self, caption: str) -> Tuple[List[str], List[str]]:
        """
        Extract recognized object words from a caption.

        Follows the gold-standard CHAIR pipeline:
        1. NLTK tokenization
        2. Singularize every token
        3. Replace double-word compounds
        4. Filter to known COCO / synset objects
        5. Map each word to its canonical (node) form

        Args:
            caption: Generated caption text

        Returns:
            Tuple of (raw_words, node_words) where:
            - raw_words: all tokens after double-word replacement
            - node_words: canonical forms of recognized object words
              (preserves duplicates)
        """
        # Standard preprocessing: tokenize and singularize
        # Only call _singularize on tokens not already recognized, since
        # singularization is relatively expensive.
        raw_tokens = nltk.word_tokenize(caption.lower())
        if self.color_guard:
            raw_tokens = [
                t for i, t in enumerate(raw_tokens)
                if not (t == 'orange' and i + 1 < len(raw_tokens)
                        and raw_tokens[i + 1].isalpha()
                        and raw_tokens[i + 1] not in _ORANGE_FRUIT_FOLLOWERS)
            ]
        words = []
        for w in raw_tokens:
            if w in self._mscoco_objects_set:
                words.append(w)
            else:
                words.append(_singularize(w))

        # Replace double words
        i = 0
        double_words: List[str] = []
        while i < len(words):
            if i < len(words) - 1:
                pair = f'{words[i]} {words[i + 1]}'
                if pair in self.double_word_dict:
                    double_words.append(self.double_word_dict[pair])
                    i += 2
                    continue
            double_words.append(words[i])
            i += 1
        words = double_words

        # "toilet seat" rule: if both "toilet" and "seat" remain, drop "seat"
        # so it does not false-fire for "chair"
        if 'toilet' in words and 'seat' in words:
            words = [w for w in words if w != 'seat']

        # Filter to known objects & map to canonical forms
        node_words: List[str] = []
        for w in words:
            if w in self._mscoco_objects_set:
                node_words.append(self.inverse_synonym_dict[w])

        return words, node_words

    def check_object_hallucination(
        self,
        caption: str,
        ground_truth_objects: List[str],
    ) -> Dict[str, Any]:
        """
        Check which object mentions in the caption are hallucinated.

        Args:
            caption: Generated caption text
            ground_truth_objects: List of objects actually present in the image

        Returns:
            Dictionary with:
            - raw_words: tokens after double-word replacement
            - coco_words: list of canonical object mentions (with duplicates)
            - hallucinated_words: list of hallucinated canonical words
            - recalled_gt: set of ground-truth objects that were mentioned
            - gt_set: set of normalized ground-truth objects
        """
        raw_words, coco_words = self.caption_to_words(caption)

        # Normalize ground truth to canonical forms
        gt_set = set(self.normalize_object(obj) for obj in ground_truth_objects)

        hallucinated_words: List[str] = []
        recalled_gt: Set[str] = set()
        for word in coco_words:
            if word not in gt_set:
                hallucinated_words.append(word)
            else:
                recalled_gt.add(word)

        return {
            'raw_words': raw_words,
            'coco_words': coco_words,
            'hallucinated_words': hallucinated_words,
            'recalled_gt': recalled_gt,
            'gt_set': gt_set,
        }

    def tag_mentions(self, caption: str, ground_truth_objects: List[str]) -> List[Dict[str, Any]]:
        """Object mentions of ``caption`` with their part of speech in context (judged CHAIR).

        Tokens, singularization and double words follow :meth:`caption_to_words`
        (without the toilet-seat and colour rules). Each mention records
        ``word`` (canonical), ``tok`` (as written), ``pos`` and ``noun`` (NLTK
        tag in context), ``hall`` (not in the annotations) and, for
        hallucinated words, ``rel``/``rel_gt``: how the word relates to an
        annotation (see :func:`variant_of_gt`).
        """
        words = []
        for tok, tag in nltk.pos_tag(nltk.word_tokenize(caption)):
            t = tok.lower()
            words.append((tok, tag, t if t in self._mscoco_objects_set else _sing(t)))
        merged, i = [], 0
        while i < len(words):
            if i + 1 < len(words):
                pair = f"{words[i][2]} {words[i + 1][2]}"
                if pair in self.double_word_dict:
                    merged.append((f"{words[i][0]} {words[i + 1][0]}", words[i + 1][1], self.double_word_dict[pair]))
                    i += 2
                    continue
            merged.append(words[i])
            i += 1
        gt = {self.normalize_object(o) for o in ground_truth_objects}
        mentions = []
        for tok, tag, w in merged:
            if w not in self._mscoco_objects_set:
                continue
            canon = self.inverse_synonym_dict[w]
            hall = canon not in gt
            rel, rel_gt = variant_of_gt(w, ground_truth_objects) if hall else (None, None)
            mentions.append({"tok": tok, "word": canon, "pos": tag, "hall": hall,
                             "noun": tag.startswith("NN"), "rel": rel, "rel_gt": rel_gt})
        return mentions

    def evaluate_caption(
        self,
        caption: str,
        ground_truth_objects: List[str],
    ) -> Dict[str, Any]:
        """
        Evaluate a single caption for hallucinations.

        Args:
            caption: Generated caption text
            ground_truth_objects: List of objects actually present in the image

        Returns:
            Dictionary containing per-caption metrics and details.
        """
        result = self.check_object_hallucination(caption, ground_truth_objects)

        coco_words = result['coco_words']
        hallucinated_words = result['hallucinated_words']
        is_hallucinated = len(hallucinated_words) > 0

        chair_i_caption = (
            len(hallucinated_words) / len(coco_words)
            if len(coco_words) > 0 else 0.0
        )
        recall_caption = (
            len(result['recalled_gt']) / len(result['gt_set'])
            if len(result['gt_set']) > 0 else 0.0
        )

        return {
            # sorted (not list(set(...))) so the emitted JSON is byte-stable:
            # set iteration order is hash-seed dependent across processes.
            'mentioned_objects': sorted(set(coco_words)),
            'hallucinated_objects': sorted(set(hallucinated_words)),
            'coco_words': coco_words,
            'hallucinated_words': hallucinated_words,
            'is_hallucinated': is_hallucinated,
            'chair_i': chair_i_caption,
            'chair_s': int(is_hallucinated),
            'recall': recall_caption,
            'gt_objects': sorted(result['gt_set']),
            'recalled_gt': sorted(result['recalled_gt']),
        }

    def evaluate(
        self,
        model_outputs: List[str],
        ground_truth_objects: List[List[str]],
        **kwargs,
    ) -> Dict[str, float]:
        """
        Evaluate model-generated captions against ground truth objects.

        Computes CHAIRs (sentence-level), CHAIRi (instance-level), and Recall
        following the gold-standard CHAIR methodology.

        Args:
            model_outputs: List of generated captions
            ground_truth_objects: List of object lists per image
            **kwargs: Additional arguments (unused)

        Returns:
            Dictionary of metric scores including chair_s, chair_i, recall,
            and supporting counts.
        """
        if len(model_outputs) != len(ground_truth_objects):
            raise ValueError(
                f"Length mismatch: {len(model_outputs)} captions vs "
                f"{len(ground_truth_objects)} ground truth lists"
            )

        for caption, gt_objects in zip(model_outputs, ground_truth_objects):
            result = self.evaluate_caption(caption, gt_objects)

            # Sentence-level
            self.total_sentences += 1
            if result['is_hallucinated']:
                self.hallucinated_sentences += 1

            # Instance-level (count every word, not unique objects)
            self.total_coco_words += len(result['coco_words'])
            self.hallucinated_word_count += len(result['hallucinated_words'])

            # Recall
            self.total_gt_objects += len(result['gt_objects'])
            self.recalled_gt_objects += len(result['recalled_gt'])

            # Track which objects were hallucinated
            for obj in result['hallucinated_words']:
                self.object_hallucination_counts[obj] += 1

            # Store details for later analysis
            self.caption_details.append({
                'caption': caption,
                'ground_truth_objects': gt_objects,
                'mentioned_objects': result['mentioned_objects'],
                'hallucinated_objects': result['hallucinated_objects'],
                'is_hallucinated': result['is_hallucinated'],
                'chair_s': result['chair_s'],
                'chair_i': result['chair_i'],
                'recall': result['recall'],
            })

        # Aggregate metrics
        chair_s = (
            self.hallucinated_sentences / self.total_sentences
            if self.total_sentences > 0 else 0.0
        )
        chair_i = (
            self.hallucinated_word_count / self.total_coco_words
            if self.total_coco_words > 0 else 0.0
        )
        recall = (
            self.recalled_gt_objects / self.total_gt_objects
            if self.total_gt_objects > 0 else 0.0
        )

        self.results = {
            'chair_s': chair_s,
            'chair_i': chair_i,
            'recall': recall,
            'num_sentences': self.total_sentences,
            'num_hallucinated_sentences': self.hallucinated_sentences,
            'num_object_mentions': self.total_coco_words,
            'num_hallucinated_mentions': self.hallucinated_word_count,
            'num_gt_objects': self.total_gt_objects,
            'num_recalled_gt_objects': self.recalled_gt_objects,
        }

        return self.results

    def format_results(self) -> Dict[str, Any]:
        """
        Format the evaluation results for reporting.

        Returns:
            Dictionary containing chair_results, hallucinated_objects counter,
            caption_details, and a summary.
        """
        return {
            'chair_results': self.results,
            'hallucinated_objects': dict(
                self.object_hallucination_counts.most_common(20)
            ),
            'caption_details': self.caption_details,
            'summary': {
                'chair_s': self.results.get('chair_s', 0.0),
                'chair_i': self.results.get('chair_i', 0.0),
                'recall': self.results.get('recall', 0.0),
                'total_captions': self.total_sentences,
                'captions_with_hallucinations': self.hallucinated_sentences,
                'hallucination_free_captions': (
                    self.total_sentences - self.hallucinated_sentences
                ),
            },
        }

    def reset(self) -> None:
        """Reset the metric state for a new evaluation."""
        super().reset()
        self.total_sentences = 0
        self.hallucinated_sentences = 0
        self.total_coco_words = 0
        self.hallucinated_word_count = 0
        self.total_gt_objects = 0
        self.recalled_gt_objects = 0
        self.caption_details = []
        self.object_hallucination_counts = Counter()
        self.normalize_object.cache_clear()

    @staticmethod
    def score(images: List[Dict], verdicts: Dict, refs: Optional[Dict[str, set]] = None,
              committee: Optional[Dict[str, Dict[str, set]]] = None) -> Dict[str, Dict]:
        """CHAIR-I on raw mentions, and Mod-CHAIR on judged ones: each object counted once per
        caption, captions under ``MIN_WORDS`` words mention nothing, coverage = present objects
        named / all present objects named by any model. ``images``: ``{"item", "models":
        {model: {"caption", "mentions"}}}``; ``refs``: item -> reference-caption objects (COCO);
        ``committee``: item -> {"present": words the frozen pool of models found present}, added
        to the coverage pool of a model scored on its own."""
        refs = refs or {}
        committee = committee or {}
        rows: Dict[str, List[Dict]] = {}
        for img in images:
            item, ref = img["item"], refs.get(img["item"], ())
            short = {m: len(c["caption"].split()) < MIN_WORDS for m, c in img["models"].items()}
            res = {m: [] if short[m] else [(x["word"], _resolve(item, m, x, verdicts, ref)) for x in c["mentions"]]
                   for m, c in img["models"].items()}
            missing = [(item, f"{m}|{w}") for m, r in res.items() for w, v in r if v == "missing"]
            if missing:
                raise MissingVerdicts("mention", missing)
            present = {m: {w for w, v in r if v == "present"} for m, r in res.items()}
            pool = set().union(*present.values(), committee.get(item, {}).get("present", ()))
            for m, c in img["models"].items():
                uniq = {}
                for w, v in res[m]:
                    uniq.setdefault(w, v)
                obj = [v for v in uniq.values() if v in ("present", "absent")]
                rows.setdefault(m, []).append({
                    "raw_m": len(c["mentions"]), "raw_h": sum(x["hall"] for x in c["mentions"]),
                    "m": len(obj), "h": obj.count("absent"), "found": len(present[m]), "pool": len(pool)})

        def ratio(rs, num, den):
            d = sum(r[den] for r in rs)
            return sum(r[num] for r in rs) / d if d else 0.0

        out = {}
        for m, rs in rows.items():
            chair_i, coverage = ratio(rs, "h", "m"), ratio(rs, "found", "pool")
            out[m] = {"raw": ratio(rs, "raw_h", "raw_m"), "head": fbeta(1 - chair_i, coverage), "mod_chair_i": chair_i,
                      "coverage": coverage, "mod_chair_s": sum(r["h"] > 0 for r in rs) / len(rs),
                      "n_raw": sum(r["raw_m"] for r in rs), "n_head": sum(r["m"] for r in rs)}
        return out

    @staticmethod
    def verified_present(images: List[Dict], verdicts: Dict) -> List[set]:
        """Per image: words any model mentioned that are annotated, related to an annotation, or judged present."""
        out = []
        for img in images:
            words = set()
            for m, c in img["models"].items():
                for x in c["mentions"]:
                    if not x["hall"] or (x["noun"] and x["rel"]) or (
                            x["noun"] and verdicts.get((img["item"], f"{m}|{x['word']}")) == "present"):
                        words.add(x["word"])
            out.append(words)
        return out


# ----------------------------------------------------------------------
# Relation of a mentioned word to the annotations (judged CHAIR and THRONE)
# ----------------------------------------------------------------------

# Hypernyms too broad to tie two names together ("object", "artifact", ...).
_GENERIC_SYNSETS = {
    "entity.n.01", "physical_entity.n.01", "object.n.01", "whole.n.02", "artifact.n.01",
    "abstraction.n.06", "matter.n.03", "causal_agent.n.01", "living_thing.n.01", "organism.n.01",
    "instrumentality.n.03", "structure.n.01", "device.n.01", "container.n.01", "covering.n.02",
    "commodity.n.01", "location.n.01", "region.n.01", "part.n.01", "thing.n.12", "group.n.01",
    "unit.n.03", "material.n.01", "substance.n.01", "natural_object.n.01", "plant_part.n.01",
    "body_part.n.01", "attribute.n.02", "measure.n.02", "psychological_feature.n.01", "event.n.01",
    "act.n.02", "communication.n.02", "relation.n.01", "point.n.02", "area.n.01", "surface.n.01",
    "solid.n.01", "fluid.n.01", "food.n.01", "food.n.02", "vertebrate.n.01", "animal.n.01",
    "plant.n.02", "vascular_plant.n.01", "conveyance.n.03", "equipment.n.01"}


@functools.lru_cache(maxsize=None)
def _sing(word: str) -> str:
    return _singularize(word.lower().strip())


@functools.lru_cache(maxsize=None)
def _head(name: str) -> str:
    toks = re.findall(r"[a-z]+", name.lower())
    return _sing(toks[-1]) if toks else ""


@functools.lru_cache(maxsize=None)
def _wordnet_relation(a: str, b: str) -> Optional[str]:
    """``synonym`` if the nouns share a synset (top 3 senses), ``hypernym`` if one
    is within 3 hypernym levels of the other (generic synsets excluded)."""
    from nltk.corpus import wordnet as wn

    if not a or not b:
        return None
    if a == b:
        return "same"
    sa, sb = wn.synsets(a, pos=wn.NOUN)[:3], wn.synsets(b, pos=wn.NOUN)[:3]
    if not sa or not sb:
        return None
    if set(sa) & set(sb):
        return "synonym"
    for x, y in ((sa, sb), (sb, sa)):
        names = {s.name() for s in y}
        for s in x:
            hyp, frontier = set(), [s]
            for _ in range(3):
                frontier = [h for f in frontier for h in f.hypernyms() + f.instance_hypernyms()]
                hyp.update(h.name() for h in frontier)
            if (hyp & names) - _GENERIC_SYNSETS:
                return "hypernym"
    return None


@functools.lru_cache(maxsize=None)
def _variant(word: str, gts: Tuple[str, ...]) -> Tuple[Optional[str], Optional[str]]:
    ws, wh = _sing(word), _head(word)
    if word in gts:
        return "exact", word
    for g in gts:
        if _sing(g) == ws:
            return "plural", g
    single = " " not in word
    for g in gts:
        if _head(g) == wh or (single and ws in {_sing(t) for t in re.findall(r"[a-z]+", g)}):
            return "phrase", g
    best = (None, None)
    for g in gts:
        r = _wordnet_relation(wh, _head(g))
        if r == "synonym":
            return "synonym", g
        if r == "hypernym" and best[0] is None:
            best = ("hypernym", g)
    return best


def variant_of_gt(word: str, gt_names: List[str]) -> Tuple[Optional[str], Optional[str]]:
    """How ``word`` relates to the closest name in ``gt_names``: ``exact``,
    ``plural``, ``phrase`` (shared head noun), ``synonym`` or ``hypernym``
    (WordNet), else None; with the matched name."""
    return _variant(word.lower().strip(), tuple(sorted(g.lower().strip() for g in gt_names)))
