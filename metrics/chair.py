import logging
import functools
from typing import Dict, List, Any, Set, Tuple, Optional
from collections import Counter

import nltk
import inflect

from .base_metric import BaseMetric

logger = logging.getLogger(__name__)

# Ensure NLTK tokenizer data is available
try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt_tab', quiet=True)

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
                canonical CHAIR behaviour; the offline COCO rescore turns it on.
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
