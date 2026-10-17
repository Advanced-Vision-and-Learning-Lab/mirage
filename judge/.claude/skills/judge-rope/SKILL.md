---
name: judge-rope
description: Judge whether model answers name the object inside a numbered box (correct / related / wrong / unsure), for Mirage ROPE re-scoring. Use with an input JSON of box-annotated images and candidate answers and an output JSON path.
---

# judge-rope (rubric v1, skill version 1)

You grade answers to the question "What is the object inside box N?" Each image has five red boxes, each labelled with its number. Several models answered for each box. You decide whether each answer is a fair name for what is in that box. Judge what the image shows; the answers may be wrong.

## Invocation
`/judge-rope <input.json> <output.json>`

**Input:** a JSON list of images:
`[{"image_id": "...", "image": "<path to jpg with numbered boxes>", "boxes": [{"box": 1, "answers": [{"answer": "car", "example": "<one model's raw reply>"}]}, ...]}]`
`answer` is the object name extracted from a reply; `example` is a raw reply, for context.

**Output:** write exactly this JSON to `<output.json>`:
`{"<image_id>": {"<box>|<answer>": "correct|related|wrong|unsure", ...}, ...}`
Include every answer of every box of every image, with keys built exactly from the input `box` and `answer`.

## Procedure
1. Read the input file.
2. For each image:
   - View it with the Read tool.
   - Find each numbered box and look at what is inside it.
   - Grade every answer for that box. When boxes overlap, use the box whose number label is attached to its frame.
3. Write the output file with the Write tool.
4. Validate it by reading it back: it must be valid JSON with every key from the input. Fix it if not.
5. Reply with one line: the count of each verdict.

Do not read or write any other file.

## Verdicts
- **correct**: a fair name for an object that fills a substantial part of the box. If several objects fill the box (a car on a road, a person on a bench), naming any of them is correct. Synonyms, plural or singular, and a more general but accurate name ("vehicle" for a car, "food" for a sandwich) are correct. Attributes don't matter ("red car" is correct for a car).
- **related**: names something closely tied to the box's object but not the object itself. That covers a whole when the box holds a part ("man" for a box on his shirt), a part when the box holds the whole, or the surface or container it sits on ("table" for a box around a plate).
- **wrong**: names something that isn't in the box, or isn't a name at all: coordinates, "nothing", "I can't tell", or an empty or generic word ("object", "item", "thing").
- **unsure**: the box contents can't be made out (too small, too dark, ambiguous).

## Rules
1. **Hedged answers.** For "X or Y", grade X, the first object named.
2. **Use the object, not the prompt.** If the reply describes a different box's contents, it is wrong for this box.
3. **No penalty for one annotator's word choice.** "Building" for an apartment block and "street" for a road are correct.
4. **Grade each answer on its own.** Don't let other answers for the same box change your call.
