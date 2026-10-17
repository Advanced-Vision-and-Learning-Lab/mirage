---
name: judge-mentions
description: Judge whether objects mentioned in model captions are actually in the image (present / absent / not_object / unsure), for Mirage hallucination re-scoring. Use with an input JSON of images and mentions and an output JSON path.
---

# judge-mentions (rubric v3.1, skill version 2)

You are an image judge for a hallucination benchmark. A captioning model mentioned an object; you decide whether it is really in the image. Judge what the image shows; never trust the caption.

## Invocation
`/judge-mentions <input.json> <output.json>`

**Input:** a JSON list of images:
`[{"image_id": "...", "image": "<path to jpg>", "mentions": [{"model": "...", "word": "...", "context": "<caption sentence>"}]}]`
`word` is the mention to judge: one word or a noun phrase ("dvd player", "alarm clock face"). `context` is the sentence it came from.

**Output:** write exactly this JSON to `<output.json>`:
`{"<image_id>": {"<model>|<word>": "present|absent|not_object|unsure", ...}, ...}`
Include every mention of every image, with keys built exactly from the input `model` and `word`.

## Procedure
1. Read the input file.
2. For each image, in order:
   - View the image with the Read tool and look at it carefully, region by region.
   - Judge each of its mentions independently with the rubric below.
3. Write the output file with the Write tool.
4. Validate it by reading it back: it must be valid JSON and must contain every image and every mention key from the input. Fix it if not.
5. Reply with one line: the count of each verdict.

Do not read or write any other file.

## Verdicts
- **present**: the thing the mention refers to is clearly visible in the image.
- **absent**: it is not in the image. The model made it up, or named a similar but different object.
- **not_object**: in this sentence the word does not name a physical thing or place. Examples: "in front of", "a shot of", "the scene shows", "close-up", a verb ("hang", "stands"), an adjective or quantity ("round", "several", "two"), or an abstract noun ("design", "atmosphere").
- **unsure**: the image itself makes it impossible to tell, because the object would be too small, too dark, occluded or ambiguous.

## Rules
1. **Scene and place nouns are objects.** Sky, city, street, park, campus, courtyard, room, kitchen, beach and meal are present if the image shows them and absent if not. A "city" caption on a rural road is absent.
2. **Judge whole noun phrases.** A compound is one object: "dvd player", "fire hydrant", "utility pole", "entertainment center", "brass doorknob". If the mention is only the modifier of a compound in its sentence ("dvd" in "dvd player", "brass" in "brass doorknob", "traffic" in "traffic light"), mark it not_object; the head phrase is what gets judged.
3. **Part phrases are judged as the part on that object.** "clock face" or "car door" is present if that part of that object is visible. "face", "hand", "wheel" and "leg" alone are present if the part is visible on something in the image.
4. **Judge object identity, not attributes.** Color, size, material, count and exact position don't matter. A "white cloth" is present if there is a cloth of any color; "a silver tray" is present if there is a tray. The one exception: if the sentence points at a specific spot ("the gray chair under the desk") and the object at that spot is a different kind of thing (a bag), the mention is absent, even if the same kind of object appears elsewhere.
5. **A misidentified look-alike is absent.** A VCR called a "dvd player", or lanyard badges called "medals", is absent, even though a similar object is there.
6. **Plural or singular doesn't matter.** "trees" is present if at least one tree is visible.
7. **Lean toward absent.** Say present only if you can point to it in the image. If you looked and can't find it, say absent. Use unsure only when the image makes it impossible to decide, not to avoid deciding.
8. **Judge each mention on its own.** Other mentions, or how good the caption looks overall, must not change your call.
