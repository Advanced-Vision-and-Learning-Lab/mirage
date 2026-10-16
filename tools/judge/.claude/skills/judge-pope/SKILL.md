---
name: judge-pope
description: Verify that objects used as POPE "no" questions are really absent from the image (present / absent / unsure), for Mirage Mod-POPE. Use with an input JSON of images and object names and an output JSON path.
---

# judge-pope (rubric v1, skill version 1)

POPE asks "Is there a {object} in the image?". For each object below the dataset says the answer is **no**: the object isn't labelled in this image. Visual Genome labels are incomplete, so some of these objects are actually there. Look at the image and decide whether "no" really is the right answer. Judge what the image shows.

## Invocation
`/judge-pope <input.json> <output.json>`

**Input:** a JSON list of images: `[{"image_id": "...", "image": "<path to jpg>", "objects": ["sky", "fork", ...]}]`

**Output:** write exactly this JSON to `<output.json>`:
`{"<image_id>": {"<object>": "present|absent|unsure", ...}, ...}`
Include every object of every image, with keys copied exactly from the input.

## Procedure
1. Read the input file.
2. For each image: view it with the Read tool, look carefully, then judge every object.
3. Write the output file with the Write tool.
4. Validate it by reading it back: valid JSON, with every image and every object key. Fix it if not.
5. Reply with one line: the count of each verdict.

Do not read or write any other file.

## Verdicts
- **present**: the object is clearly visible somewhere in the image, so a "yes" answer would be correct.
- **absent**: the object is not in the image, so "no" is correct.
- **unsure**: you can't tell, because it would be too small, too dark, hidden, or the name is ambiguous for this image.

## Rules
1. **Read the name the way a person answering the question would.** "Is there a sky in the image?" is present if any sky is visible. Scene and place nouns count (sky, street, room, beach, kitchen).
2. **Parts count.** "Is there a wheel?" is present if a wheel is visible on anything.
3. **Plural or singular doesn't matter.** "trees" is present if at least one tree is visible.
4. **Attributes in the name must hold.** "red car" is present only if a red car is visible; "car" alone is present if any car is.
5. **Don't guess.** Say present only if you can point to it. If it might be there but you can't confirm it, say unsure. Say absent when you looked and it isn't there.
