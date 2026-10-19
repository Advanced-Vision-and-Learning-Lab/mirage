---
name: judge-pope
description: Verify that objects used as POPE "no" questions are really absent from the image (present / absent / unsure), for Mirage Mod-POPE. Use with an input JSON of images and object names and an output JSON path.
---

# judge-pope

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
- **unsure**: a real object that could be in this scene, where you can't confirm it either way because the place it would be is too small, too dark, blurred or hidden. Use it rarely.

## Rules
1. **Read the name the way a person answering the question would.** "Is there a sky in the image?" is present if any sky is visible. Scene and place nouns count (sky, street, room, beach, kitchen).
2. **Parts count.** "Is there a wheel?" is present if a wheel is visible on anything.
3. **Plural or singular doesn't matter.** "trees" is present if at least one tree is visible.
4. **Attributes in the name must hold.** "red car" is present only if a red car is visible; "car" alone is present if any car is.
5. **Don't guess present.** Say present only if you can point to it.
6. **Read vague names generously.** Visual Genome names are often region descriptions ("wall panel", "worn step", "lawn grass", "top part"). If a person could point at something in this image and call it that, say present.
7. **Absent when nothing fits.** Say absent when you looked and nothing in the image matches the name. This includes names that pick out nothing: a fragment ("la", "sd"), a typo you can't read as an object, or a colour or material that nothing visible has. A person asked "Is there a sd in the image?" would say no.
8. **Unsure is rare.** Use it only for a real object that could be in this scene where the place it would be is too small, too dark, blurred or hidden to check. A vague or odd name is never a reason for unsure.
