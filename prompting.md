# Prompting rules (H3 + Klein)

- **Likeness from real photos, framing from the still (ref mode):** give each real person 2–3 real photos as the
  subject `ref` list, and pass the keyframe still as `framing`, not as first/last-frame guides. Guides pin the
  generated still's pixels, face included; a framing ref only steers camera, set and placement.
  Example: `"ref": ["refs/host.png", "refs/host_2.png", "refs/host_3.png"]` plus `"framing": "stills/k01.png"`.
  The prompt must still describe the set and layout in words ("brick wall, tall bookshelf behind; he sits
  in the left half, framed from the chest up"), or H3 takes the background from the likeness photos.
- **Framing lock (fl mode):** set `last_frame` to the same still as `first_frame`; text alone never holds framing.
- **Colour and negation priming:** any colour or object named gets attached to something; describe positively.
  Example: "dark brown eyes, bare face" instead of "no green, no glasses". A "clear desk" came out as a glass
  desk ("a bare wooden desk top" worked), and "webcam framing" put a webcam in the shot ("seen straight on at
  eye level from the viewer's seat"). "Large-format camera" in a style block put cinema cameras in half the stills.
- **Names become text:** naming a character in a prompt for a screen or sign ("CCTV footage of <Name>") prints the
  name on the screen. Describe them instead ("the young man from the reference photo, short dark hair, denim jacket").
- **Klein merges a character with a similar prop:** a puppet next to a novelty toy became half toy. Say what stays
  theirs. Example: "The puppet keeps its own face, hair and clothes; only the host holds the toy."
- **An adult squeezed into a small prop shrinks to fit it.** Example: "<Name>, a full-size adult man, …
  comically far too big for the tiny toddler car: his head and shoulders stick out above the roof."
- **Relative scale in the still prompt; never a character sheet as a ref for a prop** (its close-up scale leaks).
  Example: "a doll no bigger than her hand", refs `["@hero", "@park"]` without `@doll`.
- **Klein duplicates people when a character sheet is a ref:** use a single-pose reference and sweep seeds.
  Example: "Only this one puppet." plus seeds +1..+3 when two appear.
- **Real-world objects the model doesn't know:** give 1–2 real photos as refs, describe the object's parts, say
  what it is not ("no seat, no pedals, no chain, no metal frame"), and add "the person in that photo is only a
  reference for the object and does not appear."
- **Action: stage the aim in the still, not the muzzle flash; fixed camera; target "straight in front of the gun".**
  Example: "[2s-4s] the drone straight in front of her gun bursts apart; the camera stays fixed".
- **A/B any LoRA per shot, never film-wide.** Action LoRAs broke 2 of 3 shots in one test.
  Example: `"loras": [["<action_lora>.safetensors", 0.5]]` on one shot, compared against none.
- **Speech starts at 0 s in ref mode, whatever the prompt says** ("the first second is silent" is ignored). Never
  tell the speaker to hold still at the start: the opening words then go to another mouth in frame. One exact
  line per shot; VibeVoice garbles one-word lines, use a short phrase.
  Example: "<Name> talks straight to the camera from the first frame to the last… <Name>, the man on the
  left, in the voice of <Audio 1>, says exactly one line and nothing else: \"…\" No other voices."
  Naming the speaker's position ("the man on the left", "the felt puppet") helps H3 pick the right mouth.
- **Scale jumps between shots:** keep the camera still and let the subject move.
  Example: "Locked-off wide shot; the robot walks toward the camera until it fills the frame."
- **Two characters in frame: only one speaks per shot; describe the other as silent.**
  Example: "The puppet speaks… <Name> stays silent, holding his pose."
- **Reverse angles for a second speaker:** "Over-the-shoulder reverse shot: the back of <Name>'s head and
  shoulder soft in the left foreground; facing him, <local> … talks from the first frame to the last. <Name> is
  seen from behind, still and silent, his face turned away." Give Klein the host's photo as a ref so the back of
  the head matches.
- **A silent character with an obvious mouth (a puppet) keeps moving their lips during the other one's line**,
  mostly at the start but also mid-shot, and often only half-open. Seeds, prompts, dropping them as a subject
  and separate "listening" renders all failed. Fix it in the edit with a hold:
  `python tools/mouth_hold.py film/<f>/shots.json --ids s10 --box <x0,y0,x1,y1 around their head>` writes
  `"hold": {"from", "dur", "crop"}`. That side then shows a 2 s mouth-shut stretch of the same render, looped
  forward and backward. Check the chosen stretch by eye: red props in the box fool the metric.
  Also give the silent one a closed-mouth reference (a Klein edit of the photo), and keep "mouth" out of the
  global style.
- **Short lines in a 5 s shot get padded with invented words** (the line comes first, then babble), or read
  slowly across the whole shot. Say "That one line is the only speech in the shot; after it, silence…", and trim
  what's left with `"out"`.
- **Hands:** ask for "natural human hands with five fingers each", and stage keyframes with hands resting or
  holding a prop; raised counting hands come out with four fingers. A character "resting one hand on" a large
  prop can leave a disembodied hand on the prop once they move; put their hands out of frame instead.
