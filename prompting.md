# Prompting rules (Fizgig, H3 and Klein)

Use [Fizgig's editing/reference pattern](fizgig.md) first for image repairs and reference-guided stills,
regardless of medium. Load [custom art/animation](art-animation.md) only for requested stylized animation
or explicit 3D blocking/motion transfer; action alone does not require a Blender pipeline.
- **Cast count:** state the exact number and placement of visible people in each generated scene.
- **Speech uses MiniMax's official prompt format** (see SKILL.md §7): `<d>[English] …</d>` around the spoken words
  only, `(S1)` speaker IDs, `<Audio 1>` as a timbre reference "without copying the original signal", and a take
  length that fits the line. Quoted lines in free prose get the prompt text itself spoken aloud as babble.

- **Likeness from real photos, framing from the still (ref mode):** give each real person 2–3 real photos as the
  subject `ref` list, and pass the keyframe still as `framing`, not as first/last-frame guides. Guides pin the
  generated still's pixels, face included; a framing ref only steers camera, set and placement.
  Example: `"ref": ["refs/host.png", "refs/host_2.png", "refs/host_3.png"]` plus `"framing": "stills/k01.png"`.
  The prompt must still describe the set and layout in words ("cream wall, shelf of model kits behind; he sits
  in the left half, framed from the chest up"), or H3 takes the background from the likeness photos.
- **Framing lock (fl mode):** set `last_frame` to the same still as `first_frame`; text alone never holds framing.
- **Colour and negation priming:** any colour or object named gets attached to something; describe positively.
  Example: "dark brown eyes, bare face" instead of "no green, no glasses". A "clear desk" came out as a glass
  desk ("a bare wooden desk top" worked), and "webcam framing" put a webcam in the shot ("seen straight on at
  eye level from the viewer's seat").
- **Klein merges a character with a similar prop:** a felt puppet next to a potato-figure prop became half potato.
  Say what stays theirs. Example: "The puppet keeps his own felt face, yarn hair, blazer and T-shirt; only the
  host holds the potato figure."
- **An adult squeezed into a small prop shrinks to fit it.** Example: "The host, a full-size adult man, …
  comically far too big for the tiny toddler car: his head and shoulders stick out above the roof."
- **Relative scale in the still prompt; never a character sheet as a ref for a prop** (its close-up scale leaks).
  Example: "a figurine no bigger than her hand", refs `["@maker", "@buyer"]` without `@figurine`.
- **Klein duplicates people when a character sheet is a ref:** use a single-pose reference and sweep seeds.
  Example: "Only this one puppet." plus seeds +1..+3 when two appear.
- **Action: stage the aim in the still, not the muzzle flash; fixed camera; target "straight in front of the gun".**
  Example: "[2s-4s] the target straight in front of her gun bursts apart; the camera stays fixed".
- **State transitions, not vague action:** specify initial/final pose, occlusion, camera scale and retained
  aftermath for each beat. Weapon lifts are fast and purposeful; for drawn animation, use one coherent
  continuous drawing for difficult prop motion rather than local animated patches that can pop in.
- **A/B any LoRA per shot, never film-wide.** The Jojocodex action LoRAs broke 2 of 3 shots on a sci-fi action short.
  Example: `"loras": [["minimax_h3_wushu_action_v5_fl2va.safetensors", 0.5]]` on one shot, compared against none.
- **Speech starts at 0 s in ref mode, whatever the prompt says** ("the first second is silent" is ignored). Never
  tell the speaker to hold still at the start: the opening words then go to another mouth in frame. One exact
  line per shot; VibeVoice garbles one-word lines, use a short phrase.
  Example: "The host talks straight to the webcam from the first frame to the last… The host, the man on the
  left, in the voice of <Audio 1>, says exactly one line and nothing else: \"…\" No other voices."
  Naming the speaker's position ("the man on the left", "the felt puppet") helps H3 pick the right mouth.
- **Scale jumps between shots:** keep the camera still and let the subject move.
  Example: "Locked-off wide shot; the robot walks toward the camera until it fills the frame."
- **Two characters in frame: only one speaks per shot; describe the other as silent.**
  Example: "The puppet speaks… The host stays silent, holding his pose."
- **A silent character with an obvious mouth (a puppet) keeps moving their lips during the other one's line**,
  mostly at the start but also mid-shot, and often only half-open. Seeds, prompts, dropping them as a subject
  and separate "listening" renders all failed. Fix it in the edit with a hold:
  `python tools/mouth_hold.py film/<f>/shots.json --ids h10 --box <x0,y0,x1,y1 around their head>` writes
  `"hold": {"from", "dur", "crop"}`. That side then shows a 2 s mouth-shut stretch of the same render, looped
  forward and backward. Check the chosen stretch by eye: red props in the box fool the metric.
  Also give the silent one a closed-mouth reference (a Klein edit of the photo), and keep "mouth" out of the
  global style.
- **Short lines in a 5 s shot get padded with invented words** (the line comes first, then babble). Say
  "That one line is the only speech in the shot; after it, silence…", and trim what's left with `"out"`.
- **Hands:** ask for "natural human hands with five fingers each", and stage keyframes with hands resting or
  holding a prop; raised counting hands come out with four fingers.
- **Shots where nobody speaks (B-roll of a character walking, working): give H3 a music-only soundtrack.** With an
  ambience line ("footsteps, distant traffic. No speech, no voices.") H3 invented speech and lip-synced it; with
  "The soundtrack is a solo fiddle playing a slow waltz, and nothing else" the mouths stayed shut.
  Also use a silent subject variant whose only pictures are closed-mouth stills ("his mouth a flat shut seam"), and
  drop the shot's audio in the cut (`"bed": "none"`). ASR on such a render hallucinates text on the music ("the
  song is a tribute to…"); judge the mouth by frames, not by the transcript.
- **Never use a full scene photo as a subject picture.** H3 plays it back as a scene: in one film six ref
  shots cut to a courtroom-hearing photo for a second or more. Subjects get character pictures only; put a scene
  photo in as `first_frame` (a guide) when the shot should start from it. A tight portrait crop of a scene still
  counts as a scene photo too: one episode cut mid-shot to an artisan's shelf-background crop and to an actor's
  film stills until those subjects used only character-sheet crops on white.
- **A ref shot can open on one of its subject pictures** (a grey-backdrop portrait for the first second), or
  dissolve in from it. Pin the opening with `first_frame` (the keyframe still) as well as `framing`; in one episode that
  cleared almost every one. A desk-and-monitor scene in a "vlog" setting kept flashing to an invented webcam
  face even pinned: move screens to motion graphics and keep the person shot away from the monitor setup.
- **Close-up B-roll must not carry the whole-room description.** "A hobbyist's studio: shelves, ring light,
  two monitors, a map…" + "Close-up of the hard drive" made fl shots dissolve into the room and back, even with
  first = last frame. Describe only what the keyframe shows and end with "One continuous shot: the camera stays on
  this view and never cuts away."
- **A style names a place, and H3 goes there.** The `theatre` style ("inside a grand opera house") turned an
  opera-house exterior into a stage interior; exteriors need their own style variant.
- **Keyframe costume lives in the prompt.** Klein copied the identity photos' jacket onto the tuxedo and helmet
  variants until the still prompt said "a borrowed black tuxedo jacket a size too big…" / "a borrowed olive
  helmet over his beanie".
- **Voice references leak their words.** H3 sometimes speaks the reference clip's transcript before the line
  ("inside every dog, keep watching" from an artisan's reference of "Inside every doll…"): trim with `in`, and
  build references of ≥ 8 s from several approved lines so no single phrase dominates.
- **Emotion in H3 speech** (trial: 3 emotions × 7 prompt variants × 3 seeds, scored with audeering
  arousal/valence, CLAP, pitch spread and voice similarity; a throwaway script renders the variants and scores
  each take's vocal stem with those four measures):
  - "says in an angry tone:" barely moves the read. Write an emotional *verb + delivery* outside `<d>` ("laughs out
    loud as he says, his voice bubbling with genuine laughter"; "shouts furiously, his voice loud, harsh and
    shaking with rage"): laughter went from CLAP 0.59 to 0.96 at almost no timbre cost.
  - Anger built up with each layer: + the face/body acting it (picture and voice are generated together) + the
    non-verbal beat as its own event ("slams his fist on the desk… and shouts") and in `overall_soundscape` +
    binding the voice reference to *timbre and accent only* (our default binding also says "pitch", which pins the
    calm reference's pitch): CLAP angry 0.29 → 0.68, arousal 0.58 → 0.71, similarity 0.97 → 0.93.
  - The voice reference caps arousal: excitement stayed at 0.53–0.57 whatever the prompt; without a reference
    H3 reaches 0.84–0.90 but loses the voice (similarity ~0.70). Two-pass fix: render the line with no voice
    reference, then use that take as `<Audio 2>` "delivery reference (emotion, energy, pitch contour, pacing, not
    its voice)" next to the timbre reference: arousal 0.87–0.93 at similarity 0.80–0.83. Listen before using it:
    excited two-pass takes scored low valence (frantic rather than happy).

